from __future__ import annotations

import gzip
import io
import importlib.util
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evolution_sim.cli import (
    collect_trajectory,
    mind_fixture_labels,
    mind_horizon_labels,
    mind_v3_neural_artifact,
    mind_artifact_diagnostics,
    mind_gate,
    mind_v3_evaluate,
    mind_policy_eval,
    mind_train,
    run_headless,
)
from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import OBSERVATION_INPUT_VECTOR_SIZE
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind import torch_trainer as mind_torch_trainer
from evolution_sim.mind.artifacts import (
    MindArtifactError,
    validate_model_artifact_manifest,
    write_model_artifact,
)
from evolution_sim.mind.baseline import (
    train_baseline_with_trainer,
    train_behavior_cloning_baseline,
    train_reward_weighted_behavior_cloning_baseline,
)
from evolution_sim.mind.contracts import (
    MIND_MODEL_ARTIFACT_VERSION,
    MIND_ONLINE_LEARNING_CONTRACT_VERSION,
    MIND_RUNTIME_ENABLED_DEFAULT,
    MIND_V1_DATA_CONTRACT_VERSION,
    mind_online_learning_contract,
    mind_v1_data_contract,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
    TrajectoryDatasetError,
    TrajectoryJsonlDataset,
    build_trajectory_transitions,
    combined_dataset_provenance,
    dataset_provenance,
    discounted_return_targets,
    load_trajectory_jsonl,
    records_with_trajectory_context,
)
from evolution_sim.mind.carrion_counterfactual_labels import (
    MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
)
from evolution_sim.mind.diagnostics import (
    build_artifact_diagnostics,
    build_artifact_diagnostics_shard_stats,
    build_policy_diagnostics,
    finalize_artifact_diagnostics_shards,
)
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.feature_policy import feature_keys_from_observation
from evolution_sim.mind.gates import build_mind_v1_gate_report
from evolution_sim.mind.fixture_labels import (
    MIND_FIXTURE_LABEL_SCHEMA_VERSION,
    build_fixture_label_report,
)
from evolution_sim.mind.horizon_labels import (
    MIND_HORIZON_LABEL_SCHEMA_VERSION,
    build_horizon_label_records,
    build_horizon_label_report,
    parse_horizon_ticks,
)
from evolution_sim.mind.policy_inputs import (
    CONTROLLER_DIAGNOSTIC_INPUT_FIELDS,
    ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
    ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
    ecological_policy_input_contract,
    ecological_policy_values_from_decoded,
)
from evolution_sim.mind.v3_neural import (
    MIND_V3_HORIZON_FIXTURE_MODEL_TYPE,
    MIND_V3_HORIZON_FIXTURE_SCORE_POLICY,
    MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
    MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_BIAS_POLICY,
    MIND_V3_NEURAL_INPUT_POLICY,
    load_mind_v3_neural_artifact,
    score_mind_v3_neural_artifact,
    train_mind_v3_neural_artifact,
)
from evolution_sim.mind.learned_policy import (
    OnlineAdaptiveMindPolicy,
    LearnedPolicy,
    load_learned_policy,
    replay_online_update_traces,
)
from evolution_sim.mind.viability import (
    VIABILITY_COMPONENT_NAMES,
    VIABILITY_SUPPRESSION_COMPONENT,
)
from evolution_sim.mind.splits import deterministic_seed_split


class MindV1Tests(unittest.TestCase):
    def _write_tiny_trajectory(self, path: Path, *, seed: int = 7) -> None:
        writer = JsonlTrajectoryWriter(
            path,
            source_seeds=[seed],
            split_id="tiny_train",
        )
        SimulationWorld(WorldConfig(seed=seed, max_ticks=2)).run(
            mode=RunMode.SUMMARY_ONLY,
            trajectory_sink=writer,
        )

    def test_mind_v1_contract_declares_current_runtime_schemas_disabled_by_default(self) -> None:
        contract = mind_v1_data_contract()

        self.assertEqual(contract["contract_version"], MIND_V1_DATA_CONTRACT_VERSION)
        self.assertFalse(contract["runtime_enabled_by_default"])
        self.assertFalse(MIND_RUNTIME_ENABLED_DEFAULT)
        self.assertEqual(contract["model_artifact_version"], MIND_MODEL_ARTIFACT_VERSION)
        self.assertIn("trajectory_record_fields", contract)
        json.dumps(contract)

    def test_mind_online_learning_contract_declares_safe_ladder(self) -> None:
        contract = mind_online_learning_contract()

        self.assertEqual(
            contract["contract_version"],
            MIND_ONLINE_LEARNING_CONTRACT_VERSION,
        )
        self.assertFalse(contract["runtime_enabled_by_default"])
        self.assertFalse(contract["online_weight_updates_enabled_by_default"])
        self.assertFalse(contract["in_simulation_weight_updates_allowed"])
        self.assertEqual(
            contract["current_executable_slice"],
            "torch_discrete_iql_artifact_v1",
        )
        self.assertIn(
            "learned_policy_trajectory_collection_v1",
            contract["executable_slices"],
        )
        self.assertIn(
            "neural_actor_critic_bc_artifact_v1",
            contract["executable_slices"],
        )
        self.assertIn(
            "torch_actor_critic_bc_artifact_v1",
            contract["executable_slices"],
        )
        self.assertIn(
            "torch_advantage_actor_critic_bc_artifact_v1",
            contract["executable_slices"],
        )
        self.assertIn(
            "torch_discrete_iql_artifact_v1",
            contract["executable_slices"],
        )
        self.assertIn(
            "heuristic_free_autonomous_controller_v1",
            contract["executable_slices"],
        )
        self.assertIn(
            "in_run_contextual_bandit_adapter_v1",
            contract["executable_slices"],
        )
        self.assertTrue(
            contract["trajectory_collection"]["requires_explicit_mind_enable"]
        )
        self.assertEqual(
            contract["trajectory_collection"]["runtime_modes"],
            ["guarded", "autonomous", "autonomous-online"],
        )
        self.assertFalse(contract["autonomous_controller"]["enabled_by_default"])
        self.assertFalse(contract["autonomous_controller"]["heuristic_guard"])
        self.assertFalse(contract["autonomous_controller"]["heuristic_delegate"])
        self.assertFalse(contract["online_in_run_adapter"]["enabled_by_default"])
        self.assertFalse(contract["online_in_run_adapter"]["neural_weight_updates"])
        self.assertEqual(
            contract["online_in_run_adapter"]["policy"],
            "in_run_contextual_bandit_adapter_v1",
        )
        self.assertEqual(
            contract["online_in_run_adapter"]["update_trace_schema_version"],
            "mind_policy_update_trace_v1",
        )
        self.assertEqual(
            contract["trajectory_collection"][
                "policy_decision_diagnostics_opt_in"
            ],
            "--include-policy-diagnostics",
        )
        self.assertEqual(
            contract["trajectory_collection"]["policy_update_trace_opt_in"],
            "--include-policy-update-trace",
        )
        self.assertFalse(
            contract["trajectory_collection"][
                "diagnostic_payload_required_by_default"
            ]
        )
        self.assertEqual(
            contract["neural_actor_critic"]["trainer"],
            "neural-actor-critic-bc",
        )
        self.assertEqual(
            contract["torch_actor_critic"]["trainer"],
            "torch-actor-critic-bc",
        )
        self.assertEqual(
            contract["torch_actor_critic"]["third_party_ml_dependency"],
            "requirements-mind-ml.txt",
        )
        self.assertEqual(
            contract["torch_advantage_actor_critic"]["trainer"],
            "torch-advantage-actor-critic-bc",
        )
        self.assertEqual(
            contract["torch_advantage_actor_critic"]["model_type"],
            "guarded_torch_advantage_actor_critic_bc_v1",
        )
        self.assertEqual(
            contract["torch_discrete_iql"]["trainer"],
            "torch-discrete-iql",
        )
        self.assertEqual(
            contract["torch_discrete_iql"]["model_type"],
            "guarded_torch_discrete_iql_v1",
        )
        counterfactual_supervision = contract["torch_discrete_iql"][
            "counterfactual_label_supervision"
        ]
        self.assertFalse(counterfactual_supervision["enabled_by_default"])
        self.assertEqual(
            counterfactual_supervision["label_schema_version"],
            MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
        )
        self.assertEqual(
            counterfactual_supervision["policy"],
            "carrion_counterfactual_terminal_action_value_supervision_v1",
        )
        self.assertFalse(
            contract["torch_advantage_actor_critic"][
                "inference_requires_third_party_ml_dependency"
            ]
        )
        self.assertFalse(
            contract["torch_actor_critic"][
                "inference_requires_third_party_ml_dependency"
            ]
        )
        self.assertFalse(
            contract["neural_actor_critic"]["weights_mutable_during_run"]
        )
        stages = {stage["stage"] for stage in contract["algorithm_ladder"]}
        self.assertIn("neural_behavior_cloning_actor_critic", stages)
        self.assertIn("pytorch_behavior_cloning_actor_critic", stages)
        self.assertIn("conservative_offline_rl", stages)
        self.assertIn("discrete_iql_actor_critic", stages)
        self.assertIn("offline_to_online_finetuning", stages)
        self.assertIn("open_ended_population_search", stages)
        self.assertIn("world_model_control", stages)
        self.assertTrue(contract["promotion_gates"]["zero_per_seed_alive_regression"])
        json.dumps(contract)

    def test_mind_v3_autonomous_evolution_contract_declares_no_heuristic_fallback(
        self,
    ) -> None:
        from evolution_sim.mind.contracts import (
            mind_v3_autonomous_evolution_contract,
        )

        contract = mind_v3_autonomous_evolution_contract()

        self.assertEqual(
            contract["contract_version"],
            "mind_v3_autonomous_evolution_contract_v1",
        )
        self.assertFalse(contract["enabled_by_default"])
        self.assertEqual(
            contract["policy"],
            "mind_v3_autonomous_evolution_policy_v1",
        )
        self.assertEqual(
            contract["action_selection"],
            "inherited_controller_masked_argmax_v1",
        )
        self.assertFalse(contract["heuristic_guard"])
        self.assertFalse(contract["heuristic_delegate"])
        self.assertFalse(contract["heuristic_action_selection"])
        self.assertTrue(contract["action_mask_required"])
        self.assertEqual(
            contract["learning_mechanism"],
            "bounded_parental_inheritance_with_mutation_v1",
        )
        self.assertEqual(
            contract["mind_state_storage"],
            "agent.mind_inheritance_metadata",
        )
        self.assertEqual(
            contract["promotion_metric_family"],
            "autonomous_survival_reproduction",
        )
        self.assertEqual(
            contract["controller"]["feature_scope"],
            "policy_visible_self_local_patch_navigation",
        )
        self.assertIn(
            "local_patch.food",
            contract["controller"]["feature_fields"],
        )
        self.assertIn(
            "navigation.water",
            contract["controller"]["feature_fields"],
        )
        self.assertIn(
            "thirst_x_navigation.water.dy",
            contract["controller"]["derived_feature_fields"],
        )
        self.assertNotIn(
            "mind_inheritance_available",
            contract["controller"]["feature_fields"],
        )
        self.assertIn(
            "mind_inheritance_available",
            contract["controller"][
                "controller_private_fields_excluded_from_features"
            ],
        )
        self.assertEqual(
            contract["frozen_neural_artifact"]["schema_version"],
            MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(
            contract["frozen_neural_artifact"]["input_contract"][
                "schema_version"
            ],
            ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        )
        self.assertFalse(
            contract["frozen_neural_artifact"]["weights_mutable_during_run"]
        )
        self.assertEqual(
            contract["experimental_direct_policy_artifact"]["model_type"],
            MIND_V3_HORIZON_FIXTURE_MODEL_TYPE,
        )
        self.assertFalse(
            contract["experimental_direct_policy_artifact"][
                "linear_anchor_required"
            ]
        )
        json.dumps(contract)

    def test_mind_v3_founder_metadata_is_bounded_and_deterministic(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import (
            MIND_V3_CONTROLLER_SCHEMA_VERSION,
            founder_mind_v3_metadata,
            mind_v3_parameter_count,
        )

        first = founder_mind_v3_metadata(agent_id=7, rng=Random(123))
        second = founder_mind_v3_metadata(agent_id=7, rng=Random(123))

        self.assertEqual(first, second)
        self.assertEqual(
            first["schema_version"],
            MIND_V3_CONTROLLER_SCHEMA_VERSION,
        )
        self.assertTrue(first["inherited_state"])
        self.assertEqual(first["state_size"], mind_v3_parameter_count())
        self.assertLessEqual(first["state_size"], 512)
        self.assertEqual(
            first["architecture"],
            "need_gated_local_navigation_feature_projection_linear_action_head_v4",
        )
        self.assertEqual(
            first["founder_prior_policy"],
            "diverse_need_gated_navigation_action_prior_v3",
        )
        self.assertIn(
            first["specialization_profile"],
            {
                "forager",
                "hydration_seeker",
                "disperser",
                "reproducer",
                "scavenger",
                "predator_scavenger",
            },
        )
        self.assertIn("action_head_weights", first)
        self.assertIn("action_head_bias", first)

    def test_mind_v3_founder_initialization_preserves_behavioral_diversity(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        profiles = {
            str(
                founder_mind_v3_metadata(agent_id=agent_id, rng=Random(123))[
                    "specialization_profile"
                ]
            )
            for agent_id in range(12)
        }

        self.assertGreaterEqual(len(profiles), 3)

    def test_mind_v3_homeostatic_projection_separates_resource_needs(
        self,
    ) -> None:
        from evolution_sim.mind.evolution import (
            MIND_V3_CONTROLLER_SCHEMA_VERSION,
            mind_v3_parameter_count,
            score_mind_v3_metadata,
        )

        weights = {
            action: [0.0] * 8
            for action in ACTION_NAMES
        }
        weights["eat"][0] = 1.0
        weights["drink"][1] = 1.0
        metadata = {
            "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
            "state_size": mind_v3_parameter_count(
                architecture="homeostatic_feature_projection_linear_action_head_v2"
            ),
            "architecture": "homeostatic_feature_projection_linear_action_head_v2",
            "action_head_weights": weights,
            "action_head_bias": {action: 0.0 for action in ACTION_NAMES},
        }
        action_mask = {action: action in {"eat", "drink"} for action in ACTION_NAMES}
        hungry = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        hungry[0] = 0.1
        hungry[1] = 0.9
        thirsty = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        thirsty[0] = 0.9
        thirsty[1] = 0.1

        hungry_scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=hungry,
            action_mask=action_mask,
        )
        thirsty_scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=thirsty,
            action_mask=action_mask,
        )

        self.assertGreater(hungry_scores["eat"], hungry_scores["drink"])
        self.assertGreater(thirsty_scores["drink"], thirsty_scores["eat"])

    def test_mind_v3_homeostatic_projection_ignores_non_self_inputs(
        self,
    ) -> None:
        from evolution_sim.env.runtime.observations import SELF_INPUT_FIELDS
        from evolution_sim.mind.evolution import (
            MIND_V3_CONTROLLER_SCHEMA_VERSION,
            mind_v3_parameter_count,
            score_mind_v3_metadata,
        )

        weights = {action: [0.0] * 8 for action in ACTION_NAMES}
        weights["eat"] = [1.0] * 8
        metadata = {
            "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
            "state_size": mind_v3_parameter_count(
                architecture="homeostatic_feature_projection_linear_action_head_v2"
            ),
            "architecture": "homeostatic_feature_projection_linear_action_head_v2",
            "action_head_weights": weights,
            "action_head_bias": {action: 0.0 for action in ACTION_NAMES},
        }
        action_mask = {action: action == "eat" for action in ACTION_NAMES}
        baseline = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        baseline[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.4
        baseline[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.5
        baseline[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
        baseline[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.35
        baseline[SELF_INPUT_FIELDS.index("tile_vegetation")] = 0.6
        mutated = list(baseline)
        for index in range(len(SELF_INPUT_FIELDS), OBSERVATION_INPUT_VECTOR_SIZE):
            mutated[index] = 1.0 if index % 2 else -1.0

        self.assertEqual(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=baseline,
                action_mask=action_mask,
            ),
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=mutated,
                action_mask=action_mask,
            ),
        )

    def test_mind_v3_contextual_projection_uses_local_and_navigation_inputs(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.env.runtime.observations import (
            NAVIGATION_INPUT_FIELDS,
            NAVIGATION_TARGETS,
            PATCH_CELL_COUNT,
            PATCH_INPUT_FIELDS,
            SELF_INPUT_FIELDS,
        )
        from evolution_sim.mind.evolution import (
            founder_mind_v3_metadata,
            score_mind_v3_metadata,
        )

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        action_mask = {
            action: action in {"eat", "move_east", "move_west", "stay"}
            for action in ACTION_NAMES
        }
        baseline = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        baseline[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.25
        baseline[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.8
        baseline[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
        enriched = list(baseline)
        patch_start = len(SELF_INPUT_FIELDS)
        center_start = patch_start + (PATCH_CELL_COUNT // 2) * len(
            PATCH_INPUT_FIELDS
        )
        enriched[center_start + PATCH_INPUT_FIELDS.index("food")] = 0.8
        enriched[center_start + PATCH_INPUT_FIELDS.index("carcass_energy")] = 0.6
        navigation_start = patch_start + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        carrion_start = navigation_start + NAVIGATION_TARGETS.index("carrion") * len(
            NAVIGATION_INPUT_FIELDS
        )
        enriched[carrion_start + NAVIGATION_INPUT_FIELDS.index("dx")] = 1.0
        enriched[carrion_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.2
        enriched[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 0.9

        self.assertNotEqual(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=baseline,
                action_mask=action_mask,
            ),
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=enriched,
                action_mask=action_mask,
            ),
        )

    def test_mind_v3_need_gated_navigation_moves_toward_water_when_thirsty(
        self,
    ) -> None:
        from evolution_sim.env.runtime.observations import (
            NAVIGATION_INPUT_FIELDS,
            NAVIGATION_TARGETS,
            PATCH_CELL_COUNT,
            PATCH_INPUT_FIELDS,
            SELF_INPUT_FIELDS,
        )
        from evolution_sim.mind.evolution import (
            MIND_V3_CONTROLLER_ARCHITECTURE,
            MIND_V3_CONTROLLER_SCHEMA_VERSION,
            mind_v3_parameter_count,
            score_mind_v3_metadata,
        )

        hidden_units = 24
        weights = {action: [0.0] * hidden_units for action in ACTION_NAMES}
        weights["move_north"][17] = -1.0
        metadata = {
            "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
            "state_size": mind_v3_parameter_count(
                architecture=MIND_V3_CONTROLLER_ARCHITECTURE
            ),
            "architecture": MIND_V3_CONTROLLER_ARCHITECTURE,
            "action_head_weights": weights,
            "action_head_bias": {action: 0.0 for action in ACTION_NAMES},
        }
        action_mask = {
            action: action in {"move_north", "stay"} for action in ACTION_NAMES
        }
        quenched = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        quenched[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.8
        quenched[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.9
        quenched[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
        thirsty = list(quenched)
        thirsty[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.2
        navigation_start = len(SELF_INPUT_FIELDS) + (
            PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        )
        water_start = navigation_start + NAVIGATION_TARGETS.index("water") * len(
            NAVIGATION_INPUT_FIELDS
        )
        for observation in (quenched, thirsty):
            observation[water_start + NAVIGATION_INPUT_FIELDS.index("dy")] = -1.0
            observation[water_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.2
            observation[water_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 1.0

        quenched_scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=quenched,
            action_mask=action_mask,
        )
        thirsty_scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=thirsty,
            action_mask=action_mask,
        )

        self.assertGreater(
            thirsty_scores["move_north"],
            quenched_scores["move_north"],
        )
        self.assertGreater(thirsty_scores["move_north"], thirsty_scores["stay"])

    def test_mind_v3_scavenger_founder_prior_favors_carrion_lane(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.env.runtime.observations import (
            NAVIGATION_INPUT_FIELDS,
            NAVIGATION_TARGETS,
            PATCH_CELL_COUNT,
            PATCH_INPUT_FIELDS,
            SELF_INPUT_FIELDS,
        )
        from evolution_sim.mind.evolution import (
            founder_mind_v3_metadata,
            score_mind_v3_metadata,
        )

        metadata = founder_mind_v3_metadata(
            agent_id=4,
            rng=Random(17),
            specialization_profile="scavenger",
        )
        observation = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        observation[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.35
        observation[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.75
        observation[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
        observation[SELF_INPUT_FIELDS.index("meat_mode_code")] = 1.0
        patch_start = len(SELF_INPUT_FIELDS)
        center_start = patch_start + (PATCH_CELL_COUNT // 2) * len(
            PATCH_INPUT_FIELDS
        )
        observation[
            center_start + PATCH_INPUT_FIELDS.index("carcass_energy")
        ] = 0.8
        navigation_start = patch_start + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        carrion_start = navigation_start + NAVIGATION_TARGETS.index(
            "carrion"
        ) * len(NAVIGATION_INPUT_FIELDS)
        observation[carrion_start + NAVIGATION_INPUT_FIELDS.index("dx")] = 1.0
        observation[carrion_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.2
        observation[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 0.9
        action_mask = {
            action: action in {"eat", "move_east", "attack_east", "drink", "stay"}
            for action in ACTION_NAMES
        }

        scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=observation,
            action_mask=action_mask,
        )

        self.assertEqual(metadata["specialization_profile"], "scavenger")
        self.assertGreater(scores["eat"], scores["attack_east"])
        self.assertGreater(scores["move_east"], scores["attack_east"])

    def test_mind_v3_homeostatic_projection_ignores_mind_inheritance_availability(
        self,
    ) -> None:
        from evolution_sim.env.runtime.observations import SELF_INPUT_FIELDS
        from evolution_sim.mind.evolution import (
            MIND_V3_CONTROLLER_SCHEMA_VERSION,
            mind_v3_parameter_count,
            score_mind_v3_metadata,
        )

        weights = {action: [0.0] * 8 for action in ACTION_NAMES}
        weights["eat"][7] = 1.0
        metadata = {
            "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
            "state_size": mind_v3_parameter_count(
                architecture="homeostatic_feature_projection_linear_action_head_v2"
            ),
            "architecture": "homeostatic_feature_projection_linear_action_head_v2",
            "action_head_weights": weights,
            "action_head_bias": {action: 0.0 for action in ACTION_NAMES},
        }
        action_mask = {action: action == "eat" for action in ACTION_NAMES}
        unavailable = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        unavailable[SELF_INPUT_FIELDS.index("trophic_role_code")] = 0.4
        unavailable[SELF_INPUT_FIELDS.index("meat_mode_code")] = 0.3
        unavailable[SELF_INPUT_FIELDS.index("season_code")] = 1.0
        unavailable[SELF_INPUT_FIELDS.index("communication_signal")] = 0.2
        available = list(unavailable)
        available[SELF_INPUT_FIELDS.index("mind_inheritance_available")] = 1.0

        self.assertEqual(
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=unavailable,
                action_mask=action_mask,
            ),
            score_mind_v3_metadata(
                metadata=metadata,
                observation_input=available,
                action_mask=action_mask,
            ),
        )

    def test_mind_v3_homeostatic_feature_fields_are_policy_visible_self_state(
        self,
    ) -> None:
        from evolution_sim.env.runtime.observations import SELF_INPUT_FIELDS
        from evolution_sim.mind.evolution import MIND_V3_HOMEOSTATIC_FEATURE_FIELDS

        self.assertTrue(
            set(MIND_V3_HOMEOSTATIC_FEATURE_FIELDS).issubset(SELF_INPUT_FIELDS)
        )
        self.assertNotIn(
            "mind_inheritance_available",
            MIND_V3_HOMEOSTATIC_FEATURE_FIELDS,
        )

    def test_ecological_policy_input_contract_excludes_controller_diagnostics(
        self,
    ) -> None:
        contract = ecological_policy_input_contract()

        self.assertEqual(
            contract["schema_version"],
            ECOLOGICAL_POLICY_INPUT_SCHEMA_VERSION,
        )
        self.assertIn(
            "self.mind_inheritance_available",
            contract["excluded_controller_diagnostic_fields"],
        )
        self.assertEqual(
            contract["ecological_vector_size"],
            ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        )

    def test_ecological_policy_input_is_invariant_to_mind_inheritance_bit(
        self,
    ) -> None:
        from evolution_sim.env.runtime.observations import SELF_INPUT_FIELDS

        unavailable = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        unavailable[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.4
        unavailable[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.6
        available = list(unavailable)
        available[SELF_INPUT_FIELDS.index("mind_inheritance_available")] = 1.0

        self.assertEqual(
            ecological_policy_values_from_decoded(unavailable),
            ecological_policy_values_from_decoded(available),
        )
        self.assertEqual(
            len(ecological_policy_values_from_decoded(available)),
            ECOLOGICAL_POLICY_INPUT_VECTOR_SIZE,
        )
        self.assertEqual(
            CONTROLLER_DIAGNOSTIC_INPUT_FIELDS,
            ("self.mind_inheritance_available",),
        )

    def test_mind_v3_neural_artifact_trains_on_horizon_labels(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))
            fixture_report = build_fixture_label_report(
                [
                    {
                        "schema_version": "mind_v3_autonomous_evolution_evaluation_v1",
                        "fixture_gate": {
                            "suite": "basic",
                            "seeds": [7],
                            "ticks": 2,
                            "min_alive": 1.0,
                            "min_births": 0.0,
                            "min_mixed_stable_births": 0.0,
                            "min_energy_viability": 0.5,
                            "min_hydration_viability": 0.5,
                            "min_health_viability": 0.5,
                            "min_matched_diet_viability": 0.0,
                            "min_biologically_ready": 0.0,
                            "per_fixture": {
                                "carrion_only": {
                                    "metrics": {
                                        "alive_agents_mean": 0.0,
                                        "births_mean": 0.0,
                                        "energy_viability_share_mean": 0.0,
                                        "hydration_viability_share_mean": 0.0,
                                        "health_viability_share_mean": 0.0,
                                        "matched_diet_viability_share_mean": 0.0,
                                        "biologically_ready_agents_mean": 0.0,
                                    }
                                }
                            },
                        },
                    }
                ]
            )

            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                fixture_label_report=fixture_report,
                hidden_units=6,
                seed=5,
            )

            self.assertEqual(
                artifact["schema_version"],
                MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
            )
            self.assertEqual(artifact["input_policy"], MIND_V3_NEURAL_INPUT_POLICY)
            self.assertEqual(artifact["hidden_units"], 6)
            self.assertGreater(artifact["trained_record_count"], 0)
            self.assertGreater(
                artifact["fixture_pressure_summary"]["pressure_total"],
                0.0,
            )
            self.assertGreater(artifact["fixture_action_bias_delta"]["eat"], 0.0)
            self.assertGreater(artifact["fixture_action_bias_delta"]["drink"], 0.0)
            self.assertEqual(
                artifact["fixture_context_bias"]["policy"],
                MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_BIAS_POLICY,
            )
            self.assertGreater(
                artifact["fixture_context_bias"]["carrion_pressure"],
                0.0,
            )
            first_record = dataset.records[0]
            scores = score_mind_v3_neural_artifact(
                artifact=artifact,
                observation_input=first_record["observation_input"],
                action_mask=first_record["action_mask"],
            )

        self.assertTrue(scores)
        self.assertTrue(set(scores).issubset(ACTION_NAMES))

    def test_mind_v3_neural_artifact_applies_trajectory_weights(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)

            def action_dataset(action: str) -> TrajectoryJsonlDataset:
                records = []
                for record in dataset.records:
                    updated = dict(record)
                    updated["requested_action"] = action
                    updated["resolved_action"] = action
                    updated["action_valid"] = True
                    updated["resolution_action_valid"] = True
                    records.append(updated)
                return TrajectoryJsonlDataset(
                    path=tmp_path / f"{action}.jsonl.gz",
                    header=dict(dataset.header),
                    records=tuple(records),
                    footer=dict(dataset.footer),
                )

            eat_dataset = action_dataset("eat")
            stay_dataset = action_dataset("stay")
            horizon_report = build_horizon_label_report(
                [eat_dataset, stay_dataset],
                horizons=(1,),
            )
            artifact = train_mind_v3_neural_artifact(
                [eat_dataset, stay_dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
                trajectory_weight_multipliers=(1.0, 5.0),
            )

        self.assertEqual(artifact["trajectory_weight_multipliers"], [1.0, 5.0])
        self.assertEqual(artifact["training_summary"]["trajectory_weight_max"], 5.0)
        self.assertGreater(
            artifact["training_summary"]["sample_weight_mean"],
            artifact["training_summary"]["horizon_weight_mean"],
        )
        self.assertGreater(
            artifact["action_output_bias"]["stay"],
            artifact["action_output_bias"]["eat"],
        )

    def test_mind_v3_horizon_fixture_artifact_trains_direct_policy_values(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))

            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
                artifact_mode=MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
            )

        self.assertEqual(artifact["model_type"], MIND_V3_HORIZON_FIXTURE_MODEL_TYPE)
        self.assertEqual(
            artifact["action_value_summary"]["policy"],
            MIND_V3_HORIZON_FIXTURE_SCORE_POLICY,
        )
        self.assertEqual(set(artifact["action_value_bias"]), set(ACTION_NAMES))
        self.assertEqual(set(artifact["action_value_weights"]), set(ACTION_NAMES))

    def test_mind_v3_neural_fixture_context_bias_uses_visible_targets(
        self,
    ) -> None:
        import base64
        import struct
        import zlib

        from evolution_sim.env.runtime.observations import (
            NAVIGATION_INPUT_FIELDS,
            NAVIGATION_TARGETS,
            OBSERVATION_ENCODER_VERSION,
            OBSERVATION_INPUT_DTYPE,
            OBSERVATION_INPUT_VALUE_RANGE,
            OBSERVATION_QUANTIZATION_SCALE,
            OBSERVATION_SCHEMA_VERSION,
            OBSERVATION_STORAGE_DTYPE,
            OBSERVATION_STORAGE_ENCODING,
            PATCH_CELL_COUNT,
            PATCH_INPUT_FIELDS,
            SELF_INPUT_FIELDS,
        )

        def encoded(values: list[float]) -> dict[str, object]:
            quantized = [
                int(
                    round(
                        max(-1.0, min(1.0, value))
                        * OBSERVATION_QUANTIZATION_SCALE
                    )
                )
                for value in values
            ]
            packed = struct.pack(f"<{len(quantized)}h", *quantized)
            return {
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "encoder_version": OBSERVATION_ENCODER_VERSION,
                "decoded_dtype": OBSERVATION_INPUT_DTYPE,
                "storage_dtype": OBSERVATION_STORAGE_DTYPE,
                "storage_encoding": OBSERVATION_STORAGE_ENCODING,
                "shape": [len(values)],
                "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
                "data": base64.b64encode(zlib.compress(packed, level=6)).decode(
                    "ascii"
                ),
            }

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))
            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
            )

        artifact["action_output_weights"] = {
            action: [0.0] * int(artifact["hidden_units"]) for action in ACTION_NAMES
        }
        artifact["action_output_bias"] = {action: 0.0 for action in ACTION_NAMES}
        artifact["fixture_action_bias_delta"] = {
            action: 0.0 for action in ACTION_NAMES
        }
        artifact["fixture_context_bias"] = {
            "policy": MIND_V3_NEURAL_CONTEXTUAL_FIXTURE_BIAS_POLICY,
            "carrion_pressure": 8.0,
            "mixed_birth_pressure": 0.0,
            "max_scale": 0.18,
        }
        navigation_start = len(SELF_INPUT_FIELDS) + (
            PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        )

        thirsty = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        thirsty[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.9
        thirsty[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.2
        thirsty[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.9
        water_start = navigation_start + NAVIGATION_TARGETS.index("water") * len(
            NAVIGATION_INPUT_FIELDS
        )
        thirsty[water_start + NAVIGATION_INPUT_FIELDS.index("dy")] = -1.0
        thirsty[water_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 1.0
        water_scores = score_mind_v3_neural_artifact(
            artifact=artifact,
            observation_input=encoded(thirsty),
            action_mask={
                action: action in {"drink", "move_north", "stay"}
                for action in ACTION_NAMES
            },
        )

        scavenger = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        scavenger[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.25
        scavenger[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.9
        scavenger[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.2
        center_start = len(SELF_INPUT_FIELDS) + (
            PATCH_CELL_COUNT // 2
        ) * len(PATCH_INPUT_FIELDS)
        scavenger[center_start + PATCH_INPUT_FIELDS.index("carcass_energy")] = 0.8
        carrion_start = navigation_start + NAVIGATION_TARGETS.index(
            "carrion"
        ) * len(NAVIGATION_INPUT_FIELDS)
        scavenger[carrion_start + NAVIGATION_INPUT_FIELDS.index("dx")] = 1.0
        scavenger[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 1.0
        carrion_scores = score_mind_v3_neural_artifact(
            artifact=artifact,
            observation_input=encoded(scavenger),
            action_mask={
                action: action in {"eat", "move_east", "stay"}
                for action in ACTION_NAMES
            },
        )

        self.assertGreater(water_scores["drink"], water_scores["stay"])
        self.assertGreater(water_scores["move_north"], water_scores["stay"])
        self.assertGreater(carrion_scores["eat"], carrion_scores["stay"])
        self.assertGreater(carrion_scores["move_east"], carrion_scores["stay"])

    def test_mind_v3_neural_artifact_ignores_mind_inheritance_bit(
        self,
    ) -> None:
        import base64
        import struct
        import zlib

        from evolution_sim.env.runtime.observations import (
            OBSERVATION_ENCODER_VERSION,
            OBSERVATION_INPUT_DTYPE,
            OBSERVATION_INPUT_VALUE_RANGE,
            OBSERVATION_QUANTIZATION_SCALE,
            OBSERVATION_SCHEMA_VERSION,
            OBSERVATION_STORAGE_DTYPE,
            OBSERVATION_STORAGE_ENCODING,
            SELF_INPUT_FIELDS,
            decode_observation_input,
        )

        def encoded(values: list[float]) -> dict[str, object]:
            quantized = [
                int(
                    round(
                        max(-1.0, min(1.0, value))
                        * OBSERVATION_QUANTIZATION_SCALE
                    )
                )
                for value in values
            ]
            packed = struct.pack(f"<{len(quantized)}h", *quantized)
            return {
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "encoder_version": OBSERVATION_ENCODER_VERSION,
                "decoded_dtype": OBSERVATION_INPUT_DTYPE,
                "storage_dtype": OBSERVATION_STORAGE_DTYPE,
                "storage_encoding": OBSERVATION_STORAGE_ENCODING,
                "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
                "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
                "data": base64.b64encode(zlib.compress(packed, level=6)).decode(
                    "ascii"
                ),
            }

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))
            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
            )
            first_record = dataset.records[0]
            base_values = decode_observation_input(first_record["observation_input"])
            unavailable = list(base_values)
            available = list(base_values)
            inheritance_index = SELF_INPUT_FIELDS.index("mind_inheritance_available")
            unavailable[inheritance_index] = 0.0
            available[inheritance_index] = 1.0
            action_mask = {action: True for action in ACTION_NAMES}

            unavailable_scores = score_mind_v3_neural_artifact(
                artifact=artifact,
                observation_input=encoded(unavailable),
                action_mask=action_mask,
            )
            available_scores = score_mind_v3_neural_artifact(
                artifact=artifact,
                observation_input=encoded(available),
                action_mask=action_mask,
            )

        self.assertEqual(unavailable_scores, available_scores)

    def test_mind_v3_policy_can_use_frozen_neural_artifact(
        self,
    ) -> None:
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))
            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
            )
            first_record = dataset.records[0]
            policy = MindV3EvolutionPolicy(seed=7, neural_artifact=artifact)

            decision = policy.decide(
                {
                    "metadata": {"agent_id": 3},
                    "self": {"trophic_role": "herbivore", "meat_mode": "none"},
                    "observation_input": first_record["observation_input"],
                },
                dict(first_record["action_mask"]),
            )
            update = policy.observe_transition(dict(first_record))

        self.assertIn(decision.requested_action, ACTION_NAMES)
        self.assertEqual(
            decision.diagnostics["controller_backend"],
            "frozen_neural_artifact_linear_anchor",
        )
        self.assertEqual(
            decision.diagnostics["neural_linear_anchor_policy"],
            "linear_controller_margin_guarded_neural_residual_v2",
        )
        self.assertEqual(decision.diagnostics["neural_residual_scale"], 0.05)
        self.assertEqual(
            decision.diagnostics["neural_residual_max_linear_override_margin"],
            0.015,
        )
        self.assertIn(decision.diagnostics["neural_top_action"], ACTION_NAMES)
        self.assertIn(decision.diagnostics["linear_anchor_action"], ACTION_NAMES)
        self.assertIn(decision.diagnostics["anchored_action"], ACTION_NAMES)
        self.assertIn(
            decision.diagnostics["neural_residual_shadowed"],
            {True, False},
        )
        self.assertEqual(
            decision.diagnostics["neural_model_type"],
            "deterministic_ecological_mlp_policy_v1",
        )
        self.assertIsNone(update)

    def test_mind_v3_policy_can_use_direct_horizon_fixture_artifact(
        self,
    ) -> None:
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))
            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
                artifact_mode=MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
            )
            artifact["action_value_weights"] = {
                action: [0.0] * int(artifact["hidden_units"])
                for action in ACTION_NAMES
            }
            artifact["action_value_bias"] = {
                action: (2.0 if action == "move_east" else 0.0)
                for action in ACTION_NAMES
            }
            artifact["fixture_action_bias_delta"] = {
                action: 0.0 for action in ACTION_NAMES
            }
            first_record = dataset.records[0]
            policy = MindV3EvolutionPolicy(seed=7, neural_artifact=artifact)

            decision = policy.decide(
                {
                    "metadata": {"agent_id": 3},
                    "self": {"trophic_role": "herbivore", "meat_mode": "none"},
                    "observation_input": first_record["observation_input"],
                },
                {
                    action: action in {"stay", "move_east"}
                    for action in ACTION_NAMES
                },
            )
            update = policy.observe_transition(dict(first_record))

        self.assertEqual(decision.requested_action, "move_east")
        self.assertEqual(
            decision.diagnostics["controller_backend"],
            "frozen_horizon_fixture_policy_artifact",
        )
        self.assertEqual(
            decision.diagnostics["neural_model_type"],
            MIND_V3_HORIZON_FIXTURE_MODEL_TYPE,
        )
        self.assertTrue(decision.diagnostics["horizon_fixture_anchor_bypassed"])
        self.assertNotIn("neural_linear_anchor_policy", decision.diagnostics)
        self.assertIsNone(update)

    def test_mind_v3_neural_linear_anchor_can_counter_collapsed_logits(
        self,
    ) -> None:
        from evolution_sim.mind.v3_policy import _blend_neural_with_linear_anchor

        action_mask = {
            action: action in {"eat", "drink"} for action in ACTION_NAMES
        }
        scores = _blend_neural_with_linear_anchor(
            neural_scores={"eat": 10.0, "drink": 0.0},
            linear_scores={"eat": 0.0, "drink": 1.0},
            action_mask=action_mask,
        )

        self.assertGreater(scores["drink"], scores["eat"])
        self.assertEqual(scores["drink"], 1.0)
        self.assertEqual(scores["eat"], 0.0)
        margin_guard_scores = _blend_neural_with_linear_anchor(
            neural_scores={"move_east": 10.0, "drink": 0.0},
            linear_scores={"move_east": 0.0, "drink": 1.0},
            action_mask={
                action: action in {"move_east", "drink"}
                for action in ACTION_NAMES
            },
        )

        self.assertGreater(
            margin_guard_scores["drink"],
            margin_guard_scores["move_east"],
        )
        self.assertEqual(margin_guard_scores["drink"], 1.0)
        self.assertEqual(margin_guard_scores["move_east"], 0.0)

    def test_mind_v3_neural_policy_updates_anchor_controller_only(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "train.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_report = build_horizon_label_report([dataset], horizons=(1,))
            artifact = train_mind_v3_neural_artifact(
                [dataset],
                horizon_label_report=horizon_report,
                hidden_units=6,
                seed=5,
            )
            metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
            policy = MindV3EvolutionPolicy(seed=9, neural_artifact=artifact)
            policy.register_agent_mind(agent_id=3, metadata=metadata)
            before = policy.agent_mind_metadata(agent_id=3)

            trace = policy.observe_transition(
                {
                    "agent_id": 3,
                    "policy_id": policy.policy_id,
                    "observation_input": {
                        "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                    },
                    "action_mask": {
                        action: action in {"stay", "move_east"}
                        for action in ACTION_NAMES
                    },
                    "requested_action": "move_east",
                    "resolved_action": "move_east",
                    "action_valid": True,
                    "resolution_action_valid": True,
                    "reward": {"total": 0.8},
                }
            )
            after = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNotNone(trace)
        self.assertTrue(trace["neural_artifact_frozen"])
        self.assertTrue(trace["anchor_controller_update"])
        self.assertNotEqual(
            after["action_head_bias"]["move_east"],
            before["action_head_bias"]["move_east"],
        )

    def test_mind_v3_neural_artifact_cli_writes_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "train.jsonl.gz"
            horizon_path = tmp_path / "horizon.json"
            artifact_path = tmp_path / "artifact.json"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_path.write_text(
                json.dumps(build_horizon_label_report([dataset], horizons=(1,))),
                encoding="utf-8",
            )

            with patch(
                "sys.argv",
                [
                    "mind_v3_neural_artifact",
                    "--trajectory",
                    str(trajectory_path),
                    "--horizon-labels",
                    str(horizon_path),
                    "--hidden-units",
                    "6",
                    "--output",
                    str(artifact_path),
                ],
            ):
                mind_v3_neural_artifact.main()

            artifact = load_mind_v3_neural_artifact(artifact_path)

        self.assertEqual(
            artifact["schema_version"],
            MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(artifact["hidden_units"], 6)

    def test_mind_v3_evaluate_cli_accepts_neural_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "train.jsonl.gz"
            horizon_path = tmp_path / "horizon.json"
            artifact_path = tmp_path / "artifact.json"
            report_path = tmp_path / "eval.json"
            self._write_tiny_trajectory(trajectory_path, seed=7)
            dataset = load_trajectory_jsonl(trajectory_path)
            horizon_path.write_text(
                json.dumps(build_horizon_label_report([dataset], horizons=(1,))),
                encoding="utf-8",
            )
            with patch(
                "sys.argv",
                [
                    "mind_v3_neural_artifact",
                    "--trajectory",
                    str(trajectory_path),
                    "--horizon-labels",
                    str(horizon_path),
                    "--hidden-units",
                    "6",
                    "--output",
                    str(artifact_path),
                ],
            ):
                mind_v3_neural_artifact.main()

            with patch(
                "sys.argv",
                [
                    "mind_v3_evaluate",
                    "--seeds",
                    "7",
                    "--ticks",
                    "2",
                    "--neural-artifact",
                    str(artifact_path),
                    "--anchored-neural-artifact",
                    str(artifact_path),
                    "--compare-linear-baseline",
                    "--output",
                    str(report_path),
                ],
            ):
                mind_v3_evaluate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["policy"]["neural_artifact_schema_version"],
            MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["policy"]["neural_input_policy"],
            MIND_V3_NEURAL_INPUT_POLICY,
        )
        self.assertTrue(report["policy"]["linear_baseline_compared"])
        self.assertIn("mind_v3_linear", report["comparison"])
        self.assertIn("mind_v3_anchored_neural", report["comparison"])
        self.assertIn("neural_vs_linear_delta", report["comparison"])
        self.assertIn("primary_vs_anchored_neural_delta", report["comparison"])
        anchor_diagnostics = report["comparison"]["mind_v3"]["aggregate"][
            "neural_anchor_diagnostics"
        ]
        self.assertEqual(
            anchor_diagnostics["policy"],
            "mind_v3_neural_anchor_diagnostics_v1",
        )
        self.assertGreater(anchor_diagnostics["decision_count"], 0)
        self.assertIn("anchored_action_counts", anchor_diagnostics)
        self.assertIn("linear_to_anchored_action_counts", anchor_diagnostics)
        self.assertIn("linear_anchor_score_margin_mean", anchor_diagnostics)

    def test_mind_v3_child_metadata_mutates_from_parent(self) -> None:
        from random import Random

        from evolution_sim.mind.evolution import (
            founder_mind_v3_metadata,
            inherit_mind_v3_metadata,
        )

        parent = founder_mind_v3_metadata(agent_id=1, rng=Random(11))
        child = inherit_mind_v3_metadata(
            primary_parent_metadata=parent,
            secondary_parent_metadata=None,
            child_agent_id=2,
            rng=Random(12),
        )

        self.assertTrue(child["inherited_state"])
        self.assertEqual(
            child["parent_schema_versions"],
            ["mind_v3_controller_metadata_v1"],
        )
        self.assertNotEqual(child["action_head_bias"], parent["action_head_bias"])

    def test_mind_v3_founder_metadata_accepts_explicit_specialization_profile(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        metadata = founder_mind_v3_metadata(
            agent_id=4,
            rng=Random(12),
            specialization_profile="reproducer",
        )
        forager = founder_mind_v3_metadata(
            agent_id=4,
            rng=Random(12),
            specialization_profile="forager",
        )

        self.assertEqual(metadata["specialization_profile"], "reproducer")
        self.assertGreater(
            metadata["action_head_bias"]["mate"],
            forager["action_head_bias"]["mate"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported Mind v3"):
            founder_mind_v3_metadata(
                agent_id=5,
                rng=Random(12),
                specialization_profile="unsupported",
            )

    def test_load_mind_v3_controller_metadata_rejects_malformed_weights(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import (
            founder_mind_v3_metadata,
            load_mind_v3_controller_metadata,
            mind_v3_parameter_count,
        )

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            valid_path = tmp_path / "valid.json"
            bad_weight_path = tmp_path / "bad-weight.json"
            bad_bias_path = tmp_path / "bad-bias.json"
            bad_architecture_path = tmp_path / "bad-architecture.json"
            metadata = founder_mind_v3_metadata(agent_id=1, rng=Random(7))
            valid_path.write_text(json.dumps(metadata), encoding="utf-8")
            bad_weight = dict(metadata)
            bad_weight["action_head_weights"] = dict(metadata["action_head_weights"])
            bad_weight["action_head_weights"]["eat"] = [0.0]
            bad_weight_path.write_text(json.dumps(bad_weight), encoding="utf-8")
            bad_bias = dict(metadata)
            bad_bias["action_head_bias"] = dict(metadata["action_head_bias"])
            bad_bias["action_head_bias"]["eat"] = "not-a-number"
            bad_bias_path.write_text(json.dumps(bad_bias), encoding="utf-8")
            bad_architecture = dict(metadata)
            bad_architecture["architecture"] = "future_unknown_architecture"
            bad_architecture_path.write_text(
                json.dumps(bad_architecture),
                encoding="utf-8",
            )

            loaded = load_mind_v3_controller_metadata(valid_path)
            self.assertEqual(
                len(loaded["action_head_weights"]["eat"]),
                mind_v3_parameter_count() // len(ACTION_NAMES) - 1,
            )
            with self.assertRaisesRegex(ValueError, "action_head_weights.eat"):
                load_mind_v3_controller_metadata(bad_weight_path)
            with self.assertRaisesRegex(ValueError, "action_head_bias.eat"):
                load_mind_v3_controller_metadata(bad_bias_path)
            with self.assertRaisesRegex(ValueError, "unsupported architecture"):
                load_mind_v3_controller_metadata(bad_architecture_path)

    def test_load_mind_v3_founder_template_prefers_search_report_pool(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import (
            founder_mind_v3_metadata,
            load_mind_v3_controller_metadata,
            load_mind_v3_founder_template,
        )

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "search-report.json"
            selected = founder_mind_v3_metadata(agent_id=1, rng=Random(1))
            forager = founder_mind_v3_metadata(agent_id=2, rng=Random(2))
            forager["specialization_profile"] = "forager"
            reproducer = founder_mind_v3_metadata(agent_id=3, rng=Random(3))
            reproducer["specialization_profile"] = "reproducer"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "controller_metadata": selected,
                            "founder_template_pool": [forager, reproducer],
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded_pool = load_mind_v3_founder_template(report_path)
            loaded_selected = load_mind_v3_controller_metadata(report_path)

        self.assertIsInstance(loaded_pool, list)
        self.assertEqual(
            [metadata["specialization_profile"] for metadata in loaded_pool],
            ["forager", "reproducer"],
        )
        self.assertEqual(
            loaded_selected["specialization_profile"],
            selected["specialization_profile"],
        )

    def test_mind_v3_policy_uses_agent_metadata_without_heuristic_source(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        observation = {
            "metadata": {"agent_id": 3},
            "observation_input": {"values": [0.0] * OBSERVATION_INPUT_VECTOR_SIZE},
            "action_mask": {
                action: action in {"stay", "move_east"} for action in ACTION_NAMES
            },
        }

        decision = policy.decide(observation, dict(observation["action_mask"]))

        self.assertIn(decision.requested_action, {"stay", "move_east"})
        self.assertEqual(
            decision.policy_id,
            "mind_v3_autonomous_evolution_policy",
        )
        self.assertEqual(
            decision.policy_version,
            "mind_v3_autonomous_evolution_policy_v1",
        )
        self.assertNotIn("heuristic", decision.source)
        self.assertTrue(decision.diagnostics["heuristic_free"])

    def test_mind_v3_policy_can_initialize_founders_from_template(self) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        template = founder_mind_v3_metadata(agent_id=0, rng=Random(5))
        policy = MindV3EvolutionPolicy(
            seed=11,
            founder_template_metadata=template,
        )

        founder = policy.founder_metadata(agent_id=7)

        self.assertEqual(
            founder["parent_schema_versions"],
            ["mind_v3_controller_metadata_v1"],
        )
        self.assertNotEqual(
            founder["action_head_bias"],
            template["action_head_bias"],
        )

    def test_mind_v3_policy_spreads_founders_across_template_pool(self) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        forager = founder_mind_v3_metadata(agent_id=0, rng=Random(5))
        forager["specialization_profile"] = "forager"
        reproducer = founder_mind_v3_metadata(agent_id=1, rng=Random(6))
        reproducer["specialization_profile"] = "reproducer"
        policy = MindV3EvolutionPolicy(
            seed=11,
            founder_template_metadata=[forager, reproducer],
        )

        first = policy.founder_metadata(agent_id=0)
        second = policy.founder_metadata(agent_id=1)
        third = policy.founder_metadata(agent_id=2)

        self.assertEqual(first["specialization_profile"], "forager")
        self.assertEqual(second["specialization_profile"], "reproducer")
        self.assertEqual(third["specialization_profile"], "forager")
        self.assertEqual(
            first["parent_schema_versions"],
            ["mind_v3_controller_metadata_v1"],
        )

    def test_mind_v3_policy_assigns_template_pool_by_visible_trophic_context(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        forager = founder_mind_v3_metadata(
            agent_id=0,
            rng=Random(5),
            specialization_profile="forager",
        )
        reproducer = founder_mind_v3_metadata(
            agent_id=1,
            rng=Random(6),
            specialization_profile="reproducer",
        )
        scavenger = founder_mind_v3_metadata(
            agent_id=2,
            rng=Random(7),
            specialization_profile="scavenger",
        )
        predator = founder_mind_v3_metadata(
            agent_id=3,
            rng=Random(8),
            specialization_profile="predator_scavenger",
        )
        policy = MindV3EvolutionPolicy(
            seed=11,
            founder_template_metadata=[forager, reproducer, scavenger, predator],
        )

        scavenger_founder = policy.contextual_founder_metadata(
            agent_id=1,
            trophic_role="carnivore",
            meat_mode="scavenger",
        )
        hunter_founder = policy.contextual_founder_metadata(
            agent_id=2,
            trophic_role="carnivore",
            meat_mode="hunter",
        )
        herbivore_founder = policy.contextual_founder_metadata(
            agent_id=3,
            trophic_role="herbivore",
            meat_mode="none",
        )

        self.assertEqual(scavenger_founder["specialization_profile"], "scavenger")
        self.assertEqual(
            scavenger_founder["founder_template_source_profile"],
            "scavenger",
        )
        self.assertEqual(
            scavenger_founder["founder_template_assignment_policy"],
            "contextual_trophic_founder_template_assignment_v1",
        )
        self.assertEqual(
            scavenger_founder["founder_template_context"],
            {"trophic_role": "carnivore", "meat_mode": "scavenger"},
        )
        self.assertEqual(
            hunter_founder["specialization_profile"],
            "predator_scavenger",
        )
        self.assertEqual(herbivore_founder["specialization_profile"], "forager")

    def test_mind_v3_world_passes_founder_trophic_context_to_policy(self) -> None:
        from evolution_sim.config import WorldConfig
        from evolution_sim.env import SimulationWorld
        from evolution_sim.env.runtime.policy import ActionDecision

        class RecordingPolicy:
            policy_id = "recording_policy"
            policy_version = "recording_policy_v1"

            def __init__(self) -> None:
                self.contexts: list[dict[str, object]] = []

            def contextual_founder_metadata(
                self,
                *,
                agent_id: int,
                trophic_role: str | None = None,
                meat_mode: str | None = None,
            ) -> dict[str, object]:
                self.contexts.append(
                    {
                        "agent_id": agent_id,
                        "trophic_role": trophic_role,
                        "meat_mode": meat_mode,
                    }
                )
                return {"inherited_state": True}

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                return ActionDecision(
                    requested_action="stay",
                    source=self.policy_version,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                )

        policy = RecordingPolicy()
        SimulationWorld(
            WorldConfig(seed=3, max_ticks=1, initial_agents=4),
            policy=policy,
        )

        self.assertEqual(len(policy.contexts), 4)
        self.assertTrue(
            all(
                context["trophic_role"]
                in {"herbivore", "omnivore", "carnivore"}
                for context in policy.contexts
            )
        )
        self.assertTrue(
            all(
                context["meat_mode"]
                in {"none", "scavenger", "hunter", "mixed"}
                for context in policy.contexts
            )
        )

    def test_mind_v3_policy_observe_transition_updates_agent_controller(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        before = policy.agent_mind_metadata(agent_id=3)
        record = {
            "agent_id": 3,
            "observation_input": {
                "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
            },
            "action_mask": {
                action: action in {"stay", "move_east"} for action in ACTION_NAMES
            },
            "requested_action": "move_east",
            "resolved_action": "move_east",
            "reward": {"total": 0.8},
        }

        trace = policy.observe_transition(record)
        after = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNotNone(trace)
        self.assertEqual(
            trace["policy"],
            "bounded_reward_modulated_controller_update_v1",
        )
        self.assertNotEqual(
            after["action_head_bias"]["move_east"],
            before["action_head_bias"]["move_east"],
        )
        self.assertEqual(
            after["action_head_bias"]["stay"],
            before["action_head_bias"]["stay"],
        )

    def test_mind_v3_policy_ignores_passive_transition_feedback(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        before = policy.agent_mind_metadata(agent_id=3)

        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "action_source": "passive",
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action == "stay" for action in ACTION_NAMES
                },
                "requested_action": "stay",
                "resolved_action": "stay",
                "action_valid": True,
                "resolution_action_valid": True,
                "reward": {"total": -1.0},
            }
        )
        after = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNone(trace)
        self.assertEqual(after["action_head_bias"], before["action_head_bias"])
        self.assertEqual(after["action_head_weights"], before["action_head_weights"])

    def test_mind_v3_policy_credits_passive_terminal_feedback_to_recent_trace(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        action_mask = {
            action: action in {"stay", "move_east"} for action in ACTION_NAMES
        }
        move_record = {
            "agent_id": 3,
            "observation_input": {
                "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
            },
            "action_mask": action_mask,
            "requested_action": "move_east",
            "resolved_action": "move_east",
            "action_valid": True,
            "resolution_action_valid": True,
            "reward": {"total": 0.0},
        }

        policy.observe_transition(move_record)
        after_move = policy.agent_mind_metadata(agent_id=3)
        terminal_trace = policy.observe_transition(
            {
                "agent_id": 3,
                "action_source": "passive",
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": action_mask,
                "requested_action": "stay",
                "resolved_action": "stay",
                "action_valid": True,
                "resolution_action_valid": True,
                "outcome": {
                    "died": True,
                    "passive": {
                        "killed": True,
                        "died_before_action": True,
                    },
                },
                "reward": {"total": -1.0},
            }
        )
        after_terminal = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNotNone(terminal_trace)
        self.assertTrue(terminal_trace["terminal_feedback"])
        self.assertFalse(terminal_trace["trace_appended_action"])
        self.assertEqual(terminal_trace["action"], "stay")
        self.assertEqual(terminal_trace["credited_actions"], ["move_east"])
        self.assertLess(
            after_terminal["action_head_bias"]["move_east"],
            after_move["action_head_bias"]["move_east"],
        )
        self.assertEqual(
            after_terminal["action_head_bias"]["stay"],
            after_move["action_head_bias"]["stay"],
        )

    def test_mind_v3_policy_uses_outcome_delta_reproduction_readiness_signal(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action in {"stay", "eat"} for action in ACTION_NAMES
                },
                "requested_action": "eat",
                "resolved_action": "eat",
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {
                    "energy_ratio": 0.42,
                    "hydration_ratio": 0.43,
                    "health_ratio": 0.68,
                },
                "after": {
                    "energy_ratio": 0.62,
                    "hydration_ratio": 0.7,
                    "health_ratio": 0.7,
                },
                "outcome": {
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 1.0,
                    "components": {
                        "survival_continuation": 0.02,
                        "resource_acquisition": 1.0,
                        "reproduction_readiness": 0.05,
                    },
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertEqual(
            trace["reward_signal_policy"],
            "balanced_bottleneck_visible_navigation_carrion_readiness_signal_v8",
        )
        self.assertEqual(trace["reward_total"], 1.0)
        self.assertGreater(trace["reward_signal"], 0.75)
        self.assertGreater(
            trace["reward_signal_components"]["readiness_delta_signal"],
            0.8,
        )
        self.assertGreater(
            trace["reward_signal_components"]["component_reward_signal"],
            0.0,
        )

    def test_mind_v3_policy_penalizes_reproduction_readiness_regression(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action in {"stay", "move_east"} for action in ACTION_NAMES
                },
                "requested_action": "move_east",
                "resolved_action": "move_east",
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {
                    "energy_ratio": 0.72,
                    "hydration_ratio": 0.72,
                    "health_ratio": 0.72,
                },
                "after": {
                    "energy_ratio": 0.5,
                    "hydration_ratio": 0.5,
                    "health_ratio": 0.5,
                },
                "outcome": {
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 0.0,
                    "components": {"survival_continuation": 0.02},
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertLess(trace["reward_signal"], 0.0)
        self.assertLess(
            trace["reward_signal_components"]["readiness_delta_signal"],
            0.0,
        )

    def test_mind_v3_policy_penalizes_repeated_no_gain_eat(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        base_record = {
            "agent_id": 3,
            "observation_input": {"values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE},
            "action_mask": {
                action: action in {"stay", "eat"} for action in ACTION_NAMES
            },
            "requested_action": "eat",
            "resolved_action": "eat",
            "action_valid": True,
            "resolution_action_valid": True,
            "before": {
                "energy_ratio": 0.5,
                "hydration_ratio": 0.5,
                "health_ratio": 0.7,
            },
            "after": {
                "energy_ratio": 0.5,
                "hydration_ratio": 0.5,
                "health_ratio": 0.7,
            },
            "outcome": {
                "feeding": {"ate": False},
                "resource_gain": 0.0,
                "reproduction_ready_after": False,
                "reproduced": False,
                "died": False,
            },
            "reward": {"total": 0.0, "components": {"survival_continuation": 0.02}},
        }

        first_trace = policy.observe_transition(dict(base_record))
        second_trace = policy.observe_transition(dict(base_record))

        self.assertIsNotNone(first_trace)
        self.assertIsNotNone(second_trace)
        self.assertEqual(
            first_trace["reward_signal_components"]["no_gain_eat"],
            1.0,
        )
        self.assertEqual(
            second_trace["reward_signal_components"]["no_gain_eat_streak"],
            1.0,
        )
        self.assertLess(second_trace["reward_signal"], first_trace["reward_signal"])

    def test_mind_v3_policy_credits_observed_animal_resource_eat(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        def run(food_source: str) -> dict[str, object]:
            metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
            policy = MindV3EvolutionPolicy(seed=9)
            policy.register_agent_mind(agent_id=3, metadata=metadata)
            trace = policy.observe_transition(
                {
                    "agent_id": 3,
                    "observation_input": {
                        "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                    },
                    "action_mask": {
                        action: action in {"stay", "eat"} for action in ACTION_NAMES
                    },
                    "requested_action": "eat",
                    "resolved_action": "eat",
                    "action_valid": True,
                    "resolution_action_valid": True,
                    "before": {
                        "energy_ratio": 0.36,
                        "hydration_ratio": 0.74,
                        "health_ratio": 0.74,
                    },
                    "after": {
                        "energy_ratio": 0.54,
                        "hydration_ratio": 0.75,
                        "health_ratio": 0.74,
                    },
                    "outcome": {
                        "feeding": {
                            "ate": True,
                            "food_source": food_source,
                            "gained_energy": 0.22,
                        },
                        "resource_gain": 0.22,
                        "reproduction_ready_after": False,
                        "reproduced": False,
                        "died": False,
                    },
                    "reward": {
                        "total": 0.22,
                        "components": {
                            "survival_continuation": 0.02,
                            "resource_acquisition": 0.22,
                        },
                    },
                }
            )
            self.assertIsNotNone(trace)
            return trace

        plant_trace = run("plant")
        carrion_trace = run("carcass")

        self.assertGreater(
            carrion_trace["reward_signal_components"]["action_outcome_signal"],
            plant_trace["reward_signal_components"]["action_outcome_signal"],
        )
        self.assertGreater(
            carrion_trace["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )

    def test_mind_v3_policy_credits_visible_carrion_navigation_move(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.env.runtime.observations import (
            NAVIGATION_INPUT_FIELDS,
            NAVIGATION_TARGETS,
            PATCH_CELL_COUNT,
            PATCH_INPUT_FIELDS,
            SELF_INPUT_FIELDS,
        )
        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        def observation_values() -> list[float]:
            values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
            values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.34
            values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.76
            values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
            values[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.2
            values[SELF_INPUT_FIELDS.index("meat_mode_code")] = 1.0
            navigation_start = len(SELF_INPUT_FIELDS) + (
                PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
            )
            carrion_start = navigation_start + NAVIGATION_TARGETS.index(
                "carrion"
            ) * len(NAVIGATION_INPUT_FIELDS)
            values[carrion_start + NAVIGATION_INPUT_FIELDS.index("dx")] = 1.0
            values[carrion_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.2
            values[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 0.9
            return values

        def run(action: str) -> dict[str, object]:
            metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
            policy = MindV3EvolutionPolicy(seed=9)
            policy.register_agent_mind(agent_id=3, metadata=metadata)
            trace = policy.observe_transition(
                {
                    "agent_id": 3,
                    "observation_input": {"values": observation_values()},
                    "action_mask": {
                        candidate: candidate in {"stay", "move_east", "move_west"}
                        for candidate in ACTION_NAMES
                    },
                    "requested_action": action,
                    "resolved_action": action,
                    "action_valid": True,
                    "resolution_action_valid": True,
                    "moved": True,
                    "before": {
                        "energy_ratio": 0.34,
                        "hydration_ratio": 0.76,
                        "health_ratio": 0.9,
                    },
                    "after": {
                        "energy_ratio": 0.34,
                        "hydration_ratio": 0.76,
                        "health_ratio": 0.9,
                    },
                    "outcome": {
                        "reproduction_ready_after": False,
                        "reproduced": False,
                        "died": False,
                    },
                    "reward": {
                        "total": 0.0,
                        "components": {"survival_continuation": 0.02},
                    },
                }
            )
            self.assertIsNotNone(trace)
            return trace

        toward = run("move_east")
        away = run("move_west")

        self.assertGreater(
            toward["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )
        self.assertGreater(
            toward["reward_signal_components"]["action_outcome_signal"],
            away["reward_signal_components"]["action_outcome_signal"],
        )
        self.assertGreater(toward["reward_signal"], away["reward_signal"])

    def test_mind_v3_policy_does_not_credit_navigation_move_without_need(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.env.runtime.observations import (
            NAVIGATION_INPUT_FIELDS,
            NAVIGATION_TARGETS,
            PATCH_CELL_COUNT,
            PATCH_INPUT_FIELDS,
            SELF_INPUT_FIELDS,
        )
        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        values = [0.0] * OBSERVATION_INPUT_VECTOR_SIZE
        values[SELF_INPUT_FIELDS.index("energy_ratio")] = 0.9
        values[SELF_INPUT_FIELDS.index("hydration_ratio")] = 0.9
        values[SELF_INPUT_FIELDS.index("health_ratio")] = 0.9
        values[SELF_INPUT_FIELDS.index("matched_diet_ratio")] = 0.9
        values[SELF_INPUT_FIELDS.index("meat_mode_code")] = 1.0
        navigation_start = len(SELF_INPUT_FIELDS) + (
            PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        )
        carrion_start = navigation_start + NAVIGATION_TARGETS.index(
            "carrion"
        ) * len(NAVIGATION_INPUT_FIELDS)
        values[carrion_start + NAVIGATION_INPUT_FIELDS.index("dx")] = 1.0
        values[carrion_start + NAVIGATION_INPUT_FIELDS.index("distance")] = 0.2
        values[carrion_start + NAVIGATION_INPUT_FIELDS.index("strength")] = 0.9
        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)

        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {"values": values},
                "action_mask": {
                    action: action in {"stay", "move_east"}
                    for action in ACTION_NAMES
                },
                "requested_action": "move_east",
                "resolved_action": "move_east",
                "action_valid": True,
                "resolution_action_valid": True,
                "moved": True,
                "before": {
                    "energy_ratio": 0.9,
                    "hydration_ratio": 0.9,
                    "health_ratio": 0.9,
                },
                "after": {
                    "energy_ratio": 0.9,
                    "hydration_ratio": 0.9,
                    "health_ratio": 0.9,
                },
                "outcome": {
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 0.0,
                    "components": {"survival_continuation": 0.02},
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertEqual(
            trace["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )

    def test_mind_v3_policy_treats_zero_delta_eat_as_no_gain(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action in {"stay", "eat"} for action in ACTION_NAMES
                },
                "requested_action": "eat",
                "resolved_action": "eat",
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {
                    "energy_ratio": 0.44,
                    "hydration_ratio": 0.74,
                    "health_ratio": 0.74,
                },
                "after": {
                    "energy_ratio": 0.44,
                    "hydration_ratio": 0.74,
                    "health_ratio": 0.74,
                },
                "outcome": {
                    "feeding": {
                        "ate": True,
                        "food_source": "plant",
                        "gained_energy": 0.0,
                    },
                    "resource_gain": 0.0,
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 0.0,
                    "components": {"survival_continuation": 0.02},
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertEqual(trace["reward_signal_components"]["no_gain_eat"], 1.0)
        self.assertLess(
            trace["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )

    def test_mind_v3_policy_credits_useful_drink_outcome(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action in {"stay", "drink"} for action in ACTION_NAMES
                },
                "requested_action": "drink",
                "resolved_action": "drink",
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {
                    "energy_ratio": 0.72,
                    "hydration_ratio": 0.32,
                    "health_ratio": 0.72,
                },
                "after": {
                    "energy_ratio": 0.72,
                    "hydration_ratio": 0.62,
                    "health_ratio": 0.72,
                },
                "outcome": {
                    "drinking": {"drank": True},
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 0.0,
                    "components": {
                        "survival_continuation": 0.02,
                        "hydration_stability": 0.3,
                    },
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertGreater(
            trace["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )
        self.assertGreater(trace["reward_signal"], 0.0)

    def test_mind_v3_policy_does_not_action_credit_nonlimiting_eat(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action in {"stay", "eat"} for action in ACTION_NAMES
                },
                "requested_action": "eat",
                "resolved_action": "eat",
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {
                    "energy_ratio": 0.86,
                    "hydration_ratio": 0.36,
                    "health_ratio": 0.72,
                },
                "after": {
                    "energy_ratio": 0.88,
                    "hydration_ratio": 0.36,
                    "health_ratio": 0.72,
                },
                "outcome": {
                    "feeding": {"ate": True},
                    "resource_gain": 0.2,
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 0.2,
                    "components": {
                        "survival_continuation": 0.02,
                        "resource_acquisition": 0.2,
                    },
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertEqual(
            trace["reward_signal_components"]["limiting_readiness_field"],
            "hydration_ratio",
        )
        self.assertEqual(
            trace["reward_signal_components"]["balanced_core_readiness_delta"],
            0.0,
        )
        self.assertLess(
            trace["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )

    def test_mind_v3_policy_penalizes_no_gain_drink_outcome(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        trace = policy.observe_transition(
            {
                "agent_id": 3,
                "observation_input": {
                    "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
                },
                "action_mask": {
                    action: action in {"stay", "drink"} for action in ACTION_NAMES
                },
                "requested_action": "drink",
                "resolved_action": "drink",
                "action_valid": True,
                "resolution_action_valid": True,
                "before": {
                    "energy_ratio": 0.72,
                    "hydration_ratio": 0.92,
                    "health_ratio": 0.72,
                },
                "after": {
                    "energy_ratio": 0.72,
                    "hydration_ratio": 0.92,
                    "health_ratio": 0.72,
                },
                "outcome": {
                    "drinking": {"drank": True},
                    "reproduction_ready_after": False,
                    "reproduced": False,
                    "died": False,
                },
                "reward": {
                    "total": 0.0,
                    "components": {"survival_continuation": 0.02},
                },
            }
        )

        self.assertIsNotNone(trace)
        self.assertLess(
            trace["reward_signal_components"]["action_outcome_signal"],
            0.0,
        )
        self.assertLess(trace["reward_signal"], 0.0)

    def test_mind_v3_policy_credits_requested_action_when_resolution_races_to_stay(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        before = policy.agent_mind_metadata(agent_id=3)
        record = {
            "agent_id": 3,
            "observation_input": {
                "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
            },
            "action_mask": {
                action: action in {"stay", "move_east"} for action in ACTION_NAMES
            },
            "requested_action": "move_east",
            "resolved_action": "stay",
            "action_valid": True,
            "resolution_action_valid": False,
            "reward": {"total": 0.8},
        }

        trace = policy.observe_transition(record)
        after = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNotNone(trace)
        self.assertEqual(trace["action"], "move_east")
        self.assertEqual(trace["requested_action"], "move_east")
        self.assertEqual(trace["resolved_action"], "stay")
        self.assertNotEqual(
            after["action_head_bias"]["move_east"],
            before["action_head_bias"]["move_east"],
        )
        self.assertEqual(
            after["action_head_bias"]["stay"],
            before["action_head_bias"]["stay"],
        )

    def test_mind_v3_policy_falls_back_to_resolved_action_for_invalid_request(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        before = policy.agent_mind_metadata(agent_id=3)
        record = {
            "agent_id": 3,
            "observation_input": {
                "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
            },
            "action_mask": {
                action: action == "stay" for action in ACTION_NAMES
            },
            "requested_action": "move_east",
            "resolved_action": "stay",
            "action_valid": False,
            "resolution_action_valid": False,
            "reward": {"total": -0.5},
        }

        trace = policy.observe_transition(record)
        after = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNotNone(trace)
        self.assertEqual(trace["action"], "stay")
        self.assertEqual(trace["requested_action"], "move_east")
        self.assertEqual(trace["resolved_action"], "stay")
        self.assertEqual(
            after["action_head_bias"]["move_east"],
            before["action_head_bias"]["move_east"],
        )
        self.assertNotEqual(
            after["action_head_bias"]["stay"],
            before["action_head_bias"]["stay"],
        )

    def test_mind_v3_policy_credits_recent_actions_from_later_reward(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        base_record = {
            "agent_id": 3,
            "observation_input": {
                "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
            },
            "action_mask": {
                action: action in {"stay", "eat", "move_east"}
                for action in ACTION_NAMES
            },
            "resolved_action": "move_east",
            "requested_action": "move_east",
            "reward": {"total": -0.2},
        }

        policy.observe_transition(base_record)
        after_move = policy.agent_mind_metadata(agent_id=3)
        eat_record = dict(base_record)
        eat_record["resolved_action"] = "eat"
        eat_record["requested_action"] = "eat"
        eat_record["reward"] = {"total": 0.8}
        trace = policy.observe_transition(eat_record)
        after_eat = policy.agent_mind_metadata(agent_id=3)

        self.assertEqual(
            trace["credit_assignment"],
            "policy_valid_requested_action_horizon_eligibility_trace_v3",
        )
        self.assertGreater(
            after_eat["action_head_bias"]["move_east"],
            after_move["action_head_bias"]["move_east"],
        )

    def test_mind_v3_policy_credits_long_water_path_from_delayed_drink(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
        policy = MindV3EvolutionPolicy(seed=9)
        policy.register_agent_mind(agent_id=3, metadata=metadata)
        base_record = {
            "agent_id": 3,
            "observation_input": {
                "values": [0.1] * OBSERVATION_INPUT_VECTOR_SIZE
            },
            "action_mask": {
                action: action in {"stay", "drink", "move_north"}
                for action in ACTION_NAMES
            },
            "before": {
                "energy_ratio": 0.7,
                "hydration_ratio": 0.32,
                "health_ratio": 0.82,
            },
            "after": {
                "energy_ratio": 0.7,
                "hydration_ratio": 0.32,
                "health_ratio": 0.82,
            },
            "outcome": {
                "reproduction_ready_after": False,
                "reproduced": False,
                "died": False,
            },
            "reward": {"total": 0.0},
        }

        for _ in range(8):
            move_record = dict(base_record)
            move_record["requested_action"] = "move_north"
            move_record["resolved_action"] = "move_north"
            policy.observe_transition(move_record)

        after_moves = policy.agent_mind_metadata(agent_id=3)
        drink_record = dict(base_record)
        drink_record["requested_action"] = "drink"
        drink_record["resolved_action"] = "drink"
        drink_record["after"] = {
            "energy_ratio": 0.69,
            "hydration_ratio": 0.72,
            "health_ratio": 0.82,
        }
        drink_record["outcome"] = {
            "drinking": {"drank": True},
            "reproduction_ready_after": False,
            "reproduced": False,
            "died": False,
        }
        drink_record["reward"] = {
            "total": 0.4,
            "components": {
                "hydration_stability": 0.4,
                "survival_continuation": 0.02,
            },
        }

        trace = policy.observe_transition(drink_record)
        after_drink = policy.agent_mind_metadata(agent_id=3)

        self.assertIsNotNone(trace)
        self.assertEqual(trace["credited_actions"][0], "drink")
        self.assertEqual(trace["credited_actions"].count("move_north"), 8)
        self.assertGreater(
            after_drink["action_head_bias"]["move_north"],
            after_moves["action_head_bias"]["move_north"],
        )

    def test_mind_v3_policy_initializes_founders_and_children(self) -> None:
        from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

        policy = MindV3EvolutionPolicy(seed=7)
        world = SimulationWorld(
            WorldConfig(seed=7, max_ticks=160, initial_agents=12, max_agents=80),
            policy=policy,
        )
        world.run(mode=RunMode.SUMMARY_ONLY)
        catalog = {
            str(agent.agent_id): {
                "parent_id": agent.parent_id,
                "mind_inheritance": agent.mind_inheritance_metadata,
            }
            for agent in world.agents.values()
        }
        founders = [
            agent for agent in catalog.values() if agent["parent_id"] is None
        ]
        children = [
            agent for agent in catalog.values() if agent["parent_id"] is not None
        ]

        self.assertTrue(founders)
        self.assertTrue(
            all(agent["mind_inheritance"]["inherited_state"] for agent in founders)
        )
        if children:
            self.assertTrue(
                all(
                    agent["mind_inheritance"]["inherited_state"]
                    for agent in children
                )
            )

    def test_trajectory_jsonl_loader_validates_header_records_and_footer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(path)

            dataset = load_trajectory_jsonl(path)

        self.assertEqual(dataset.header["type"], "header")
        self.assertEqual(dataset.header["provenance"]["source_seeds"], [7])
        self.assertEqual(dataset.header["provenance"]["split_id"], "tiny_train")
        self.assertGreater(dataset.record_count, 0)
        self.assertEqual(dataset.footer["type"], "footer")
        self.assertEqual(
            dataset.footer["trajectory_summary"]["record_count"],
            dataset.record_count,
        )
        self.assertEqual(
            dataset.footer["provenance"]["record_count"],
            dataset.record_count,
        )

    def test_trajectory_jsonl_loader_rejects_stale_footer_schema(self) -> None:
        with TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "trajectory.jsonl.gz"
            stale_path = Path(tmpdir) / "stale-trajectory.jsonl.gz"
            self._write_tiny_trajectory(original_path)
            with gzip.open(original_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            rows[-1]["summary"]["summary_schema_version"] = "stale"
            with gzip.open(stale_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            with self.assertRaisesRegex(TrajectoryDatasetError, "summary schema"):
                load_trajectory_jsonl(stale_path)

    def test_trajectory_jsonl_loader_rejects_nonfinite_policy_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "trajectory.jsonl.gz"
            stale_path = Path(tmpdir) / "nonfinite-diagnostics.jsonl.gz"
            self._write_tiny_trajectory(original_path)
            with gzip.open(original_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            rows[1]["record"]["policy_decision_diagnostics"] = {
                "score": float("nan")
            }
            with gzip.open(stale_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            with self.assertRaisesRegex(
                TrajectoryDatasetError,
                "policy_decision_diagnostics",
            ):
                load_trajectory_jsonl(stale_path)

    def test_trajectory_writer_rejects_nonfinite_policy_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "trajectory.jsonl.gz"
            rejected_path = Path(tmpdir) / "rejected.jsonl.gz"
            self._write_tiny_trajectory(original_path)
            with gzip.open(original_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            record = dict(rows[1]["record"])
            record["policy_decision_diagnostics"] = {"score": float("nan")}

            writer = JsonlTrajectoryWriter(
                rejected_path,
                source_seeds=[7],
                split_id="rejected",
                include_policy_decision_diagnostics=True,
            )
            with self.assertRaisesRegex(ValueError, "Out of range"):
                with writer:
                    writer.begin(
                        run_id="rejected",
                        config=rows[0]["config"],
                        contract=rows[0]["trajectory_contract"],
                    )
                    writer.write_record(record)

    def test_trajectory_transition_adapter_links_next_agent_observation(self) -> None:
        records = (
            {
                "tick": 0,
                "agent_id": 1,
                "observation_input": {"values": "agent1-tick0"},
                "action_mask": {"stay": True, "eat": True},
                "requested_action": "eat",
                "resolved_action": "eat",
                "resolution_action_valid": True,
                "after": {"alive": True},
                "reward": {"total": 0.25},
            },
            {
                "tick": 0,
                "agent_id": 2,
                "observation_input": {"values": "agent2-tick0"},
                "action_mask": {"stay": True, "eat": False},
                "requested_action": "stay",
                "resolved_action": "stay",
                "resolution_action_valid": True,
                "after": {"alive": True},
                "reward": {"total": 0.0},
            },
            {
                "tick": 1,
                "agent_id": 1,
                "observation_input": {"values": "agent1-tick1"},
                "action_mask": {"stay": True, "eat": False},
                "requested_action": "stay",
                "resolved_action": "stay",
                "resolution_action_valid": True,
                "after": {"alive": False},
                "reward": {"total": -1.0},
            },
        )

        transitions = build_trajectory_transitions(records)

        self.assertEqual(len(transitions), 3)
        self.assertEqual(transitions[0].agent_id, 1)
        self.assertEqual(transitions[0].action, "eat")
        self.assertEqual(transitions[0].reward_total, 0.25)
        self.assertFalse(transitions[0].done)
        self.assertEqual(
            transitions[0].next_observation_input,
            {"values": "agent1-tick1"},
        )
        self.assertEqual(
            transitions[0].next_action_mask,
            {"stay": True, "eat": False},
        )
        self.assertTrue(transitions[1].done)
        self.assertIsNone(transitions[1].next_observation_input)
        self.assertTrue(transitions[2].done)

    def test_trajectory_transition_adapter_does_not_cross_episode_boundary(self) -> None:
        records = (
            {
                "tick": 2,
                "agent_id": 1,
                "observation_input": {"values": "episode1-terminal-horizon"},
                "action_mask": {"stay": True, "eat": True},
                "requested_action": "stay",
                "resolved_action": "stay",
                "resolution_action_valid": True,
                "after": {"alive": True},
                "reward": {"total": 0.0},
            },
            {
                "tick": 0,
                "agent_id": 1,
                "observation_input": {"values": "episode2-start"},
                "action_mask": {"stay": True, "eat": False},
                "requested_action": "eat",
                "resolved_action": "stay",
                "resolution_action_valid": False,
                "after": {"alive": True},
                "reward": {"total": -0.05},
            },
        )

        transitions = build_trajectory_transitions(records)

        self.assertEqual(len(transitions), 2)
        self.assertNotEqual(transitions[0].episode_id, transitions[1].episode_id)
        self.assertTrue(transitions[0].done)
        self.assertIsNone(transitions[0].next_observation_input)
        self.assertEqual(transitions[1].action, "stay")

    def test_discounted_return_targets_follow_agent_episode_boundaries(self) -> None:
        records = (
            {
                "tick": 0,
                "agent_id": 1,
                "observation_input": {"values": "agent1-tick0"},
                "action_mask": {"stay": True},
                "requested_action": "stay",
                "resolved_action": "stay",
                "resolution_action_valid": True,
                "after": {"alive": True},
                "reward": {"total": 1.0},
            },
            {
                "tick": 0,
                "agent_id": 2,
                "observation_input": {"values": "agent2-tick0"},
                "action_mask": {"stay": True},
                "requested_action": "stay",
                "resolved_action": "stay",
                "resolution_action_valid": True,
                "after": {"alive": True},
                "reward": {"total": 3.0},
            },
            {
                "tick": 1,
                "agent_id": 1,
                "observation_input": {"values": "agent1-tick1"},
                "action_mask": {"stay": True},
                "requested_action": "stay",
                "resolved_action": "stay",
                "resolution_action_valid": True,
                "after": {"alive": False},
                "reward": {"total": 2.0},
            },
        )

        targets = discounted_return_targets(
            build_trajectory_transitions(records),
            discount=0.5,
        )

        self.assertEqual(targets, (2.0, 3.0, 2.0))

    def test_mind_horizon_labels_mark_survival_reproduction_and_censoring(self) -> None:
        def record(
            tick: int,
            agent_id: int,
            *,
            before_energy: float,
            after_energy: float,
            alive_after: bool = True,
            reproduced: bool = False,
            food_source: str | None = None,
        ) -> dict[str, object]:
            feeding = (
                {
                    "ate": True,
                    "food_source": food_source,
                    "gained_energy": 0.2,
                }
                if food_source is not None
                else {"ate": False}
            )
            return {
                "tick": tick,
                "agent_id": agent_id,
                "lineage_id": agent_id,
                "runtime_species_id": agent_id + 10,
                "runtime_ecotype_id": None,
                "before": {
                    "alive": True,
                    "energy_ratio": before_energy,
                    "hydration_ratio": 0.7,
                    "health_ratio": 0.8,
                },
                "after": {
                    "alive": alive_after,
                    "energy_ratio": after_energy,
                    "hydration_ratio": 0.65 if alive_after else 0.1,
                    "health_ratio": 0.75 if alive_after else 0.0,
                },
                "outcome": {
                    "reproduced": reproduced,
                    "died": not alive_after,
                    "resource_gain": 0.2 if food_source is not None else 0.0,
                    "feeding": feeding,
                    "passive": {
                        "hazard_damage_taken": 0.0,
                        "attack_damage_taken": 0.0,
                    },
                },
                "requested_action": "eat" if food_source is not None else "stay",
                "resolved_action": "eat" if food_source is not None else "stay",
                "action_source": "test_policy",
                "policy_id": "test_policy",
                "policy_version": "test_policy_v1",
            }

        records = (
            record(0, 1, before_energy=0.6, after_energy=0.7, food_source="carcass"),
            record(1, 1, before_energy=0.7, after_energy=0.8, reproduced=True),
            record(2, 1, before_energy=0.8, after_energy=0.75),
            record(3, 1, before_energy=0.75, after_energy=0.05, alive_after=False),
            record(0, 2, before_energy=0.9, after_energy=0.85),
            record(1, 2, before_energy=0.85, after_energy=0.8),
        )

        labels = build_horizon_label_records(records, horizons=(1, 2, 4))
        agent_one_tick_zero = next(
            label
            for label in labels
            if label["agent_id"] == 1 and label["tick"] == 0
        )
        agent_two_tick_zero = next(
            label
            for label in labels
            if label["agent_id"] == 2 and label["tick"] == 0
        )

        self.assertEqual(
            agent_one_tick_zero["schema_version"],
            MIND_HORIZON_LABEL_SCHEMA_VERSION,
        )
        self.assertTrue(agent_one_tick_zero["horizons"]["1"]["observed"])
        self.assertTrue(agent_one_tick_zero["horizons"]["1"]["survived"])
        self.assertTrue(agent_one_tick_zero["horizons"]["1"]["reproduced"])
        self.assertTrue(
            agent_one_tick_zero["horizons"]["1"]["animal_resource"][
                "animal_resource_consumed"
            ]
        )
        self.assertTrue(agent_one_tick_zero["horizons"]["4"]["observed"])
        self.assertFalse(agent_one_tick_zero["horizons"]["4"]["survived"])
        self.assertFalse(
            agent_one_tick_zero["horizons"]["4"]["animal_resource"][
                "survived_after_first_contact"
            ]
        )
        self.assertFalse(agent_two_tick_zero["horizons"]["2"]["observed"])
        self.assertTrue(agent_two_tick_zero["horizons"]["2"]["censored"])

    def test_mind_horizon_label_report_uses_trajectory_provenance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

            report = build_horizon_label_report([dataset], horizons=(1, 2))

        self.assertEqual(report["schema_version"], MIND_HORIZON_LABEL_SCHEMA_VERSION)
        self.assertEqual(report["source"]["record_count"], dataset.record_count)
        self.assertEqual(report["aggregate"]["label_count"], dataset.record_count)
        self.assertEqual(
            report["provenance"]["record_count"],
            dataset.record_count,
        )
        self.assertEqual(report["label_contract"]["horizon_ticks"], [1, 2])
        self.assertEqual(len(report["labels"]), dataset.record_count)

    def test_mind_horizon_labels_cli_writes_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "trajectory.jsonl.gz"
            output_path = tmp_path / "horizon-labels.json"
            self._write_tiny_trajectory(trajectory_path)
            stdout = io.StringIO()

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_horizon_labels",
                        "--trajectory",
                        str(trajectory_path),
                        "--horizons",
                        "1,2",
                        "--output",
                        str(output_path),
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_horizon_labels.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("schema_version=mind_horizon_labels_v1", stdout.getvalue())
        self.assertEqual(report["schema_version"], MIND_HORIZON_LABEL_SCHEMA_VERSION)
        self.assertEqual(report["label_contract"]["horizon_ticks"], [1, 2])

    def test_mind_horizon_labels_reject_empty_horizon_list(self) -> None:
        with self.assertRaises(ValueError):
            parse_horizon_ticks("1,,2")

    def test_mind_fixture_labels_extract_floor_gaps_from_gate(self) -> None:
        report = {
            "schema_version": "mind_v3_evolution_search_v1",
            "fixture_gate": {
                "min_alive": 1.0,
                "min_births": 0.0,
                "min_mixed_stable_births": 1.0,
                "min_energy_viability": 0.2,
                "min_hydration_viability": 0.2,
                "min_health_viability": 0.2,
                "min_matched_diet_viability": 0.2,
                "min_biologically_ready": 0.0,
                "per_horizon": {
                    "120": {
                        "passed": False,
                        "per_fixture": {
                            "carrion_only": {
                                "metrics": {
                                    "alive_agents_mean": 0.0,
                                    "births_mean": 2.0,
                                    "energy_viability_share_mean": 0.0,
                                    "hydration_viability_share_mean": 0.5,
                                    "health_viability_share_mean": 0.5,
                                    "matched_diet_viability_share_mean": 0.0,
                                    "biologically_ready_agents_mean": 0.0,
                                }
                            },
                            "mixed_stable": {
                                "metrics": {
                                    "alive_agents_mean": 5.0,
                                    "births_mean": 0.0,
                                    "energy_viability_share_mean": 0.4,
                                    "hydration_viability_share_mean": 0.4,
                                    "health_viability_share_mean": 0.4,
                                    "matched_diet_viability_share_mean": 0.4,
                                    "biologically_ready_agents_mean": 0.0,
                                }
                            },
                        },
                    }
                },
            },
        }

        labels = build_fixture_label_report([report])["labels"]
        carrion_alive = next(
            label
            for label in labels
            if label["fixture"] == "carrion_only"
            and label["reason"] == "fixture_alive_floor"
        )
        mixed_birth = next(
            label
            for label in labels
            if label["fixture"] == "mixed_stable"
            and label["reason"] == "fixture_mixed_stable_birth_floor"
        )

        self.assertEqual(
            carrion_alive["schema_version"],
            MIND_FIXTURE_LABEL_SCHEMA_VERSION,
        )
        self.assertFalse(carrion_alive["passed"])
        self.assertEqual(carrion_alive["ticks"], 120)
        self.assertEqual(carrion_alive["gap"], 1.0)
        self.assertEqual(carrion_alive["pressure"], 3.0)
        self.assertFalse(mixed_birth["passed"])
        self.assertEqual(mixed_birth["gap"], 1.0)

    def test_mind_fixture_labels_cli_writes_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "fixture-report.json"
            output_path = tmp_path / "fixture-labels.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evaluation_v1",
                        "fixture_gate": {
                            "min_alive": 1.0,
                            "min_births": 0.0,
                            "min_mixed_stable_births": 0.0,
                            "min_energy_viability": 0.2,
                            "min_hydration_viability": 0.2,
                            "min_health_viability": 0.2,
                            "min_matched_diet_viability": 0.2,
                            "min_biologically_ready": 0.0,
                            "blockers": [
                                {
                                    "fixture": "carrion_only",
                                    "ticks": 80,
                                    "reason": "fixture_alive_floor",
                                    "metric": "alive_agents_mean",
                                    "value": 0.0,
                                    "floor": 1.0,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_fixture_labels",
                        "--report",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_fixture_labels.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("schema_version=mind_fixture_blocker_labels_v1", stdout.getvalue())
        self.assertEqual(report["schema_version"], MIND_FIXTURE_LABEL_SCHEMA_VERSION)
        self.assertEqual(report["aggregate"]["failed_label_count"], 1)

    def test_deterministic_seed_split_is_stable_and_disjoint(self) -> None:
        first = deterministic_seed_split(
            [5, 4, 3, 2, 1, 5],
            validation_fraction=0.4,
            split_seed=17,
        )
        second = deterministic_seed_split(
            [5, 4, 3, 2, 1, 5],
            validation_fraction=0.4,
            split_seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first["train"]) & set(first["validation"]), set())
        self.assertEqual(sorted(first["train"] + first["validation"]), [1, 2, 3, 4, 5])

    def test_behavior_cloning_artifact_requires_explicit_mind_enable_flag(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "bc-artifact.json"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            with self.assertRaises(MindArtifactError):
                load_learned_policy(artifact_path)
            policy = load_learned_policy(artifact_path, enable_mind=True)

        self.assertEqual(policy.policy_id, "mind_v1_learned_policy")
        self.assertTrue(policy.heuristic_guard)
        self.assertEqual(policy.heuristic_override_min_margin, 1.0)
        self.assertTrue(policy.heuristic_delegate)
        self.assertEqual(policy.heuristic_delegate_max_training_score_margin, 0.221)
        self.assertIsNone(policy.heuristic_safe_local_eat_min_score)
        self.assertIsNone(policy.heuristic_safe_local_eat_min_food)
        self.assertIsNone(policy.heuristic_safe_local_eat_min_plant_ratio)
        self.assertIsNone(policy.heuristic_safe_plant_move_min_score)
        self.assertIsNone(policy.heuristic_safe_plant_move_min_strength)
        self.assertIsNone(policy.heuristic_safe_plant_move_max_local_food_ratio)
        self.assertIsNone(policy.heuristic_safe_plant_move_max_distance)

    def test_load_learned_policy_rejects_malformed_loaded_artifact_without_assert(
        self,
    ) -> None:
        with (
            patch(
                "evolution_sim.mind.learned_policy.load_model_artifact",
                return_value={
                    "manifest": [],
                    "model": {
                        "action_scores": {
                            action: 1.0 if action == "stay" else 0.0
                            for action in ACTION_NAMES
                        }
                    },
                },
            ),
            self.assertRaisesRegex(ValueError, "manifest"),
        ):
            load_learned_policy("unused-artifact.json", enable_mind=True)

    def test_behavior_cloning_artifact_manifest_requires_provenance(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["manifest"]["provenance"]

            with self.assertRaisesRegex(MindArtifactError, "provenance"):
                validate_model_artifact_manifest(artifact)

    def test_behavior_cloning_artifact_rejects_incomplete_action_scores(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["action_scores"]["eat"]

            with self.assertRaisesRegex(MindArtifactError, "action_scores"):
                validate_model_artifact_manifest(artifact)

    def test_behavior_cloning_artifact_requires_explicit_safe_deviation_fields(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["heuristic_safe_local_eat_min_score"]

            with self.assertRaisesRegex(
                MindArtifactError,
                "heuristic_safe_local_eat_min_score",
            ):
                validate_model_artifact_manifest(artifact)

    def test_behavior_cloning_artifact_requires_score_policy_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["conditional_score_policy"]

            with self.assertRaisesRegex(
                MindArtifactError,
                "conditional_score_policy",
            ):
                validate_model_artifact_manifest(artifact)

    def test_behavior_cloning_artifact_requires_training_weight_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            artifact = baseline.to_artifact()
            del artifact["model"]["sample_weight_policy"]

            with self.assertRaisesRegex(
                MindArtifactError,
                "sample_weight_policy",
            ):
                validate_model_artifact_manifest(artifact)

    def test_mind_train_cli_writes_valid_bc_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "bc-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)
            self.assertEqual(
                artifact["manifest"]["model_type"],
                "guarded_contextual_local_prior_bc_v2",
            )
            self.assertIn(
                "heuristic_override_min_margin",
                artifact["model"],
            )
            self.assertEqual(artifact["model"]["heuristic_override_min_margin"], 1.0)
            self.assertEqual(
                artifact["model"]["heuristic_delegate_policy"],
                "observation_heuristic_confidence_delegate_v1",
            )
            self.assertEqual(
                artifact["model"]["heuristic_delegate_max_training_score_margin"],
                0.221,
            )
            self.assertEqual(
                artifact["model"]["conditional_score_policy"],
                "smoothed_contextual_action_prior_v1",
            )
            self.assertEqual(artifact["model"]["trainer"], "contextual-prior")
            self.assertEqual(artifact["model"]["sample_weight_policy"], "uniform_v1")
            self.assertAlmostEqual(
                artifact["model"]["sample_weight_total"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertAlmostEqual(
                artifact["model"]["conditional_prior_correction_exponent"],
                0.0,
            )
            self.assertAlmostEqual(
                artifact["model"]["conditional_score_smoothing_alpha"],
                0.1,
            )
            self.assertIsNone(artifact["model"]["heuristic_safe_local_eat_min_score"])
            self.assertIsNone(artifact["model"]["heuristic_safe_plant_move_min_score"])
            self.assertEqual(
                artifact["model"]["action_score_metadata"]["record_count"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertEqual(
                artifact["manifest"]["provenance"]["record_count"],
                artifact["manifest"]["trained_record_count"],
            )

    def test_reward_weighted_baseline_prefers_higher_reward_labels(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        base_record = dict(dataset.records[0])

        def training_record(action: str, reward_total: float) -> dict[str, object]:
            record = dict(base_record)
            record["requested_action"] = action
            record["resolved_action"] = action
            record["resolution_action_valid"] = True
            reward = dict(record["reward"])
            reward["total"] = reward_total
            record["reward"] = reward
            return record

        records = [
            training_record("eat", -0.5),
            training_record("eat", -0.5),
            training_record("stay", 1.0),
        ]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)

        baseline = train_reward_weighted_behavior_cloning_baseline(
            records,
            provenance=provenance,
        )
        artifact = baseline.to_artifact()
        validate_model_artifact_manifest(artifact)

        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_reward_weighted_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["trainer"],
            "reward-weighted-contextual-prior",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "reward_total_shifted_clamp_v1",
        )
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["top_action"],
            "stay",
        )
        self.assertGreater(
            artifact["model"]["action_scores"]["stay"],
            artifact["model"]["action_scores"]["eat"],
        )
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["record_count"],
            len(records),
        )
        self.assertAlmostEqual(artifact["model"]["sample_weight_total"], 3.0)

    def test_advantage_calibrated_baseline_prefers_contextual_advantage(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        base_record = dict(dataset.records[0])

        def training_record(action: str, reward_total: float) -> dict[str, object]:
            record = dict(base_record)
            record["requested_action"] = action
            record["resolved_action"] = action
            record["resolution_action_valid"] = True
            reward = dict(record["reward"])
            reward["total"] = reward_total
            record["reward"] = reward
            return record

        records = [
            *(training_record("eat", -0.1) for _ in range(6)),
            *(training_record("stay", 0.6) for _ in range(2)),
        ]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)

        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="advantage-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        validate_model_artifact_manifest(artifact)

        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_advantage_calibrated_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["trainer"],
            "advantage-calibrated-contextual-prior",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "contextual_reward_advantage_adjusted_counts_v1",
        )
        self.assertEqual(
            artifact["model"]["reward_advantage_policy"],
            "contextual_reward_advantage_lift_v1",
        )
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["top_action"],
            "stay",
        )
        self.assertGreater(
            artifact["model"]["action_scores"]["stay"],
            artifact["model"]["action_scores"]["eat"],
        )
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["record_count"],
            len(records),
        )
        self.assertGreater(
            artifact["model"]["action_score_metadata"]["weighted_record_count"],
            0.0,
        )
        self.assertIn(
            "delegate_score_margin",
            artifact["model"]["action_score_metadata"],
        )
        self.assertLess(
            artifact["model"]["action_score_metadata"]["delegate_score_margin"],
            artifact["model"]["action_score_metadata"]["score_margin"],
        )

    def test_advantage_blended_baseline_anchors_contextual_advantage(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        base_record = dict(dataset.records[0])

        def training_record(action: str, reward_total: float) -> dict[str, object]:
            record = dict(base_record)
            record["requested_action"] = action
            record["resolved_action"] = action
            record["resolution_action_valid"] = True
            reward = dict(record["reward"])
            reward["total"] = reward_total
            record["reward"] = reward
            return record

        records = [
            *(training_record("eat", -0.5) for _ in range(6)),
            *(training_record("stay", 1.0) for _ in range(2)),
        ]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        uniform = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="contextual-prior",
        ).to_artifact()
        advantage = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="advantage-calibrated-contextual-prior",
        ).to_artifact()

        blended = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="advantage-blended-contextual-prior",
        ).to_artifact()
        validate_model_artifact_manifest(blended)

        self.assertEqual(
            blended["manifest"]["model_type"],
            "guarded_advantage_blended_contextual_prior_bc_v1",
        )
        self.assertEqual(
            blended["model"]["trainer"],
            "advantage-blended-contextual-prior",
        )
        self.assertEqual(
            blended["model"]["sample_weight_policy"],
            "contextual_reward_advantage_blended_counts_v1",
        )
        self.assertEqual(blended["model"]["reward_advantage_blend_weight"], 0.2)
        self.assertGreater(
            blended["model"]["action_scores"]["stay"],
            uniform["model"]["action_scores"]["stay"],
        )
        self.assertLess(
            blended["model"]["action_scores"]["stay"],
            advantage["model"]["action_scores"]["stay"],
        )
        self.assertIn(
            "delegate_score_margin",
            blended["model"]["action_score_metadata"],
        )

    def test_value_calibrated_baseline_writes_value_head_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        base_record = dict(dataset.records[0])

        def training_record(action: str, reward_total: float) -> dict[str, object]:
            record = dict(base_record)
            record["requested_action"] = action
            record["resolved_action"] = action
            record["resolution_action_valid"] = True
            reward = dict(record["reward"])
            reward["total"] = reward_total
            record["reward"] = reward
            return record

        records = [
            *(training_record("eat", -0.5) for _ in range(5)),
            *(training_record("stay", 1.0) for _ in range(4)),
        ]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)

        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="value-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        validate_model_artifact_manifest(artifact)

        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_value_calibrated_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["trainer"],
            "value-calibrated-contextual-prior",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "contextual_value_calibrated_score_blend_v1",
        )
        self.assertEqual(
            artifact["model"]["value_estimation_policy"],
            "mean_reward_action_value_v1",
        )
        self.assertEqual(
            artifact["model"]["value_score_blend_policy"],
            "prior_value_score_blend_v1",
        )
        self.assertAlmostEqual(artifact["model"]["value_score_blend_weight"], 0.05)
        self.assertEqual(
            artifact["model"]["value_supported_deviation_policy"],
            "positive_value_safe_deviation_v1",
        )
        self.assertEqual(
            artifact["model"]["value_supported_deviation_min_support"],
            32,
        )
        self.assertAlmostEqual(
            artifact["model"]["value_supported_deviation_min_value_margin"],
            0.04,
        )
        self.assertAlmostEqual(
            artifact["model"]["value_supported_deviation_min_learned_value"],
            0.02,
        )
        self.assertAlmostEqual(
            artifact["model"]["value_supported_deviation_min_score_margin"],
            0.25,
        )
        self.assertAlmostEqual(
            artifact["model"][
                "value_supported_deviation_min_predicted_advantage"
            ],
            0.24,
        )
        self.assertGreater(
            artifact["model"]["action_value_estimates"]["stay"],
            artifact["model"]["action_value_estimates"]["eat"],
        )
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["top_action"],
            "stay",
        )
        self.assertGreater(
            artifact["model"]["action_scores"]["stay"],
            artifact["model"]["action_scores"]["eat"],
        )
        self.assertIn(
            "delegate_score_margin",
            artifact["model"]["action_score_metadata"],
        )

    def test_value_calibrated_artifact_requires_value_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(record) for record in dataset.records[:4]]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="value-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        del artifact["model"]["action_value_estimates"]

        with self.assertRaisesRegex(
            MindArtifactError,
            "action_value_estimates",
        ):
            validate_model_artifact_manifest(artifact)

    def test_value_calibrated_artifact_requires_deviation_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(record) for record in dataset.records[:4]]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="value-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        del artifact["model"]["value_supported_deviation_policy"]

        with self.assertRaisesRegex(
            MindArtifactError,
            "value_supported_deviation_policy",
        ):
            validate_model_artifact_manifest(artifact)

    def test_value_calibrated_artifact_requires_deviation_calibration_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(record) for record in dataset.records[:4]]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="value-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        del artifact["model"]["value_supported_deviation_min_score_margin"]

        with self.assertRaisesRegex(
            MindArtifactError,
            "value_supported_deviation_min_score_margin",
        ):
            validate_model_artifact_manifest(artifact)

    def test_advantage_calibrated_artifact_requires_calibration_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(record) for record in dataset.records[:4]]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="advantage-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        del artifact["model"]["reward_advantage_policy"]

        with self.assertRaisesRegex(
            MindArtifactError,
            "reward_advantage_policy",
        ):
            validate_model_artifact_manifest(artifact)

    def test_advantage_blended_artifact_requires_blend_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(record) for record in dataset.records[:4]]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="advantage-blended-contextual-prior",
        )
        artifact = baseline.to_artifact()
        del artifact["model"]["reward_advantage_blend_weight"]

        with self.assertRaisesRegex(
            MindArtifactError,
            "reward_advantage_blend_weight",
        ):
            validate_model_artifact_manifest(artifact)

    def test_advantage_calibrated_artifact_rejects_invalid_delegate_margin(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(record) for record in dataset.records[:4]]
        provenance = dict(dataset_provenance(dataset))
        provenance["record_count"] = len(records)
        baseline = train_baseline_with_trainer(
            records,
            provenance=provenance,
            trainer="advantage-calibrated-contextual-prior",
        )
        artifact = baseline.to_artifact()
        del artifact["model"]["action_score_metadata"]["delegate_score_margin"]

        with self.assertRaisesRegex(
            MindArtifactError,
            "delegate_score_margin",
        ):
            validate_model_artifact_manifest(artifact)

        artifact = baseline.to_artifact()
        artifact["model"]["action_score_metadata"]["delegate_score_margin"] = (
            artifact["model"]["action_score_metadata"]["score_margin"] + 0.1
        )

        with self.assertRaisesRegex(
            MindArtifactError,
            "delegate_score_margin",
        ):
            validate_model_artifact_manifest(artifact)

    def test_mind_train_cli_writes_reward_weighted_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "reward-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                        "--trainer",
                        "reward-weighted-contextual-prior",
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)

        self.assertIn("trainer=reward-weighted-contextual-prior", stdout.getvalue())
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_reward_weighted_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "reward_total_shifted_clamp_v1",
        )

    def test_mind_train_cli_writes_advantage_calibrated_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "advantage-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                        "--trainer",
                        "advantage-calibrated-contextual-prior",
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)

        self.assertIn(
            "trainer=advantage-calibrated-contextual-prior",
            stdout.getvalue(),
        )
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_advantage_calibrated_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "contextual_reward_advantage_adjusted_counts_v1",
        )
        self.assertEqual(
            artifact["model"]["reward_advantage_policy"],
            "contextual_reward_advantage_lift_v1",
        )

    def test_mind_train_cli_writes_advantage_blended_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "advantage-blended-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                        "--trainer",
                        "advantage-blended-contextual-prior",
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)

        self.assertIn(
            "trainer=advantage-blended-contextual-prior",
            stdout.getvalue(),
        )
        self.assertIn(
            "reward_advantage_blend_weight=0.2",
            stdout.getvalue(),
        )
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_advantage_blended_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "contextual_reward_advantage_blended_counts_v1",
        )
        self.assertEqual(artifact["model"]["reward_advantage_blend_weight"], 0.2)

    def test_mind_train_cli_writes_value_calibrated_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "value-calibrated-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                        "--trainer",
                        "value-calibrated-contextual-prior",
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)

        self.assertIn(
            "trainer=value-calibrated-contextual-prior",
            stdout.getvalue(),
        )
        self.assertIn(
            "value_score_blend_weight=0.05",
            stdout.getvalue(),
        )
        self.assertIn(
            "value_supported_deviation_policy=positive_value_safe_deviation_v1",
            stdout.getvalue(),
        )
        self.assertIn(
            "value_supported_deviation_min_support=32",
            stdout.getvalue(),
        )
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_value_calibrated_contextual_prior_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "contextual_value_calibrated_score_blend_v1",
        )
        self.assertEqual(
            artifact["model"]["value_estimation_policy"],
            "mean_reward_action_value_v1",
        )

    def test_neural_actor_critic_trainer_writes_deterministic_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            first = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="neural-actor-critic-bc",
            ).to_artifact()
            second = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="neural-actor-critic-bc",
            ).to_artifact()

        validate_model_artifact_manifest(first)
        diagnostics = build_artifact_diagnostics(first, [dataset])
        self.assertEqual(first, second)
        self.assertEqual(
            first["manifest"]["model_type"],
            "guarded_neural_actor_critic_bc_v1",
        )
        self.assertEqual(first["model"]["trainer"], "neural-actor-critic-bc")
        self.assertEqual(
            first["model"]["sample_weight_policy"],
            "neural_actor_critic_bc_uniform_v1",
        )
        self.assertEqual(
            first["model"]["neural_backend"],
            "pure_python_deterministic_v1",
        )
        self.assertEqual(
            first["model"]["neural_architecture"],
            "fixed_random_feature_mlp_actor_critic_v1",
        )
        self.assertEqual(
            first["model"]["neural_input_size"],
            OBSERVATION_INPUT_VECTOR_SIZE,
        )
        network = first["model"]["neural_network"]
        self.assertEqual(set(network["actor_output_weights"]), set(ACTION_NAMES))
        self.assertEqual(set(network["action_value_output_weights"]), set(ACTION_NAMES))
        self.assertEqual(len(network["hidden_weights"]), 8)
        self.assertEqual(len(network["hidden_weights"][0]), OBSERVATION_INPUT_VECTOR_SIZE)
        self.assertIn("neural_calibration", diagnostics)
        neural_calibration = diagnostics["neural_calibration"]
        self.assertEqual(
            neural_calibration["record_count"],
            dataset.record_count,
        )
        self.assertGreaterEqual(neural_calibration["actor_top1_accuracy"], 0.0)
        self.assertLessEqual(neural_calibration["actor_top1_accuracy"], 1.0)
        self.assertIn("by_score_margin_bucket", neural_calibration)
        self.assertIn("by_action_value_margin_bucket", neural_calibration)
        self.assertIn("by_predicted_advantage_bucket", neural_calibration)
        self.assertIn("action_value_mean_abs_error", neural_calibration)
        self.assertIn("state_value_mean_abs_error", neural_calibration)
        self.assertEqual(
            neural_calibration["critic_return_target_policy"],
            "discounted_return_target_v1",
        )
        self.assertIn("action_value_return_mean_abs_error", neural_calibration)
        self.assertIn("state_value_return_mean_abs_error", neural_calibration)
        self.assertEqual(
            neural_calibration["score_normalization_policy"],
            "mask_renormalized_neural_actor_scores_v1",
        )
        self.assertIn("viability_calibration", diagnostics)
        viability_calibration = diagnostics["viability_calibration"]
        self.assertEqual(
            viability_calibration["schema_version"],
            "mind_viability_critic_diagnostics_v0",
        )
        self.assertEqual(
            viability_calibration["runtime_decision_policy"],
            "diagnostics_only_not_used_for_runtime_v0",
        )
        self.assertEqual(
            viability_calibration["record_count"],
            dataset.record_count,
        )
        self.assertEqual(
            viability_calibration["risk_score_policy"],
            "normalized_action_value_constraint_risk_proxy_v0",
        )
        self.assertEqual(
            set(viability_calibration["component_names"]),
            set(VIABILITY_COMPONENT_NAMES),
        )
        self.assertIn("risk_brier_score", viability_calibration)
        self.assertIn("risk_auc", viability_calibration)
        self.assertIn("by_component", viability_calibration)
        self.assertIn("by_logged_action", viability_calibration)
        self.assertIn("by_trophic_role", viability_calibration)
        self.assertIn("by_meat_mode", viability_calibration)
        json.dumps(first)

    def test_neural_artifact_diagnostics_shards_match_serial(self) -> None:
        with TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "seed7.jsonl.gz"
            second_path = Path(tmpdir) / "seed8.jsonl.gz"
            self._write_tiny_trajectory(first_path, seed=7)
            self._write_tiny_trajectory(second_path, seed=8)
            first_dataset = load_trajectory_jsonl(first_path)
            second_dataset = load_trajectory_jsonl(second_path)
            datasets = [first_dataset, second_dataset]
            artifact = train_baseline_with_trainer(
                records_with_trajectory_context(datasets),
                provenance=combined_dataset_provenance(datasets),
                trainer="neural-actor-critic-bc",
            ).to_artifact()

            serial = build_artifact_diagnostics(artifact, datasets)
            sharded = finalize_artifact_diagnostics_shards(
                artifact,
                [
                    build_artifact_diagnostics_shard_stats(artifact, [dataset])
                    for dataset in datasets
                ],
            )

        self.assertEqual(sharded, serial)
        self.assertIn("neural_calibration", sharded)
        self.assertIn("viability_calibration", sharded)

    def test_neural_trainer_rejects_malformed_network_without_assert(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

            with (
                patch(
                    "evolution_sim.mind.baseline.train_neural_actor_critic_network",
                    return_value={
                        "action_value_output_bias": [],
                    },
                ),
                self.assertRaisesRegex(ValueError, "action_value_output_bias"),
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="neural-actor-critic-bc",
                )

    def test_neural_artifact_rejects_incomplete_weight_shapes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="neural-actor-critic-bc",
            ).to_artifact()
            artifact["model"]["neural_network"]["hidden_weights"][0].pop()

        with self.assertRaisesRegex(MindArtifactError, "neural_network"):
            validate_model_artifact_manifest(artifact)

    def test_torch_actor_critic_trainer_reports_missing_optional_ml_stack(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

            with (
                patch(
                    "evolution_sim.mind.torch_trainer._load_torch",
                    side_effect=RuntimeError(
                        "PyTorch Mind training requires optional ML dependencies. "
                        "Install them with: python3 -m pip install "
                        "-r requirements-mind-ml.txt"
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "requirements-mind-ml.txt"),
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="torch-actor-critic-bc",
                )

    def test_mind_train_parser_accepts_torch_device_flag(self) -> None:
        args = mind_train.build_parser().parse_args(
            [
                "--trajectory",
                "trajectory.jsonl.gz",
                "--trainer",
                "torch-discrete-iql",
                "--torch-device",
                "auto",
            ]
        )

        self.assertEqual(args.torch_device, "auto")

    def test_mind_train_parser_accepts_counterfactual_label_flags(self) -> None:
        args = mind_train.build_parser().parse_args(
            [
                "--trajectory",
                "trajectory.jsonl.gz",
                "--trainer",
                "torch-discrete-iql",
                "--torch-iql-counterfactual-labels",
                "labels.json",
                "--torch-iql-counterfactual-label-weight-scale",
                "3.5",
            ]
        )

        self.assertEqual(args.torch_iql_counterfactual_labels, Path("labels.json"))
        self.assertEqual(args.torch_iql_counterfactual_label_weight_scale, 3.5)

    def test_mind_gate_parser_accepts_torch_device_flag(self) -> None:
        args = mind_gate.build_parser().parse_args(
            [
                "--trainer",
                "torch-discrete-iql",
                "--torch-device",
                "cuda",
            ]
        )

        self.assertEqual(args.torch_device, "cuda")

    def test_mind_gate_parser_accepts_counterfactual_label_flags(self) -> None:
        args = mind_gate.build_parser().parse_args(
            [
                "--trainer",
                "torch-discrete-iql",
                "--torch-iql-counterfactual-labels",
                "labels.json",
                "--torch-iql-counterfactual-label-weight-scale",
                "2.25",
            ]
        )

        self.assertEqual(args.torch_iql_counterfactual_labels, Path("labels.json"))
        self.assertEqual(args.torch_iql_counterfactual_label_weight_scale, 2.25)

    def test_mind_gate_parser_accepts_evaluation_workers_flag(self) -> None:
        args = mind_gate.build_parser().parse_args(
            [
                "--evaluation-workers",
                "3",
            ]
        )

        self.assertEqual(args.evaluation_workers, 3)

    def test_mind_gate_parser_accepts_artifact_diagnostics_workers_flag(self) -> None:
        args = mind_gate.build_parser().parse_args(
            [
                "--artifact-diagnostics-workers",
                "2",
            ]
        )

        self.assertEqual(args.artifact_diagnostics_workers, 2)

    def test_torch_device_flag_is_rejected_for_non_torch_trainer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

            with self.assertRaisesRegex(ValueError, "torch_device"):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_device="cuda",
                )

    def test_torch_counterfactual_labels_are_rejected_for_non_iql_trainer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

            with self.assertRaisesRegex(ValueError, "counterfactual_label_report"):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_counterfactual_label_report={
                        "schema_version": (
                            MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION
                        ),
                        "labels": [],
                    },
                )

    def test_torch_counterfactual_labels_weight_matching_training_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            records = tuple(records_with_trajectory_context([dataset]))
            transitions = build_trajectory_transitions(records)
            action = transitions[0].action
            label_report = {
                "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
                "source": {"trajectory_paths": [str(trajectory_path)]},
                "aggregate": {"label_count": 1},
                "labels": [
                    {
                        "episode_id": records[0][TRAJECTORY_EPISODE_ID_FIELD],
                        "dataset_record_index": records[0][
                            TRAJECTORY_DATASET_RECORD_INDEX_FIELD
                        ],
                        "trajectory_path": records[0][TRAJECTORY_SOURCE_PATH_FIELD],
                        "tick": records[0]["tick"],
                        "agent_id": records[0]["agent_id"],
                        "source_script": "hydration_safe_carrion_cycle",
                        "action_support": {
                            "logged_action": action,
                            "logged_action_legal": True,
                        },
                        "rollout_terminal_target": {
                            "terminal_alive": True,
                            "terminal_state": {
                                "energy_ratio": 0.72,
                                "hydration_ratio": 0.81,
                                "health_ratio": 0.91,
                                "matched_diet_ratio": 1.0,
                            },
                            "animal_resource_gain_to_terminal": 1.25,
                            "action_value": {"score": 0.75},
                        },
                    }
                ],
            }

        supervision = mind_torch_trainer._counterfactual_label_supervision(
            records,
            transitions,
            label_report,
            action_index={
                action_name: index
                for index, action_name in enumerate(ACTION_NAMES)
            },
            weight_scale=2.0,
        )

        self.assertEqual(
            supervision["diagnostics"]["matched_record_count"],
            1,
        )
        self.assertEqual(
            supervision["diagnostics"]["source_script_counts"],
            {"hydration_safe_carrion_cycle": 1},
        )
        self.assertEqual(supervision["sample_weights"][0], 2.5)
        self.assertEqual(supervision["loss_weights"][0], 2.5)
        self.assertEqual(supervision["action_value_targets"][0], 0.75)
        self.assertTrue(
            all(weight == 1.0 for weight in supervision["sample_weights"][1:])
        )
        logged_action_index = ACTION_NAMES.index(action)
        self.assertEqual(
            supervision["action_viability_observed"][0][logged_action_index],
            tuple(1.0 for _ in VIABILITY_COMPONENT_NAMES),
        )
        self.assertEqual(
            supervision["viability_targets"][0],
            tuple(0.0 for _ in VIABILITY_COMPONENT_NAMES),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_actor_critic_trainer_writes_real_ml_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-actor-critic-bc",
            ).to_artifact()

        validate_model_artifact_manifest(artifact)
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_torch_actor_critic_bc_v1",
        )
        self.assertEqual(artifact["model"]["trainer"], "torch-actor-critic-bc")
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "torch_actor_critic_bc_uniform_v1",
        )
        self.assertEqual(artifact["model"]["neural_backend"], "pytorch_optional_v1")
        self.assertEqual(
            artifact["model"]["neural_architecture"],
            "torch_mlp_actor_critic_v1",
        )
        network = artifact["model"]["neural_network"]
        self.assertEqual(len(network["hidden_weights"]), 256)
        self.assertIn("training_metrics", network)
        training_metrics = network["training_metrics"]
        self.assertIsInstance(training_metrics, dict)
        dependency_versions = training_metrics["ml_dependency_versions"]
        self.assertIsInstance(dependency_versions, dict)
        self.assertIn("torch", dependency_versions)
        self.assertIsNotNone(dependency_versions["torch"])
        self.assertIn("python_version", training_metrics)
        self.assertGreaterEqual(training_metrics["actor_accuracy"], 0.0)
        self.assertLessEqual(training_metrics["actor_accuracy"], 1.0)
        self.assertGreaterEqual(training_metrics["actor_mean_top_margin"], 0.0)
        self.assertEqual(
            set(training_metrics["action_counts"]),
            set(ACTION_NAMES),
        )
        self.assertEqual(
            set(training_metrics["action_accuracy"]),
            set(ACTION_NAMES),
        )
        self.assertIn("action_value_mean_abs_error", training_metrics)
        self.assertIn("state_value_mean_abs_error", training_metrics)
        json.dumps(artifact, allow_nan=False)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_trainer_records_requested_device_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-actor-critic-bc",
                torch_device="cpu",
            ).to_artifact()

        device_metadata = artifact["model"]["torch_device_metadata"]
        self.assertEqual(device_metadata["policy"], "torch_device_resolution_v1")
        self.assertEqual(device_metadata["requested_device"], "cpu")
        self.assertEqual(device_metadata["resolved_device"], "cpu")
        self.assertIn("cuda_available", device_metadata)
        self.assertIn("mps_available", device_metadata)
        self.assertEqual(
            artifact["model"]["neural_network"]["torch_device_metadata"],
            device_metadata,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_advantage_actor_critic_trainer_writes_real_ml_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-advantage-actor-critic-bc",
            ).to_artifact()

        validate_model_artifact_manifest(artifact)
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_torch_advantage_actor_critic_bc_v1",
        )
        self.assertEqual(
            artifact["model"]["trainer"],
            "torch-advantage-actor-critic-bc",
        )
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "torch_advantage_actor_critic_bc_contextual_advantage_weighted_v1",
        )
        self.assertEqual(
            artifact["model"]["neural_training_policy"],
            "adamw_contextual_advantage_weighted_actor_critic_v1",
        )
        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(training_metrics["advantage_weighted"])
        self.assertEqual(
            training_metrics["actor_weighting_policy"],
            "contextual_advantage_weighted_cross_entropy_v1",
        )
        self.assertIn("advantage_contextual_record_rate", training_metrics)
        self.assertGreaterEqual(
            training_metrics["advantage_sample_weight_min_observed"],
            0.0,
        )
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_trainer_writes_real_ml_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            records = [dict(record) for record in dataset.records]
            hard_feedback_record_found = False
            delegate_feedback_record_found = False
            for record in records:
                label = (
                    str(record["requested_action"])
                    if bool(record.get("resolution_action_valid", False))
                    else str(record["resolved_action"])
                )
                action_mask = record["action_mask"]
                if not isinstance(action_mask, dict):
                    continue
                alternatives = [
                    action
                    for action in ACTION_NAMES
                    if action != label and bool(action_mask.get(action, False))
                ]
                if not alternatives:
                    continue
                record["action_source"] = (
                    "mind_v2_neural_policy:"
                    "observation_heuristic_confidence_delegate_v1"
                )
                record["policy_decision_diagnostics"] = {
                    "heuristic_delegate_used": True,
                    "learned_action": alternatives[0],
                    "heuristic_action": label,
                }
                record["policy_update_trace"] = {
                    "schema_version": "mind_policy_update_trace_v1",
                    "policy": "in_run_contextual_bandit_adapter_v1",
                    "update_index": 1,
                    "context_key": "test-context",
                    "action": label,
                    "reward_total": -1.0,
                    "reward_signal": -1.0,
                    "previous_adjustment": 0.0,
                    "updated_adjustment": -0.05,
                    "learning_rate": 0.05,
                    "min_adjustment": -0.5,
                    "max_adjustment": 0.5,
                }
                feedback_record_found = True
                break
            self.assertTrue(feedback_record_found)
            artifact = train_baseline_with_trainer(
                tuple(records),
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
            ).to_artifact()

        validate_model_artifact_manifest(artifact)
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_torch_discrete_iql_v1",
        )
        self.assertEqual(artifact["model"]["trainer"], "torch-discrete-iql")
        self.assertEqual(
            artifact["model"]["sample_weight_policy"],
            "torch_discrete_iql_transition_expectile_awbc_v1",
        )
        self.assertEqual(
            artifact["model"]["neural_training_policy"],
            "adamw_discrete_iql_expectile_advantage_weighted_v1",
        )
        self.assertEqual(
            artifact["model"]["neural_actor_prior_policy"],
            "contextual_prior_score_anchor_v1",
        )
        self.assertAlmostEqual(
            artifact["model"]["neural_actor_prior_blend_weight"],
            0.9,
        )
        self.assertEqual(
            artifact["model"]["heuristic_delegate_max_training_score_margin"],
            0.25,
        )
        self.assertNotIn("reward_advantage_blend_policy", artifact["model"])
        self.assertNotIn("value_supported_deviation_policy", artifact["model"])
        self.assertNotIn(
            "value_supported_deviation_min_score_margin",
            artifact["model"],
        )
        network = artifact["model"]["neural_network"]
        self.assertEqual(
            set(network["viability_component_output_weights"]),
            set(VIABILITY_COMPONENT_NAMES),
        )
        self.assertEqual(
            set(network["viability_component_output_bias"]),
            set(VIABILITY_COMPONENT_NAMES),
        )
        self.assertEqual(
            set(network["action_viability_component_output_weights"]),
            set(ACTION_NAMES),
        )
        self.assertEqual(
            set(network["action_viability_component_output_bias"]),
            set(ACTION_NAMES),
        )
        self.assertEqual(
            set(
                network["action_viability_component_output_weights"]["eat"]
            ),
            set(VIABILITY_COMPONENT_NAMES),
        )
        training_metrics = network["training_metrics"]
        self.assertEqual(
            training_metrics["critic_policy"],
            "td0_expectile_q_v_v1",
        )
        self.assertEqual(
            training_metrics["critic_regularization_policy"],
            "cql_masked_legal_logsumexp_selected_action_gap_head_only_v1",
        )
        self.assertIn("final_cql_loss", training_metrics)
        self.assertIn("cql_gap_mean", training_metrics)
        self.assertFalse(training_metrics["critic_regularization_enabled"])
        self.assertEqual(training_metrics["cql_loss_weight"], 0.0)
        self.assertGreater(training_metrics["cql_temperature"], 0.0)
        self.assertEqual(
            training_metrics["actor_weighting_policy"],
            "iql_masked_advantage_weighted_bc_v1",
        )
        self.assertFalse(training_metrics["actor_advantage_calibration_enabled"])
        self.assertIsNone(training_metrics["actor_advantage_calibration_policy"])
        self.assertIn("actor_sample_weight_mean", training_metrics)
        self.assertIn("actor_advantage_scale", training_metrics)
        self.assertEqual(
            training_metrics["behavior_anchor_policy"],
            "replay_weighted_behavior_cross_entropy_anchor_v1",
        )
        self.assertIn("final_behavior_anchor_loss", training_metrics)
        self.assertFalse(training_metrics["behavior_anchor_enabled"])
        self.assertEqual(training_metrics["behavior_anchor_loss_weight"], 0.0)
        self.assertEqual(
            training_metrics["behavior_margin_anchor_policy"],
            "viability_safe_logged_action_margin_anchor_v1",
        )
        self.assertFalse(training_metrics["behavior_margin_anchor_enabled"])
        self.assertEqual(training_metrics["behavior_margin_anchor_loss_weight"], 0.0)
        self.assertIn("final_behavior_margin_anchor_loss", training_metrics)
        self.assertGreater(
            training_metrics["behavior_margin_anchor_eligible_count"],
            0,
        )
        self.assertEqual(
            training_metrics["guard_feedback_policy"],
            "runtime_suppressed_learned_action_margin_penalty_v1",
        )
        self.assertGreaterEqual(training_metrics["guard_feedback_count"], 1)
        self.assertGreaterEqual(training_metrics["guard_feedback_rate"], 0.0)
        self.assertIn("final_guard_feedback_loss", training_metrics)
        self.assertEqual(
            training_metrics["online_update_feedback_policy"],
            "replayable_online_update_signed_actor_margin_v1",
        )
        self.assertIn("final_online_update_feedback_loss", training_metrics)
        self.assertGreaterEqual(
            training_metrics["online_update_feedback_count"],
            1,
        )
        self.assertGreaterEqual(
            training_metrics["online_update_feedback_negative_count"],
            1,
        )
        self.assertEqual(
            training_metrics["learned_replay_weight_policy"],
            "learned_rollout_self_action_downweighted_guard_feedback_v1",
        )
        self.assertGreaterEqual(
            training_metrics["learned_replay_record_count"],
            1,
        )
        self.assertGreaterEqual(
            training_metrics["replay_sample_weight_min_observed"],
            0.0,
        )
        self.assertGreater(training_metrics["transition_count"], 0)
        self.assertIn("iql_expectile", training_metrics)
        self.assertIn("iql_discount", training_metrics)
        self.assertIn("q_value_mean_abs_error", training_metrics)
        self.assertIn("q_value_return_mean_abs_error", training_metrics)
        self.assertIn("state_value_mean_abs_error", training_metrics)
        self.assertIn("state_value_return_mean_abs_error", training_metrics)
        self.assertFalse(training_metrics["critic_return_calibration_enabled"])
        self.assertEqual(training_metrics["critic_return_calibration_loss_weight"], 0.0)
        self.assertFalse(
            training_metrics["critic_suppression_calibration_enabled"]
        )
        self.assertEqual(
            training_metrics["critic_suppression_calibration_loss_weight"],
            0.0,
        )
        self.assertIn(
            "final_suppression_critic_calibration_loss",
            training_metrics,
        )
        self.assertFalse(
            training_metrics["actor_action_distribution_regularization_enabled"]
        )
        self.assertEqual(
            training_metrics["actor_action_distribution_loss_weight"],
            0.0,
        )
        self.assertIn("final_action_distribution_loss", training_metrics)
        self.assertFalse(training_metrics["actor_constraint_awareness_enabled"])
        self.assertEqual(
            training_metrics["actor_constraint_awareness_policy"],
            "observed_viability_safe_iql_actor_weight_filter_v1",
        )
        self.assertEqual(training_metrics["actor_constraint_weight_min"], 1.0)
        self.assertIn(
            "actor_constraint_risky_logged_action_count",
            training_metrics,
        )
        self.assertFalse(training_metrics["actor_risk_adjusted_extraction_enabled"])
        self.assertEqual(training_metrics["actor_risk_adjusted_loss_weight"], 0.0)
        self.assertEqual(
            training_metrics["actor_risk_adjusted_extraction_policy"],
            "detached_q_minus_action_viability_risk_actor_distillation_v1",
        )
        self.assertEqual(
            training_metrics["actor_risk_score_policy"],
            "detached_max_non_suppression_action_viability_risk_v1",
        )
        self.assertIn("final_risk_adjusted_actor_loss", training_metrics)
        self.assertFalse(
            training_metrics["actor_calibrated_supported_extraction_enabled"]
        )
        self.assertEqual(
            training_metrics["actor_calibrated_supported_loss_weight"],
            0.0,
        )
        self.assertIn(
            "actor_calibrated_supported_calibration",
            training_metrics,
        )
        self.assertIn(
            "actor_calibrated_supported_rejection_reasons",
            training_metrics,
        )
        self.assertTrue(training_metrics["viability_head_enabled"])
        self.assertEqual(
            training_metrics["viability_head_policy"],
            "multi_component_constraint_viability_head_v1",
        )
        self.assertEqual(
            training_metrics["viability_representation_policy"],
            "shared_hidden_auxiliary_heads_v1",
        )
        self.assertEqual(
            set(training_metrics["viability_component_names"]),
            set(VIABILITY_COMPONENT_NAMES),
        )
        self.assertIn("final_viability_loss", training_metrics)
        self.assertEqual(
            training_metrics["viability_pos_weight_policy"],
            "observed_state_component_balance_v1",
        )
        self.assertEqual(
            training_metrics["viability_suppression_supervision_policy"],
            "suppression_owned_by_action_viability_head_v1",
        )
        self.assertEqual(
            training_metrics["viability_component_observed_counts"][
                VIABILITY_SUPPRESSION_COMPONENT
            ],
            0,
        )
        self.assertEqual(
            training_metrics["viability_suppression_observed_count"],
            0,
        )
        self.assertEqual(
            training_metrics["viability_pos_weights"][
                VIABILITY_SUPPRESSION_COMPONENT
            ],
            1.0,
        )
        self.assertTrue(training_metrics["action_viability_head_enabled"])
        self.assertEqual(
            training_metrics["action_viability_head_policy"],
            "action_conditioned_multi_component_constraint_viability_head_v1",
        )
        self.assertEqual(
            training_metrics["action_viability_supervision_policy"],
            "logged_action_components_plus_learned_action_suppression_v2",
        )
        self.assertEqual(
            training_metrics["action_viability_pos_weight_policy"],
            "observed_action_component_balance_v1",
        )
        self.assertIn("final_action_viability_loss", training_metrics)
        self.assertGreater(
            training_metrics["action_viability_observed_component_count"],
            0,
        )
        self.assertIn(
            "action_viability_component_observed_counts",
            training_metrics,
        )
        self.assertGreaterEqual(
            training_metrics["action_viability_suppression_positive_count"],
            1,
        )
        self.assertGreaterEqual(
            training_metrics["action_viability_suppression_observed_count"],
            training_metrics["action_viability_suppression_positive_count"],
        )
        self.assertEqual(
            training_metrics[
                "action_viability_logged_suppression_positive_count"
            ],
            0,
        )
        self.assertIn("viability_component_positive_counts", training_metrics)
        self.assertIn(
            "viability_target_component_positive_counts",
            training_metrics,
        )
        self.assertEqual(
            training_metrics["action_viability_pos_weights"][
                VIABILITY_SUPPRESSION_COMPONENT
            ],
            1.0,
        )
        modified_dataset = dataset.__class__(
            path=dataset.path,
            header=dataset.header,
            records=tuple(records),
            footer=dataset.footer,
        )
        diagnostics = build_artifact_diagnostics(artifact, [modified_dataset])
        viability_calibration = diagnostics["viability_calibration"]
        self.assertEqual(
            viability_calibration["risk_score_policy"],
            "action_conditioned_multi_component_constraint_viability_head_v1",
        )
        self.assertEqual(
            viability_calibration["target_policy"],
            "survival_floor_invalid_suppression_constraint_target_v0",
        )
        self.assertEqual(
            viability_calibration["viability_head_conditioning"],
            "action",
        )
        self.assertEqual(
            viability_calibration["action_conditioned_score_policy"],
            "logged_action_components_plus_learned_action_suppression_v2",
        )
        self.assertFalse(viability_calibration["action_conditioned_logged_action_only"])
        self.assertGreaterEqual(
            viability_calibration[
                "action_conditioned_suppressed_action_score_count"
            ],
            1,
        )
        self.assertIn("by_component", viability_calibration)
        malformed = json.loads(json.dumps(artifact))
        del malformed["model"]["neural_network"]["viability_component_output_bias"]
        with self.assertRaisesRegex(MindArtifactError, "viability component"):
            validate_model_artifact_manifest(malformed)
        malformed = json.loads(json.dumps(artifact))
        del malformed["model"]["neural_network"][
            "action_viability_component_output_bias"
        ]
        with self.assertRaisesRegex(MindArtifactError, "action viability component"):
            validate_model_artifact_manifest(malformed)
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_detached_viability_heads_are_opt_in(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_detach_viability_heads=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertEqual(
            training_metrics["viability_representation_policy"],
            "detached_shared_hidden_auxiliary_heads_v1",
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_behavior_margin_anchor_is_opt_in(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_behavior_margin_anchor=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(training_metrics["behavior_margin_anchor_enabled"])
        self.assertGreater(
            training_metrics["behavior_margin_anchor_loss_weight"],
            0.0,
        )
        self.assertGreater(
            training_metrics["behavior_margin_anchor_target"],
            0.0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_neural_prior_blend_weight_is_opt_in(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_neural_actor_prior_blend_weight=0.85,
            ).to_artifact()

        validate_model_artifact_manifest(artifact)
        self.assertAlmostEqual(
            artifact["model"]["neural_actor_prior_blend_weight"],
            0.85,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_calibrated_actor_extraction_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_calibrated_actor_extraction=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertEqual(
            training_metrics["actor_weighting_policy"],
            "behavior_anchored_batch_standardized_iql_advantage_weighted_bc_v2",
        )
        self.assertTrue(training_metrics["actor_advantage_calibration_enabled"])
        self.assertEqual(
            training_metrics["actor_advantage_calibration_policy"],
            "behavior_anchored_batch_standardized_iql_advantage_weighted_bc_v2",
        )
        self.assertGreater(
            training_metrics["actor_advantage_calibration_temperature"],
            0.0,
        )
        self.assertGreater(
            training_metrics["actor_advantage_calibration_weight_blend"],
            0.0,
        )
        self.assertLess(
            training_metrics["actor_advantage_calibration_weight_blend"],
            1.0,
        )
        self.assertGreater(
            training_metrics["actor_advantage_calibration_weight_min"],
            0.0,
        )

    def test_torch_discrete_iql_calibrated_actor_extraction_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_calibrated_actor_extraction",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_calibrated_actor_extraction=True,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_return_calibration_is_opt_in(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_return_calibration=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(training_metrics["critic_return_calibration_enabled"])
        self.assertEqual(
            training_metrics["critic_return_calibration_policy"],
            "discounted_return_q_v_auxiliary_v1",
        )
        self.assertGreater(
            training_metrics["critic_return_calibration_loss_weight"],
            0.0,
        )
        self.assertIn("q_value_return_mean_abs_error", training_metrics)
        self.assertIn("state_value_return_mean_abs_error", training_metrics)

    def test_torch_discrete_iql_return_calibration_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_return_calibration",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_return_calibration=True,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_suppression_critic_calibration_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            records = [dict(record) for record in dataset.records]
            feedback_record_found = False
            for record in records:
                label = (
                    str(record["requested_action"])
                    if bool(record.get("resolution_action_valid", False))
                    else str(record["resolved_action"])
                )
                action_mask = record["action_mask"]
                if not isinstance(action_mask, dict):
                    continue
                alternatives = [
                    action
                    for action in ACTION_NAMES
                    if action != label and bool(action_mask.get(action, False))
                ]
                if not alternatives:
                    continue
                record["action_source"] = (
                    "mind_v2_neural_policy:"
                    "observation_heuristic_confidence_delegate_v1"
                )
                record["policy_decision_diagnostics"] = {
                    "heuristic_delegate_used": True,
                    "heuristic_action": label,
                    "learned_action": alternatives[0],
                }
                feedback_record_found = True
                break
            self.assertTrue(feedback_record_found)
            artifact = train_baseline_with_trainer(
                tuple(records),
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_suppression_critic_calibration=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics["critic_suppression_calibration_enabled"]
        )
        self.assertEqual(
            training_metrics["critic_suppression_calibration_policy"],
            "runtime_suppressed_action_q_v_margin_calibration_v1",
        )
        self.assertGreater(
            training_metrics["critic_suppression_calibration_loss_weight"],
            0.0,
        )
        self.assertIn(
            "final_suppression_critic_calibration_loss",
            training_metrics,
        )
        self.assertGreaterEqual(
            training_metrics["critic_suppression_calibration_count"],
            1,
        )

    def test_torch_discrete_iql_suppression_critic_calibration_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_suppression_critic_calibration",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_suppression_critic_calibration=True,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_action_distribution_loss_prefers_matching_marginal(
        self,
    ) -> None:
        import torch

        labels = torch.tensor([1, 1, 2, 3], dtype=torch.long)
        weights = torch.ones(4, dtype=torch.float32)
        mask = torch.zeros((4, len(ACTION_NAMES)), dtype=torch.bool)
        mask[:, 1] = True
        mask[:, 2] = True
        mask[:, 3] = True
        matching_logits = torch.full((4, len(ACTION_NAMES)), -8.0)
        distribution_temperature = (
            mind_torch_trainer.TORCH_IQL_ACTION_DISTRIBUTION_TEMPERATURE
        )
        matching_logits[:, 1] = distribution_temperature * math.log(0.5)
        matching_logits[:, 2] = distribution_temperature * math.log(0.25)
        matching_logits[:, 3] = distribution_temperature * math.log(0.25)
        collapsed_logits = torch.full((4, len(ACTION_NAMES)), -8.0)
        collapsed_logits[:, 1] = 4.0
        collapsed_logits[:, 2] = -4.0
        collapsed_logits[:, 3] = -4.0

        matching_loss, matching_stats = (
            mind_torch_trainer._action_distribution_actor_loss(
                torch,
                matching_logits,
                mask,
                labels,
                weights,
            )
        )
        collapsed_loss, collapsed_stats = (
            mind_torch_trainer._action_distribution_actor_loss(
                torch,
                collapsed_logits,
                mask,
                labels,
                weights,
            )
        )

        self.assertLess(float(matching_loss.item()), float(collapsed_loss.item()))
        self.assertLess(
            matching_stats["action_distribution_tvd"],
            collapsed_stats["action_distribution_tvd"],
        )
        self.assertEqual(
            matching_stats["schema_version"],
            "mind_action_distribution_actor_regularization_v1",
        )
        self.assertEqual(
            matching_stats["action_distribution_temperature"],
            distribution_temperature,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_action_distribution_regularization_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_action_distribution_regularization=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics["actor_action_distribution_regularization_enabled"]
        )
        self.assertEqual(
            training_metrics["actor_action_distribution_regularization_policy"],
            "batch_logged_sharp_action_marginal_kl_v2",
        )
        self.assertGreater(
            training_metrics["actor_action_distribution_loss_weight"],
            0.0,
        )
        self.assertIn("final_action_distribution_loss", training_metrics)
        self.assertEqual(
            training_metrics["actor_action_distribution"][
                "schema_version"
            ],
            "mind_action_distribution_actor_regularization_v1",
        )

    def test_torch_discrete_iql_action_distribution_regularization_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_action_distribution_regularization",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_action_distribution_regularization=True,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_constraint_aware_actor_extraction_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_constraint_aware_actor_extraction=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(training_metrics["actor_constraint_awareness_enabled"])
        self.assertEqual(
            training_metrics["actor_constraint_awareness_policy"],
            "observed_viability_safe_iql_actor_weight_filter_v1",
        )
        self.assertEqual(
            training_metrics["actor_constraint_component_policy"],
            "logged_action_observed_non_suppression_components_v1",
        )
        self.assertGreaterEqual(
            training_metrics["actor_constraint_weight_mean"],
            training_metrics["actor_constraint_weight_min"],
        )
        self.assertGreaterEqual(
            training_metrics["actor_constraint_risky_logged_action_count"],
            0,
        )
        self.assertIn(
            "death_or_survival_horizon_risk",
            training_metrics[
                "actor_constraint_risky_logged_action_component_counts"
            ],
        )

    def test_torch_discrete_iql_constraint_aware_actor_extraction_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_constraint_aware_actor_extraction",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_constraint_aware_actor_extraction=True,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_risk_adjusted_actor_extraction_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_risk_adjusted_actor_extraction=True,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(training_metrics["actor_risk_adjusted_extraction_enabled"])
        self.assertIn(
            "detached_q_minus_action_viability_risk_actor_distillation_v1",
            training_metrics["actor_weighting_policy"],
        )
        self.assertEqual(
            training_metrics["actor_risk_adjusted_extraction_policy"],
            "detached_q_minus_action_viability_risk_actor_distillation_v1",
        )
        self.assertEqual(
            training_metrics["actor_risk_score_policy"],
            "detached_max_non_suppression_action_viability_risk_v1",
        )
        self.assertGreater(training_metrics["actor_risk_adjusted_loss_weight"], 0.0)
        self.assertGreater(training_metrics["final_risk_adjusted_actor_loss"], 0.0)
        self.assertGreaterEqual(
            training_metrics["actor_risk_adjusted_target_logged_probability_mean"],
            0.0,
        )
        self.assertGreaterEqual(
            training_metrics["actor_risk_adjusted_target_logged_top1_rate"],
            0.0,
        )

    def test_torch_discrete_iql_risk_adjusted_actor_extraction_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_risk_adjusted_actor_extraction",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_risk_adjusted_actor_extraction=True,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_calibrated_supported_actor_extraction_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_calibrated_actor_extraction=True,
                torch_iql_calibrated_supported_actor_extraction=True,
                torch_iql_actor_calibration_records=calibration_dataset.records,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics["actor_calibrated_supported_extraction_enabled"]
        )
        self.assertIn(
            "calibrated_supported_actor_extraction_v1",
            training_metrics["actor_weighting_policy"],
        )
        self.assertEqual(
            training_metrics["actor_calibrated_supported_extraction_policy"],
            "calibrated_supported_actor_extraction_v1",
        )
        calibration = training_metrics[
            "actor_calibrated_supported_calibration"
        ]
        self.assertEqual(
            calibration["schema_version"],
            "mind_calibrated_supported_actor_extraction_v1",
        )
        self.assertGreater(calibration["calibration_record_count"], 0)
        self.assertIn("risk_brier_score", calibration)
        self.assertIn("risk_auc", calibration)
        self.assertIn("risk_ece", calibration)
        self.assertIn("reliability_buckets", calibration)
        self.assertIn("by_action", calibration)
        self.assertIn("by_component", calibration)
        self.assertEqual(set(calibration["by_action"]), set(ACTION_NAMES))
        self.assertEqual(
            set(calibration["by_component"]),
            set(VIABILITY_COMPONENT_NAMES),
        )
        self.assertIn("eat", calibration["action_support_counts"])
        self.assertIn("eat", calibration["action_component_support_counts"])
        extraction = training_metrics[
            "actor_calibrated_supported_extraction"
        ]
        self.assertEqual(
            extraction["schema_version"],
            "mind_calibrated_supported_actor_targets_v1",
        )
        self.assertIn("rejection_reasons", extraction)
        self.assertIn("low_support", extraction["rejection_reasons"])
        self.assertIn("high_risk", extraction["rejection_reasons"])
        self.assertIn("negative_advantage", extraction["rejection_reasons"])
        self.assertIn("no_legal_candidate", extraction["rejection_reasons"])
        self.assertIn(
            "actor_calibrated_supported_selected_count",
            training_metrics,
        )
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_calibrated_supported_actor_extraction_rejects_missing_calibration_bank(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(ValueError, "calibration"):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="torch-discrete-iql",
                    torch_iql_calibrated_supported_actor_extraction=True,
                )

    def test_torch_discrete_iql_calibrated_supported_actor_extraction_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_calibrated_supported_actor_extraction",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_calibrated_supported_actor_extraction=True,
                    torch_iql_actor_calibration_records=(
                        calibration_dataset.records
                    ),
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_behavior_supported_actor_extraction_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            calibration_validation_path = (
                Path(tmpdir) / "calibration-validation.jsonl.gz"
            )
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            self._write_tiny_trajectory(calibration_validation_path, seed=9)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            calibration_validation_dataset = load_trajectory_jsonl(
                calibration_validation_path
            )
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_calibrated_actor_extraction=True,
                torch_iql_contextual_behavior_supported_actor_extraction=True,
                torch_iql_actor_calibration_records=calibration_dataset.records,
                torch_iql_actor_calibration_validation_records=(
                    calibration_validation_dataset.records
                ),
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics[
                "actor_contextual_behavior_supported_extraction_enabled"
            ]
        )
        self.assertIn(
            "contextual_behavior_proximity_supported_actor_extraction_v2",
            training_metrics["actor_weighting_policy"],
        )
        self.assertEqual(
            training_metrics[
                "actor_contextual_behavior_supported_extraction_policy"
            ],
            "contextual_behavior_proximity_supported_actor_extraction_v2",
        )
        fit = training_metrics["actor_calibrated_supported_calibration"]
        validation = training_metrics[
            "actor_calibrated_supported_calibration_validation"
        ]
        support = training_metrics["actor_contextual_supported_support"]
        extraction = training_metrics["actor_calibrated_supported_extraction"]
        self.assertEqual(
            validation["schema_version"],
            "mind_calibrated_supported_actor_calibration_validation_v1",
        )
        self.assertGreater(fit["calibration_record_count"], 0)
        self.assertGreater(validation["calibration_record_count"], 0)
        self.assertIn("risk_brier_score", validation)
        self.assertIn("risk_ece", validation)
        self.assertEqual(
            support["schema_version"],
            "mind_contextual_behavior_support_v1",
        )
        self.assertIn("fallback_depth_counts", support)
        self.assertIn("by_action", support)
        self.assertEqual(
            extraction["schema_version"],
            "mind_contextual_behavior_supported_actor_targets_v2",
        )
        self.assertIn("behavior_proximity", extraction["rejection_reasons"])
        self.assertIn("low_context_support", extraction["rejection_reasons"])
        self.assertIn("target_distribution_tvd_from_logged", extraction)
        self.assertIn("movement_or_stay_to_resource_count", extraction)
        self.assertIn("rejection_reasons_by_logged_action", extraction)
        self.assertIn("rejection_reasons_by_context", extraction)
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_behavior_caps_enforce_final_selected_rates(
        self,
    ) -> None:
        torch = __import__("torch")
        action_count = len(ACTION_NAMES)
        eat_index = ACTION_NAMES.index("eat")
        move_index = ACTION_NAMES.index("move_north")
        row_count = 100
        logged_actions = [eat_index] * 40 + [move_index] * 60
        y = torch.tensor(logged_actions, dtype=torch.long)
        candidate_mask = torch.zeros((row_count, action_count), dtype=torch.bool)
        target_scores = torch.full((row_count, action_count), -1000.0)
        candidate_mask[:, eat_index] = True
        target_scores[:, eat_index] = torch.linspace(1.0, 0.001, row_count)
        action_family = torch.tensor(
            [
                mind_torch_trainer._action_family_index(action)
                for action in ACTION_NAMES
            ],
            dtype=torch.long,
        )

        capped_mask, target_actions, cap_masks = (
            mind_torch_trainer._apply_behavior_proximity_caps(
                torch,
                y,
                candidate_mask,
                target_scores,
                action_family,
            )
        )

        selected = capped_mask.any(dim=1)
        selected_count = int(selected.sum().item())
        selected_targets = target_actions[selected]
        selected_logged = y[selected]
        eat_target_count = int((selected_targets == eat_index).sum().item())
        eat_logged_count = int((selected_logged == eat_index).sum().item())
        source_families = action_family[selected_logged]
        target_families = action_family[selected_targets]
        movement_family = mind_torch_trainer._action_family_index("move_north")
        stay_family = mind_torch_trainer._action_family_index("stay")
        resource_family = mind_torch_trainer._action_family_index("eat")
        movement_or_stay_to_resource_count = int(
            (
                (
                    (source_families == movement_family)
                    | (source_families == stay_family)
                )
                & (target_families == resource_family)
            )
            .sum()
            .item()
        )

        self.assertLessEqual(
            eat_target_count,
            math.floor(
                eat_logged_count
                * mind_torch_trainer.TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_TARGET_ACTION_EXPANSION_RATIO
            ),
        )
        self.assertLessEqual(
            movement_or_stay_to_resource_count,
            math.floor(
                selected_count
                * mind_torch_trainer.TORCH_IQL_CONTEXTUAL_SUPPORTED_MAX_MOVEMENT_STAY_RESOURCE_RATE
            ),
        )
        self.assertGreater(
            int(cap_masks["target_action_expansion"].sum().item()),
            0,
        )
        self.assertGreater(
            int(cap_masks["movement_stay_resource_conversion"].sum().item()),
            0,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_behavior_prior_loss_prefers_prior_matching_logits(
        self,
    ) -> None:
        torch = __import__("torch")
        action_count = len(ACTION_NAMES)
        eat_index = ACTION_NAMES.index("eat")
        move_index = ACTION_NAMES.index("move_north")
        action_mask = torch.zeros((2, action_count), dtype=torch.bool)
        action_mask[:, eat_index] = True
        action_mask[:, move_index] = True
        context_probabilities = torch.zeros((2, action_count), dtype=torch.float32)
        context_probabilities[0, eat_index] = 0.8
        context_probabilities[0, move_index] = 0.2
        context_probabilities[1, eat_index] = 0.1
        context_probabilities[1, move_index] = 0.9
        replay_weights = torch.ones(2, dtype=torch.float32)
        matching_logits = torch.full((2, action_count), -1000.0)
        matching_logits[0, eat_index] = math.log(0.8)
        matching_logits[0, move_index] = math.log(0.2)
        matching_logits[1, eat_index] = math.log(0.1)
        matching_logits[1, move_index] = math.log(0.9)
        swapped_logits = torch.full((2, action_count), -1000.0)
        swapped_logits[0, eat_index] = math.log(0.2)
        swapped_logits[0, move_index] = math.log(0.8)
        swapped_logits[1, eat_index] = math.log(0.9)
        swapped_logits[1, move_index] = math.log(0.1)

        matching_loss, stats = (
            mind_torch_trainer._contextual_behavior_prior_actor_loss(
                torch,
                matching_logits,
                action_mask,
                context_probabilities,
                replay_weights,
            )
        )
        swapped_loss, _ = mind_torch_trainer._contextual_behavior_prior_actor_loss(
            torch,
            swapped_logits,
            action_mask,
            context_probabilities,
            replay_weights,
        )

        self.assertLess(
            float(matching_loss.detach().cpu().item()),
            float(swapped_loss.detach().cpu().item()),
        )
        self.assertEqual(stats["row_count"], 2)
        self.assertGreater(stats["behavior_prior_entropy_mean"], 0.0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_behavior_prior_regularization_is_opt_in(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            calibration_validation_path = (
                Path(tmpdir) / "calibration-validation.jsonl.gz"
            )
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            self._write_tiny_trajectory(calibration_validation_path, seed=9)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            calibration_validation_dataset = load_trajectory_jsonl(
                calibration_validation_path
            )
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_calibrated_actor_extraction=True,
                torch_iql_contextual_behavior_supported_actor_extraction=True,
                torch_iql_contextual_behavior_prior_regularization=True,
                torch_iql_actor_calibration_records=calibration_dataset.records,
                torch_iql_actor_calibration_validation_records=(
                    calibration_validation_dataset.records
                ),
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics[
                "actor_contextual_behavior_prior_regularization_enabled"
            ]
        )
        self.assertIn(
            "contextual_behavior_prior_cross_entropy_actor_regularization_v1",
            training_metrics["actor_weighting_policy"],
        )
        self.assertEqual(
            training_metrics[
                "actor_contextual_behavior_prior_regularization_policy"
            ],
            "contextual_behavior_prior_cross_entropy_actor_regularization_v1",
        )
        self.assertGreaterEqual(
            training_metrics["final_contextual_behavior_prior_loss"],
            0.0,
        )
        self.assertIn(
            "actor_contextual_behavior_prior",
            training_metrics,
        )
        self.assertEqual(
            training_metrics["actor_contextual_behavior_prior"]["schema_version"],
            "mind_contextual_behavior_prior_actor_regularization_v1",
        )
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_behavior_prior_regularization_does_not_require_target_extraction(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_contextual_behavior_prior_regularization=True,
                torch_iql_actor_calibration_records=calibration_dataset.records,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics[
                "actor_contextual_behavior_prior_regularization_enabled"
            ]
        )
        self.assertFalse(
            training_metrics["actor_calibrated_supported_extraction_enabled"]
        )
        self.assertFalse(
            training_metrics[
                "actor_contextual_behavior_supported_extraction_enabled"
            ]
        )
        self.assertIn(
            "contextual_behavior_prior_cross_entropy_actor_regularization_v1",
            training_metrics["actor_weighting_policy"],
        )
        self.assertNotIn(
            "contextual_behavior_proximity_supported_actor_extraction_v2",
            training_metrics["actor_weighting_policy"],
        )
        self.assertGreaterEqual(
            training_metrics["final_contextual_behavior_prior_loss"],
            0.0,
        )
        self.assertEqual(
            training_metrics["final_calibrated_supported_actor_loss"],
            0.0,
        )
        self.assertTrue(
            math.isfinite(
                training_metrics["final_calibrated_supported_actor_loss"]
            )
        )
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_prior_finetune_keeps_guard_feedback_margin(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            records = [dict(record) for record in dataset.records]
            hard_feedback_record_found = False
            delegate_feedback_record_found = False
            for record in records:
                label = (
                    str(record["requested_action"])
                    if bool(record.get("resolution_action_valid", False))
                    else str(record["resolved_action"])
                )
                action_mask = record["action_mask"]
                if not isinstance(action_mask, dict):
                    continue
                alternatives = [
                    action
                    for action in ACTION_NAMES
                    if action != label and bool(action_mask.get(action, False))
                ]
                if not alternatives:
                    continue
                if not hard_feedback_record_found:
                    record["action_source"] = (
                        "mind_v2_neural_policy:"
                        "observation_heuristic_safety_floor_v1"
                    )
                    record["policy_decision_diagnostics"] = {
                        "guard_used": True,
                        "heuristic_action": label,
                        "learned_action": alternatives[0],
                    }
                    hard_feedback_record_found = True
                    continue
                if not delegate_feedback_record_found:
                    record["action_source"] = (
                        "mind_v2_neural_policy:"
                        "observation_heuristic_confidence_delegate_v1"
                    )
                    record["policy_decision_diagnostics"] = {
                        "heuristic_delegate_used": True,
                        "heuristic_action": label,
                        "learned_action": alternatives[0],
                    }
                    delegate_feedback_record_found = True
                    break
            self.assertTrue(hard_feedback_record_found)
            self.assertTrue(delegate_feedback_record_found)
            artifact = train_baseline_with_trainer(
                tuple(records),
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_contextual_behavior_prior_regularization=True,
                torch_iql_actor_calibration_records=calibration_dataset.records,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(training_metrics["actor_finetune_guard_feedback_enabled"])
        self.assertEqual(
            training_metrics["actor_finetune_guard_feedback_policy"],
            "runtime_suppressed_learned_action_margin_finetune_carryover_v1",
        )
        self.assertGreaterEqual(training_metrics["guard_feedback_count"], 2)
        self.assertGreaterEqual(
            training_metrics["actor_finetune_guard_feedback_count"],
            2,
        )
        self.assertGreaterEqual(
            training_metrics["actor_finetune_guard_feedback_rate"],
            0.0,
        )
        self.assertGreater(
            training_metrics["actor_finetune_guard_feedback_loss_weight"],
            0.0,
        )
        self.assertGreaterEqual(
            training_metrics["final_actor_finetune_guard_feedback_loss"],
            0.0,
        )
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_prior_finetune_keeps_behavior_margin_anchor(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
                torch_iql_behavior_margin_anchor=True,
                torch_iql_contextual_behavior_prior_regularization=True,
                torch_iql_actor_calibration_records=calibration_dataset.records,
            ).to_artifact()

        training_metrics = artifact["model"]["neural_network"]["training_metrics"]
        self.assertTrue(
            training_metrics["actor_finetune_behavior_margin_anchor_enabled"]
        )
        self.assertEqual(
            training_metrics["actor_finetune_behavior_margin_anchor_policy"],
            "viability_safe_logged_action_margin_finetune_carryover_v1",
        )
        self.assertGreater(
            training_metrics["actor_finetune_behavior_margin_anchor_loss_weight"],
            0.0,
        )
        self.assertGreaterEqual(
            training_metrics["actor_finetune_behavior_margin_anchor_eligible_count"],
            1,
        )
        self.assertGreaterEqual(
            training_metrics["final_actor_finetune_behavior_margin_anchor_loss"],
            0.0,
        )
        json.dumps(artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_contextual_behavior_supported_actor_extraction_rejects_missing_validation_bank(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            with self.assertRaisesRegex(ValueError, "validation"):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="torch-discrete-iql",
                    torch_iql_contextual_behavior_supported_actor_extraction=True,
                    torch_iql_actor_calibration_records=(
                        calibration_dataset.records
                    ),
                )

    def test_torch_discrete_iql_contextual_behavior_supported_actor_extraction_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            calibration_path = Path(tmpdir) / "calibration.jsonl.gz"
            calibration_validation_path = (
                Path(tmpdir) / "calibration-validation.jsonl.gz"
            )
            self._write_tiny_trajectory(trajectory_path)
            self._write_tiny_trajectory(calibration_path, seed=8)
            self._write_tiny_trajectory(calibration_validation_path, seed=9)
            dataset = load_trajectory_jsonl(trajectory_path)
            calibration_dataset = load_trajectory_jsonl(calibration_path)
            calibration_validation_dataset = load_trajectory_jsonl(
                calibration_validation_path
            )
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_contextual_behavior_supported_actor_extraction",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_contextual_behavior_supported_actor_extraction=True,
                    torch_iql_actor_calibration_records=(
                        calibration_dataset.records
                    ),
                    torch_iql_actor_calibration_validation_records=(
                        calibration_validation_dataset.records
                    ),
                )

    def test_torch_discrete_iql_contextual_behavior_prior_regularization_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_contextual_behavior_prior_regularization",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_contextual_behavior_prior_regularization=True,
                )

    def test_torch_discrete_iql_neural_prior_blend_weight_rejects_other_trainers(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(
                ValueError,
                "torch_iql_neural_actor_prior_blend_weight",
            ):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="contextual-prior",
                    torch_iql_neural_actor_prior_blend_weight=0.85,
                )

    def test_torch_discrete_iql_neural_prior_blend_weight_rejects_out_of_range(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            with self.assertRaisesRegex(ValueError, "in \\[0.0, 1.0\\]"):
                train_baseline_with_trainer(
                    dataset.records,
                    provenance=dataset_provenance(dataset),
                    trainer="torch-discrete-iql",
                    torch_iql_neural_actor_prior_blend_weight=1.25,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is an optional Mind ML dependency",
    )
    def test_torch_discrete_iql_artifact_rejects_runtime_value_deviation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            artifact = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="torch-discrete-iql",
            ).to_artifact()

        artifact["model"]["value_supported_deviation_policy"] = (
            "positive_value_safe_deviation_v1"
        )
        artifact["model"]["value_supported_deviation_min_support"] = 32
        artifact["model"]["value_supported_deviation_min_value_margin"] = 0.04
        artifact["model"]["value_supported_deviation_min_learned_value"] = 0.02
        artifact["model"]["value_supported_deviation_min_score_margin"] = 0.25
        artifact["model"]["value_supported_deviation_min_predicted_advantage"] = 0.24

        json.dumps(artifact, allow_nan=False)

        with self.assertRaisesRegex(MindArtifactError, "torch-discrete-iql"):
            validate_model_artifact_manifest(artifact)

    def test_mind_train_cli_writes_neural_actor_critic_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            artifact_path = Path(tmpdir) / "neural-artifact.json"
            self._write_tiny_trajectory(trajectory_path)

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(trajectory_path),
                        "--output",
                        str(artifact_path),
                        "--trainer",
                        "neural-actor-critic-bc",
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)

        self.assertIn("trainer=neural-actor-critic-bc", stdout.getvalue())
        self.assertIn("neural_backend=pure_python_deterministic_v1", stdout.getvalue())
        self.assertIn(
            "neural_architecture=fixed_random_feature_mlp_actor_critic_v1",
            stdout.getvalue(),
        )
        self.assertEqual(
            artifact["manifest"]["model_type"],
            "guarded_neural_actor_critic_bc_v1",
        )

    def test_mind_artifact_diagnostics_cli_writes_viability_report_and_ledger(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "trajectory.jsonl.gz"
            artifact_path = tmp_path / "neural-artifact.json"
            report_path = tmp_path / "diagnostics-report.json"
            ledger_path = tmp_path / "mind-diagnostics-ledger.jsonl"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="neural-actor-critic-bc",
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_artifact_diagnostics",
                        "--artifact",
                        str(artifact_path),
                        "--trajectory",
                        str(trajectory_path),
                        "--calibration-trajectory",
                        str(trajectory_path),
                        "--output",
                        str(report_path),
                        "--experiment-ledger-output",
                        str(ledger_path),
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                mind_artifact_diagnostics.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            stdout_report = json.loads(stdout.getvalue())
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(report["schema_version"], "mind_artifact_diagnostics_report_v1")
        self.assertEqual(report, stdout_report)
        self.assertIn("viability_calibration", report["artifact_diagnostics"])
        self.assertIn("calibration_diagnostics", report)
        self.assertIn("viability_calibration", report["calibration_diagnostics"])
        viability = report["artifact_diagnostics"]["viability_calibration"]
        self.assertEqual(
            viability["runtime_decision_policy"],
            "diagnostics_only_not_used_for_runtime_v0",
        )
        self.assertEqual(
            set(viability["component_names"]),
            set(VIABILITY_COMPONENT_NAMES),
        )
        self.assertEqual(ledger_rows, [report["experiment_ledger_entry"]])
        self.assertEqual(
            ledger_rows[0]["schema_version"],
            "mind_artifact_diagnostics_ledger_v1",
        )
        self.assertIn("viability_risk_brier_score", ledger_rows[0])
        self.assertIn("calibration_viability_risk_brier_score", ledger_rows[0])

    def test_collect_trajectory_cli_writes_neural_policy_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            training_path = tmp_path / "training.jsonl.gz"
            artifact_path = tmp_path / "neural-artifact.json"
            learned_path = tmp_path / "neural-learned.jsonl.gz"
            self._write_tiny_trajectory(training_path)
            dataset = load_trajectory_jsonl(training_path)
            baseline = train_baseline_with_trainer(
                dataset.records,
                provenance=dataset_provenance(dataset),
                trainer="neural-actor-critic-bc",
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "collect_trajectory",
                        "--seed",
                        "8",
                        "--ticks",
                        "2",
                        "--output",
                        str(learned_path),
                        "--split-id",
                        "neural-online-probe",
                        "--mind-artifact",
                        str(artifact_path),
                        "--enable-mind",
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                collect_trajectory.main()

            learned_dataset = load_trajectory_jsonl(learned_path)

        self.assertIn("mind_policy=mind_v2_neural_policy", stdout.getvalue())
        self.assertEqual(
            learned_dataset.header["provenance"]["split_id"],
            "neural-online-probe",
        )
        self.assertTrue(
            any(
                record["policy_id"] == "mind_v2_neural_policy"
                for record in learned_dataset.records
            )
        )

    def test_run_headless_cli_writes_mind_viewer_replay(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "trajectory.jsonl.gz"
            artifact_path = tmp_path / "mind-artifact.json"
            replay_path = tmp_path / "mind-replay.json"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "run_headless",
                        "--seed",
                        "8",
                        "--ticks",
                        "2",
                        "--output",
                        str(replay_path),
                        "--mind-artifact",
                        str(artifact_path),
                        "--enable-mind",
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                run_headless.main()

            replay = json.loads(replay_path.read_text(encoding="utf-8"))

        self.assertIn("mind_policy=mind_v1_learned_policy", stdout.getvalue())
        records = replay["viewer"]["trajectory"]["records"]
        self.assertGreater(len(records), 0)
        self.assertTrue(
            any(record["policy_id"] == "mind_v1_learned_policy" for record in records)
        )
        self.assertEqual(
            replay["viewer"]["trajectory"]["policy_interface_version"],
            "mind_policy_interface_v1",
        )

    def test_run_headless_cli_requires_explicit_mind_enable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "mind-artifact.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "run_headless",
                        "--mind-artifact",
                        str(artifact_path),
                    ],
                ),
                self.assertRaisesRegex(SystemExit, "--mind-artifact requires --enable-mind"),
            ):
                run_headless.main()

    def test_run_headless_cli_accepts_mind_v3_founder_template(self) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_path = tmp_path / "template-report.json"
            replay_path = tmp_path / "mind-v3-template-replay.json"
            template_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "controller_metadata": founder_mind_v3_metadata(
                                agent_id=3,
                                rng=Random(17),
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "run_headless",
                        "--seed",
                        "8",
                        "--ticks",
                        "2",
                        "--output",
                        str(replay_path),
                        "--mind-v3-autonomous-evolution",
                        "--mind-v3-founder-template",
                        str(template_path),
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                run_headless.main()

            replay = json.loads(replay_path.read_text(encoding="utf-8"))

        self.assertIn(
            f"mind_v3_founder_template={template_path}",
            stdout.getvalue(),
        )
        records = replay["viewer"]["trajectory"]["records"]
        self.assertGreater(len(records), 0)
        self.assertTrue(
            all(
                record["policy_id"] == "mind_v3_autonomous_evolution_policy"
                for record in records
            )
        )
        self.assertFalse(
            any("heuristic" in record["action_source"] for record in records)
        )

    def test_collect_trajectory_cli_writes_mind_v3_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "mind-v3-trajectory.jsonl.gz"
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "collect_trajectory",
                        "--seed",
                        "8",
                        "--ticks",
                        "3",
                        "--output",
                        str(trajectory_path),
                        "--split-id",
                        "mind-v3-probe",
                        "--mind-v3-autonomous-evolution",
                        "--include-policy-diagnostics",
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                collect_trajectory.main()

            dataset = load_trajectory_jsonl(trajectory_path)

        self.assertIn(
            "mind_policy=mind_v3_autonomous_evolution_policy",
            stdout.getvalue(),
        )
        self.assertGreater(dataset.record_count, 0)
        self.assertTrue(
            all(
                record["policy_id"] == "mind_v3_autonomous_evolution_policy"
                for record in dataset.records
            )
        )
        self.assertFalse(
            any("heuristic" in record["action_source"] for record in dataset.records)
        )
        self.assertTrue(
            all(
                record["policy_decision_diagnostics"]["heuristic_free"]
                for record in dataset.records
                if "policy_decision_diagnostics" in record
            )
        )

    def test_collect_trajectory_cli_writes_loadable_mind_v3_update_traces(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "mind-v3-update-trace.jsonl.gz"
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "collect_trajectory",
                        "--seed",
                        "8",
                        "--ticks",
                        "3",
                        "--output",
                        str(trajectory_path),
                        "--split-id",
                        "mind-v3-update-trace-probe",
                        "--mind-v3-autonomous-evolution",
                        "--include-policy-update-trace",
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                collect_trajectory.main()

            dataset = load_trajectory_jsonl(trajectory_path)

        self.assertIn("policy_update_trace=included", stdout.getvalue())
        traces = [
            record["policy_update_trace"]
            for record in dataset.records
            if isinstance(record.get("policy_update_trace"), dict)
        ]
        self.assertGreater(len(traces), 0)
        self.assertEqual(
            traces[0]["schema_version"],
            "mind_v3_controller_update_trace_v1",
        )
        self.assertEqual(
            traces[0]["policy"],
            "bounded_reward_modulated_controller_update_v1",
        )
        transitions = build_trajectory_transitions(dataset.records)
        traced_transitions = [
            transition
            for transition in transitions
            if transition.policy_update_trace is not None
        ]
        self.assertGreater(len(traced_transitions), 0)
        self.assertEqual(
            traced_transitions[0].policy_update_trace["schema_version"],
            "mind_v3_controller_update_trace_v1",
        )

    def test_collect_trajectory_cli_accepts_mind_v3_founder_template(self) -> None:
        from random import Random

        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_path = tmp_path / "template-report.json"
            trajectory_path = tmp_path / "mind-v3-template-trajectory.jsonl.gz"
            template_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "controller_metadata": founder_mind_v3_metadata(
                                agent_id=2,
                                rng=Random(13),
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "collect_trajectory",
                        "--seed",
                        "8",
                        "--ticks",
                        "3",
                        "--output",
                        str(trajectory_path),
                        "--split-id",
                        "mind-v3-template-probe",
                        "--mind-v3-autonomous-evolution",
                        "--mind-v3-founder-template",
                        str(template_path),
                        "--include-policy-diagnostics",
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                collect_trajectory.main()

            dataset = load_trajectory_jsonl(trajectory_path)

        self.assertIn(
            f"mind_v3_founder_template={template_path}",
            stdout.getvalue(),
        )
        self.assertGreater(dataset.record_count, 0)
        self.assertTrue(
            all(
                record["policy_id"] == "mind_v3_autonomous_evolution_policy"
                for record in dataset.records
            )
        )
        self.assertFalse(
            any("heuristic" in record["action_source"] for record in dataset.records)
        )

    def test_mind_v3_evaluate_cli_reports_zero_heuristic_action_sources(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evaluate

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-eval.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evaluate",
                        "--seeds",
                        "5",
                        "--ticks",
                        "20",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evaluate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["schema_version"],
            "mind_v3_autonomous_evolution_evaluation_v1",
        )
        self.assertTrue(report["policy"]["heuristic_free"])
        self.assertEqual(report["comparison"]["heuristic"]["runs"][0]["seed"], 5)
        self.assertEqual(report["comparison"]["mind_v3"]["runs"][0]["seed"], 5)
        self.assertEqual(
            report["comparison"]["mind_v3"]["aggregate"][
                "heuristic_action_source_count"
            ],
            0,
        )
        self.assertIn(
            "dominant_requested_action_share",
            report["comparison"]["mind_v3"]["aggregate"],
        )
        self.assertIn(
            "terminal_energy_requirement_satisfaction_mean",
            report["comparison"]["mind_v3"]["aggregate"],
        )
        self.assertIn(
            "terminal_energy_viability_share_mean",
            report["comparison"]["mind_v3"]["aggregate"],
        )
        self.assertIn(
            "biologically_reproduction_ready_agents_mean",
            report["comparison"]["mind_v3"]["aggregate"],
        )
        self.assertIn(
            "founder_template_count",
            report["policy"],
        )
        run_attribution = report["comparison"]["mind_v3"]["runs"][0][
            "reproduction_failure_attribution"
        ]
        self.assertEqual(
            run_attribution["policy"],
            mind_v3_evaluate.MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
        )
        self.assertIn("hydration", run_attribution["terminal_viability_shares"])
        self.assertIn(
            "terminal_energy_requirement_satisfaction",
            run_attribution,
        )
        aggregate_attribution = report["comparison"]["mind_v3"]["aggregate"][
            "reproduction_failure_attribution"
        ]
        self.assertIn(
            "terminal_readiness_by_trophic_role",
            aggregate_attribution,
        )
        self.assertIn(
            "terminal_energy_requirement_satisfaction_mean",
            aggregate_attribution,
        )
        self.assertIn(
            "biologically_ready_agents_mean",
            report["comparison"]["delta"],
        )

    def test_mind_v3_evaluate_reports_real_energy_requirement_satisfaction(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evaluate

        attribution = mind_v3_evaluate._reproduction_failure_attribution(
            {
                "alive_agents": 2,
                "reproduction_end": {
                    "ready_agents": 0,
                    "biologically_ready_agents": 0,
                    "biological_blocker_counts": {
                        "energy": 1,
                        "hydration": 0,
                        "health": 0,
                        "matched_diet": 0,
                    },
                    "energy_readiness_by_meat_mode": {
                        "none": {
                            "alive_agents": 1,
                            "energy_shortfall_agents": 0,
                            "energy_total": 0.8,
                            "energy_required_total": 0.7,
                            "energy_gap_total": 0.0,
                        },
                        "scavenger": {
                            "alive_agents": 1,
                            "energy_shortfall_agents": 1,
                            "energy_total": 0.2,
                            "energy_required_total": 0.7,
                            "energy_gap_total": 0.5,
                        },
                    },
                },
            }
        )

        self.assertEqual(
            attribution["policy"],
            mind_v3_evaluate.MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
        )
        self.assertEqual(
            attribution["terminal_energy_requirement_satisfaction"],
            0.6429,
        )
        self.assertEqual(attribution["terminal_energy_shortfall_share"], 0.5)
        self.assertEqual(attribution["terminal_energy_required_total"], 1.4)
        self.assertEqual(
            attribution["terminal_energy_hydration_balance"],
            0.6429,
        )
        self.assertGreater(
            attribution["terminal_balanced_reproduction_readiness"],
            0.0,
        )

        aggregate = mind_v3_evaluate._aggregate_runs(
            [
                {
                    "alive_agents": 2,
                    "births": 0,
                    "deaths": 0,
                    "trajectory_record_count": 0,
                    "heuristic_action_source_count": 0,
                    "unique_requested_actions": 0,
                    "action_source_counts": {},
                    "policy_id_counts": {},
                    "requested_action_counts": {},
                    "resolved_action_counts": {},
                    "reproduction_failure_attribution": attribution,
                    "temporal_readiness_attribution": {
                        "policy": (
                            mind_v3_evaluate.MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY
                        ),
                        "alive_agent_tick_count": 2,
                        "core_blocker_agent_tick_counts": {
                            "energy": 1,
                            "hydration": 0,
                            "health": 0,
                        },
                        "primary_core_blocker_agent_tick_counts": {
                            "energy": 1,
                            "hydration": 0,
                            "health": 0,
                        },
                    },
                }
            ]
        )

        self.assertEqual(
            aggregate["terminal_energy_requirement_satisfaction_mean"],
            0.6429,
        )
        self.assertEqual(aggregate["terminal_energy_shortfall_share_mean"], 0.5)
        self.assertEqual(aggregate["terminal_energy_viability_share_mean"], 0.5)
        self.assertEqual(aggregate["terminal_energy_hydration_balance_mean"], 0.6429)
        self.assertEqual(aggregate["primary_temporal_readiness_blocker"], "energy")
        self.assertEqual(aggregate["reproduction_ready_agents_mean"], 0.0)

    def test_mind_v3_evaluate_cli_runs_controlled_fixture_suite(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evaluate

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-fixture-eval.json"
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evaluate",
                        "--seeds",
                        "5",
                        "--ticks",
                        "6",
                        "--fixture-suite",
                        "basic",
                        "--fixture-ticks",
                        "6",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evaluate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("mind_v3_fixture_count=4", stdout.getvalue())
        self.assertIn("mind_v3_fixture_gate_passed=True", stdout.getvalue())
        fixture_suite = report["fixture_suite"]
        self.assertEqual(
            fixture_suite["policy"],
            mind_v3_evaluate.MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
        )
        fixture_gate = report["fixture_gate"]
        self.assertEqual(
            fixture_gate["policy"],
            mind_v3_evaluate.MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
        )
        self.assertTrue(fixture_gate["passed"])
        self.assertEqual(fixture_gate["blockers"], [])
        self.assertEqual(
            sorted(fixture_gate["per_fixture"]),
            sorted(mind_v3_evaluate.CONTROLLED_FIXTURE_NAMES),
        )
        self.assertEqual(
            fixture_suite["fixture_names"],
            list(mind_v3_evaluate.CONTROLLED_FIXTURE_NAMES),
        )
        self.assertEqual(len(fixture_suite["fixtures"]), 4)
        for fixture in fixture_suite["fixtures"]:
            self.assertIn("scenario_config", fixture)
            self.assertIn("initial_archetype_counts", fixture["scenario_config"])
            comparison = fixture["comparison"]
            self.assertIn("biologically_ready_agents_mean", comparison["delta"])
            for policy_name in ("heuristic", "mind_v3"):
                run = comparison[policy_name]["runs"][0]
                self.assertEqual(
                    run["evaluation_context"],
                    "controlled_ecology_fixture",
                )
                self.assertEqual(run["fixture"], fixture["fixture"])
                self.assertGreater(run["alive_agents"], 0)
                self.assertIn("trophic_role_counts_at_end", run)
                self.assertIn(
                    "animal_resource_opportunity_by_meat_mode_end",
                    run,
                )
                self.assertIn("diet_by_trophic_role_end", run)
                self.assertIn("combat_end", run)
                attribution = run["reproduction_failure_attribution"]
                self.assertEqual(
                    attribution["policy"],
                    mind_v3_evaluate.MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
                )
                self.assertIn("energy", attribution["biological_blocker_counts"])
                self.assertIn(
                    "biological_blocker_counts_by_trophic_role",
                    attribution,
                )
            self.assertEqual(
                comparison["mind_v3"]["aggregate"][
                    "heuristic_action_source_count"
                ],
                0,
            )
            self.assertIn(
                "terminal_readiness_by_trophic_role",
                comparison["mind_v3"]["aggregate"][
                    "reproduction_failure_attribution"
                ],
            )

    def test_mind_v3_evaluate_cli_runs_controlled_fixture_subset(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evaluate

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-carrion-fixture-eval.json"
            trajectory_dir = Path(tmpdir) / "trajectories"
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evaluate",
                        "--seeds",
                        "5",
                        "--ticks",
                        "2",
                        "--fixture-suite",
                        "basic",
                        "--fixture-names",
                        "carrion_only",
                        "--fixture-ticks",
                        "2",
                        "--trajectory-output-dir",
                        str(trajectory_dir),
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evaluate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            fixture_suite = report["fixture_suite"]
            open_trajectory_path = Path(
                report["comparison"]["mind_v3"]["runs"][0]["trajectory_path"]
            )
            fixture_trajectory_path = Path(
                fixture_suite["fixtures"][0]["comparison"]["mind_v3"]["runs"][0][
                    "trajectory_path"
                ]
            )
            open_trajectory_exists = open_trajectory_path.exists()
            fixture_trajectory_exists = fixture_trajectory_path.exists()
            open_trajectory_record_count = load_trajectory_jsonl(
                open_trajectory_path
            ).record_count
            fixture_trajectory_record_count = load_trajectory_jsonl(
                fixture_trajectory_path
            ).record_count

        self.assertIn("mind_v3_fixture_count=1", stdout.getvalue())
        self.assertEqual(fixture_suite["fixture_names"], ["carrion_only"])
        self.assertEqual(len(fixture_suite["fixtures"]), 1)
        self.assertEqual(
            sorted(report["fixture_gate"]["per_fixture"]),
            ["carrion_only"],
        )
        self.assertEqual(
            report["fixture_gate"]["fixture_names"],
            ["carrion_only"],
        )
        self.assertTrue(open_trajectory_exists)
        self.assertTrue(fixture_trajectory_exists)
        self.assertEqual(
            open_trajectory_record_count,
            report["comparison"]["mind_v3"]["runs"][0][
                "trajectory_record_count"
            ],
        )
        self.assertEqual(
            fixture_trajectory_record_count,
            fixture_suite["fixtures"][0]["comparison"]["mind_v3"]["runs"][0][
                "trajectory_record_count"
            ],
        )

    def test_mind_v3_fixture_gate_blocks_mixed_stable_birth_floor(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evaluate

        fixture_suite = {
            "policy": mind_v3_evaluate.MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
            "suite": "basic",
            "seeds": [5],
            "ticks": 10,
            "fixtures": [
                {
                    "fixture": "mixed_stable",
                    "comparison": {
                        "mind_v3": {
                            "aggregate": {
                                "alive_agents_mean": 3.0,
                                "births_mean": 0.0,
                                "reproduction_failure_attribution": {
                                    "biologically_ready_agents_mean": 1.0,
                                    "terminal_viability_shares_mean": {
                                        "energy": 1.0,
                                        "hydration": 1.0,
                                        "health": 1.0,
                                        "matched_diet": 1.0,
                                    },
                                },
                            }
                        }
                    },
                }
            ],
        }
        fixture_config = mind_v3_evaluate.mind_v3_fixture_gate_config(
            suite="basic",
            seeds=[5],
            ticks=10,
            min_alive=1.0,
            min_births=0.0,
            min_mixed_stable_births=1.0,
            min_energy_viability=0.0,
            min_hydration_viability=0.0,
            min_health_viability=0.0,
            min_matched_diet_viability=0.0,
            min_biologically_ready=0.0,
        )

        gate = mind_v3_evaluate.mind_v3_fixture_gate_status(
            fixture_suite=fixture_suite,
            fixture_config=fixture_config,
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["blockers"][0]["reason"],
            "fixture_mixed_stable_birth_floor",
        )

    def test_mind_v3_evolve_cli_writes_generation_report_without_heuristics(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-evolve.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evolve",
                        "--seeds",
                        "5",
                        "--ticks",
                        "20",
                        "--population-size",
                        "2",
                        "--generations",
                        "1",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "mind_v3_evolution_search_v1")
        self.assertEqual(report["search"]["generation_count"], 1)
        self.assertEqual(len(report["generations"]), 1)
        self.assertIn("controller_metadata", report["best_candidate"])
        self.assertEqual(
            report["best_candidate"]["heuristic_action_source_count"],
            0,
        )
        for generation in report["generations"]:
            for candidate in generation["candidates"]:
                self.assertEqual(candidate["heuristic_action_source_count"], 0)

    def test_mind_v3_evolve_cli_records_parallel_rollout_metadata(self) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-evolve-workers.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evolve",
                        "--seeds",
                        "5",
                        "--ticks",
                        "8",
                        "--population-size",
                        "2",
                        "--generations",
                        "1",
                        "--rollout-workers",
                        "2",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        rollout_execution = report["search"]["rollout_execution"]
        self.assertEqual(
            rollout_execution["policy"],
            "mind_v3_candidate_process_pool_rollouts_v1",
        )
        self.assertEqual(rollout_execution["requested_workers"], 2)
        self.assertEqual(rollout_execution["generation_workers"], 2)
        self.assertEqual(rollout_execution["holdout_workers"], 0)

    def test_mind_v3_evolve_cli_writes_archive_and_holdout_report(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-evolve.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evolve",
                        "--seeds",
                        "5",
                        "--holdout-seeds",
                        "13",
                        "--ticks",
                        "12",
                        "--population-size",
                        "3",
                        "--generations",
                        "1",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["archive"]["policy"], "quality_diversity_archive_v2")
        self.assertIn("survival", report["archive"]["elites"])
        self.assertIn("resource_use", report["archive"]["elites"])
        self.assertIn("score_components", report["best_candidate"])
        self.assertIn("resource_event_rate", report["best_candidate"])
        self.assertEqual(
            report["holdout_evaluation"]["aggregate"][
                "heuristic_action_source_count"
            ],
            0,
        )

    def test_mind_v3_evolve_cli_records_fixture_gate_report(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-evolve-fixture.json"
            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evolve",
                        "--seeds",
                        "5",
                        "--ticks",
                        "6",
                        "--population-size",
                        "2",
                        "--generations",
                        "1",
                        "--rollout-workers",
                        "2",
                        "--fixture-suite",
                        "basic",
                        "--fixture-ticks",
                        "6",
                        "--fixture-rerank-top-k",
                        "2",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn("fixture_gate_passed=True", stdout.getvalue())
        self.assertEqual(
            report["search"]["fixture_gate_policy"],
            "mind_v3_controlled_fixture_hard_gate_v1",
        )
        self.assertEqual(
            report["search"]["fixture_rerank_policy"],
            "fixture_holdout_top_k_multi_horizon_scavenger_lane_rerank_v7",
        )
        self.assertEqual(report["search"]["fixture_rerank_top_k"], 2)
        self.assertEqual(
            report["search"]["score_policy"],
            "need_gated_navigation_visible_movement_fixture_selection_qd_v26",
        )
        self.assertEqual(
            report["search"]["fixture_selection_nominee_policy"],
            "generation_fixture_diverse_nominee_pool_v1",
        )
        self.assertNotIn("warm_start_policy", report["search"])
        self.assertIsNone(report["warm_start"])
        self.assertEqual(
            report["search"]["founder_template_assignment_policy"],
            "contextual_trophic_founder_template_assignment_v1",
        )
        self.assertIn(
            "need_gated_local_navigation_feature_projection_linear_action_head_v4",
            report["search"]["controller_architecture_counts"],
        )
        self.assertEqual(
            report["search"]["best_candidate_controller_architecture"],
            report["best_candidate"]["controller_metadata"]["architecture"],
        )
        self.assertIn(
            "terminal_hydration_viability_share_mean",
            report["best_candidate"],
        )
        self.assertIn(
            "terminal_energy_requirement_satisfaction_mean",
            report["best_candidate"],
        )
        self.assertIn(
            "reproduction_ready_agent_tick_share_mean",
            report["best_candidate"],
        )
        self.assertEqual(
            report["fixture_suite"]["policy"],
            "mind_v3_controlled_ecology_fixture_suite_v1",
        )
        self.assertEqual(
            report["fixture_rerank"]["selected_candidate_id"],
            report["best_candidate"]["candidate_id"],
        )
        self.assertEqual(report["fixture_rerank"]["initial_candidate_count"], 2)
        self.assertGreaterEqual(report["fixture_rerank"]["candidate_count"], 2)
        self.assertIn("repair_candidate_count", report["fixture_rerank"])
        self.assertEqual(
            report["fixture_rerank"]["execution"]["policy"],
            "fixture_rerank_candidate_process_pool_v1",
        )
        self.assertEqual(
            report["fixture_rerank"]["execution"]["requested_workers"],
            2,
        )
        self.assertGreaterEqual(
            report["fixture_rerank"]["execution"]["initial_workers"],
            1,
        )
        self.assertTrue(report["fixture_gate"]["passed"])
        self.assertEqual(report["fixture_gate"]["blockers"], [])
        self.assertEqual(
            report["fixture_gate"]["suite"],
            "basic",
        )
        self.assertIn("fixture_selection", report["generations"][0])
        self.assertIn(
            "blocker_counts_by_fixture",
            report["generations"][0]["fixture_selection"],
        )

    def test_mind_v3_fixture_rerank_prefers_gate_pass_over_search_score(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        failed_high_score = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [{"reason": "fixture_alive_floor"}],
            },
            "fixture_summary": {
                "alive_agents_mean": 4.0,
                "births_mean": 4.0,
                "mixed_stable_births_mean": 2.0,
                "terminal_reproduction_viability_min": 0.5,
            },
            "holdout_aggregate": {
                "alive_agents_mean": 8.0,
                "births_mean": 2.0,
                "terminal_energy_viability_share_mean": 0.5,
                "terminal_hydration_viability_share_mean": 0.5,
                "terminal_health_viability_share_mean": 0.5,
                "terminal_matched_diet_viability_share_mean": 0.5,
                "dominant_requested_action_share": 0.5,
            },
        }
        passed_lower_score = {
            **failed_high_score,
            "prefilter_rank": 1,
            "search_score": 50.0,
            "fixture_gate": {"passed": True, "blockers": []},
        }

        self.assertGreater(
            _fixture_rerank_selection_key(passed_lower_score),
            _fixture_rerank_selection_key(failed_high_score),
        )

    def test_mind_v3_fixture_rerank_prefers_sustained_holdout_readiness(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        base = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {"passed": True, "blockers": []},
            "fixture_summary": {
                "alive_agents_mean": 4.0,
                "births_mean": 2.0,
                "mixed_stable_births_mean": 1.0,
                "terminal_reproduction_viability_min": 0.5,
                "biologically_ready_agents_mean": 0.0,
            },
            "holdout_aggregate": {
                "alive_agents_mean": 8.0,
                "births_mean": 2.0,
                "terminal_energy_viability_share_mean": 0.5,
                "terminal_hydration_viability_share_mean": 0.5,
                "terminal_health_viability_share_mean": 0.5,
                "terminal_matched_diet_viability_share_mean": 0.5,
                "dominant_requested_action_share": 0.5,
                "reproduction_ready_agent_tick_share_mean": 0.0,
                "reproduction_ready_pair_tick_share_mean": 0.0,
                "terminal_ready_group_count_mean": 0.0,
            },
        }
        sustained = {
            **base,
            "prefilter_rank": 1,
            "search_score": 50.0,
            "holdout_aggregate": {
                **base["holdout_aggregate"],
                "reproduction_ready_agent_tick_share_mean": 0.2,
                "reproduction_ready_pair_tick_share_mean": 0.1,
                "terminal_ready_group_count_mean": 1.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(sustained),
            _fixture_rerank_selection_key(base),
        )

    def test_mind_v3_fixture_rerank_prefers_balanced_holdout_resources(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        energy_overcorrected = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {"passed": True, "blockers": []},
            "fixture_summary": {
                "alive_agents_mean": 4.0,
                "births_mean": 2.0,
                "mixed_stable_births_mean": 1.0,
                "terminal_reproduction_viability_min": 0.25,
                "biologically_ready_agents_mean": 0.0,
            },
            "holdout_aggregate": {
                "alive_agents_mean": 8.0,
                "births_mean": 2.0,
                "terminal_energy_viability_share_mean": 0.8,
                "terminal_energy_requirement_satisfaction_mean": 0.9,
                "terminal_hydration_viability_share_mean": 0.25,
                "terminal_health_viability_share_mean": 0.8,
                "terminal_matched_diet_viability_share_mean": 0.8,
                "dominant_requested_action_share": 0.5,
            },
        }
        balanced = {
            **energy_overcorrected,
            "prefilter_rank": 1,
            "search_score": 50.0,
            "holdout_aggregate": {
                **energy_overcorrected["holdout_aggregate"],
                "terminal_energy_requirement_satisfaction_mean": 0.68,
                "terminal_hydration_viability_share_mean": 0.68,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(balanced),
            _fixture_rerank_selection_key(energy_overcorrected),
        )

    def test_mind_v3_fixture_rerank_prefers_holdout_before_fixture_yield(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        high_fixture_low_holdout = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {"passed": True, "blockers": []},
            "fixture_summary": {
                "alive_agents_mean": 20.0,
                "births_mean": 20.0,
                "mixed_stable_births_mean": 20.0,
                "terminal_reproduction_viability_min": 0.5,
                "biologically_ready_agents_mean": 0.0,
            },
            "holdout_aggregate": {
                "alive_agents_mean": 4.0,
                "births_mean": 1.0,
                "terminal_energy_viability_share_mean": 0.5,
                "terminal_hydration_viability_share_mean": 0.5,
                "terminal_health_viability_share_mean": 0.5,
                "terminal_matched_diet_viability_share_mean": 0.5,
                "dominant_requested_action_share": 0.4,
            },
        }
        lower_fixture_better_holdout = {
            **high_fixture_low_holdout,
            "prefilter_rank": 1,
            "search_score": 50.0,
            "fixture_summary": {
                "alive_agents_mean": 5.0,
                "births_mean": 5.0,
                "mixed_stable_births_mean": 5.0,
                "terminal_reproduction_viability_min": 0.5,
                "biologically_ready_agents_mean": 0.0,
            },
            "holdout_aggregate": {
                **high_fixture_low_holdout["holdout_aggregate"],
                "alive_agents_mean": 6.0,
                "births_mean": 2.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(lower_fixture_better_holdout),
            _fixture_rerank_selection_key(high_fixture_low_holdout),
        )

    def test_mind_v3_fixture_rerank_prioritizes_carrion_blocker_lane(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        weak_carrion_high_holdout = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_alive_floor",
                    }
                ],
            },
            "fixture_summary": {
                "alive_agents_mean": 10.0,
                "births_mean": 10.0,
                "mixed_stable_births_mean": 4.0,
                "terminal_reproduction_viability_min": 0.2,
                "biologically_ready_agents_mean": 0.0,
                "per_fixture": {
                    "carrion_only": {
                        "alive_agents_mean": 0.0,
                        "births_mean": 1.0,
                        "terminal_energy_hydration_balance_mean": 0.0,
                        "terminal_reproduction_viability_min": 0.0,
                    }
                },
            },
            "holdout_aggregate": {
                "alive_agents_mean": 20.0,
                "births_mean": 8.0,
                "terminal_energy_viability_share_mean": 0.8,
                "terminal_energy_requirement_satisfaction_mean": 0.8,
                "terminal_hydration_viability_share_mean": 0.8,
                "terminal_health_viability_share_mean": 0.8,
                "terminal_matched_diet_viability_share_mean": 0.8,
                "dominant_requested_action_share": 0.4,
            },
        }
        better_carrion_lower_holdout = {
            **weak_carrion_high_holdout,
            "prefilter_rank": 1,
            "search_score": 20.0,
            "fixture_summary": {
                **weak_carrion_high_holdout["fixture_summary"],
                "per_fixture": {
                    "carrion_only": {
                        "alive_agents_mean": 1.0,
                        "births_mean": 2.0,
                        "terminal_energy_hydration_balance_mean": 0.25,
                        "terminal_reproduction_viability_min": 0.2,
                    }
                },
            },
            "holdout_aggregate": {
                **weak_carrion_high_holdout["holdout_aggregate"],
                "alive_agents_mean": 5.0,
                "births_mean": 1.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(better_carrion_lower_holdout),
            _fixture_rerank_selection_key(weak_carrion_high_holdout),
        )

    def test_mind_v3_fixture_rerank_prioritizes_carrion_energy_readiness(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        base = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_energy_viability_floor",
                    }
                ],
            },
            "fixture_summary": {
                "alive_agents_mean": 10.0,
                "births_mean": 10.0,
                "mixed_stable_births_mean": 5.0,
                "terminal_reproduction_viability_min": 0.2,
                "biologically_ready_agents_mean": 0.0,
                "per_fixture": {
                    "carrion_only": {
                        "alive_agents_mean": 2.0,
                        "births_mean": 1.0,
                        "terminal_energy_requirement_satisfaction_mean": 0.2,
                        "terminal_energy_viability_share_mean": 0.0,
                        "terminal_energy_hydration_balance_mean": 0.2,
                        "terminal_reproduction_viability_min": 0.0,
                        "animal_resource_consumption_events_mean": 0.5,
                        "animal_resource_gained_energy_mean": 0.1,
                        "dominant_requested_action_share": 1.0,
                    }
                },
            },
            "holdout_aggregate": {
                "alive_agents_mean": 20.0,
                "births_mean": 8.0,
                "terminal_energy_viability_share_mean": 0.8,
                "terminal_energy_requirement_satisfaction_mean": 0.8,
                "terminal_hydration_viability_share_mean": 0.8,
                "terminal_health_viability_share_mean": 0.8,
                "terminal_matched_diet_viability_share_mean": 0.8,
                "dominant_requested_action_share": 0.4,
            },
        }
        higher_energy_lower_holdout = {
            **base,
            "prefilter_rank": 1,
            "search_score": 20.0,
            "fixture_summary": {
                **base["fixture_summary"],
                "per_fixture": {
                    "carrion_only": {
                        **base["fixture_summary"]["per_fixture"]["carrion_only"],
                        "alive_agents_mean": 1.0,
                        "births_mean": 0.0,
                        "terminal_energy_requirement_satisfaction_mean": 0.45,
                        "animal_resource_consumption_events_mean": 2.0,
                        "animal_resource_gained_energy_mean": 0.5,
                        "dominant_requested_action_share": 0.7,
                    }
                },
            },
            "holdout_aggregate": {
                **base["holdout_aggregate"],
                "alive_agents_mean": 4.0,
                "births_mean": 1.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(higher_energy_lower_holdout),
            _fixture_rerank_selection_key(base),
        )

    def test_mind_v3_generation_fixture_pressure_penalizes_blockers(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _fixture_blocker_pressure_summary,
            _fixture_selection_score_delta,
        )

        passing = {
            "fixture_gate": {"passed": True, "blockers": []},
            "fixture_horizon_summary": {
                "passed_horizon_count": 2,
                "carrion_only_alive_agents_min": 2.0,
                "carrion_only_births_min": 2.0,
                "carrion_only_terminal_energy_requirement_satisfaction_min": 0.6,
                "carrion_only_terminal_energy_viability_share_min": 0.5,
                "carrion_only_terminal_hydration_viability_share_min": 0.5,
                "carrion_only_terminal_matched_diet_viability_share_min": 0.5,
                "carrion_only_animal_resource_consumption_events_min": 4.0,
                "carrion_only_animal_resource_gained_energy_min": 2.0,
                "terminal_reproduction_viability_min": 0.4,
                "terminal_energy_hydration_balance_min": 0.5,
                "births_min": 6.0,
                "alive_agents_min": 8.0,
            },
            "fixture_horizons": [
                {"fixture_gate": {"passed": True}},
                {"fixture_gate": {"passed": True}},
            ],
            "fixture_summary": {},
        }
        blocked = {
            **passing,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {"fixture": "carrion_only", "reason": f"b{index}"}
                    for index in range(6)
                ],
            },
            "fixture_horizons": [
                {"fixture_gate": {"passed": False}},
                {"fixture_gate": {"passed": False}},
            ],
            "fixture_horizon_summary": {
                **passing["fixture_horizon_summary"],
                "passed_horizon_count": 0,
            },
        }

        self.assertGreater(
            _fixture_selection_score_delta(passing),
            _fixture_selection_score_delta(blocked),
        )
        pressure = _fixture_blocker_pressure_summary(blocked["fixture_gate"])
        self.assertEqual(pressure["carrion_only_blocker_count"], 6)
        self.assertEqual(
            pressure["blocker_counts_by_fixture"],
            {"carrion_only": 6},
        )

    def test_mind_v3_fixture_blocker_pressure_weights_carrion_alive_gap(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _fixture_blocker_pressure_summary,
        )

        pressure = _fixture_blocker_pressure_summary(
            {
                "blockers": [
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_alive_floor",
                        "metric": "alive_agents_mean",
                        "floor": 1.0,
                        "value": 0.0,
                    },
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_hydration_viability_floor",
                        "metric": "hydration_viability_share_mean",
                        "floor": 0.2,
                        "value": 0.1,
                    },
                    {
                        "fixture": "plant_only",
                        "reason": "fixture_alive_floor",
                        "metric": "alive_agents_mean",
                        "floor": 1.0,
                        "value": 0.9,
                    },
                ],
            }
        )

        self.assertEqual(pressure["blocker_count"], 3)
        self.assertEqual(pressure["carrion_only_blocker_count"], 2)
        self.assertEqual(pressure["weighted_blocker_pressure"], 3.45)
        self.assertEqual(
            pressure["weighted_pressure_by_fixture"]["carrion_only"],
            3.225,
        )
        self.assertEqual(pressure["worst_fixture"], "carrion_only")
        self.assertEqual(pressure["worst_reason"], "fixture_alive_floor")

    def test_mind_v3_fixture_selection_nominee_pool_keeps_current_arch_lane(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _fixture_selection_candidate_pool,
        )
        from evolution_sim.mind.evolution import (
            MIND_V3_CONTROLLER_ARCHITECTURE,
            MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
        )

        def candidate(
            candidate_id: str,
            *,
            score: float,
            architecture: str,
            profile: str = "forager",
            warm_start: bool = False,
            balance: float = 0.0,
        ) -> dict[str, object]:
            payload = {
                "candidate_id": candidate_id,
                "candidate_index": int(candidate_id.rsplit("c", 1)[-1]),
                "score": score,
                "alive_agents_mean": score / 10.0,
                "births_mean": 1.0,
                "deaths_mean": 1.0,
                "alive_agent_ticks_per_tick_mean": 1.0,
                "resource_event_rate": 0.1,
                "movement_event_rate": 0.1,
                "dominant_requested_action_share": 0.4,
                "terminal_energy_hydration_balance_mean": balance,
                "terminal_energy_requirement_satisfaction_mean": balance,
                "terminal_energy_viability_share_mean": balance,
                "terminal_hydration_viability_share_mean": balance,
                "terminal_matched_diet_viability_share_mean": balance,
                "terminal_balanced_reproduction_readiness_mean": balance,
                "controller_metadata": {
                    "architecture": architecture,
                    "specialization_profile": profile,
                },
            }
            if warm_start:
                payload["warm_start"] = {
                    "policy": "fixture_archive_report_warm_start_v1"
                }
            return payload

        nominees = _fixture_selection_candidate_pool(
            [
                candidate(
                    "g0-c0",
                    score=100.0,
                    architecture=MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
                    warm_start=True,
                ),
                candidate(
                    "g0-c1",
                    score=20.0,
                    architecture=MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
                    profile="scavenger",
                ),
                candidate(
                    "g0-c2",
                    score=10.0,
                    architecture=MIND_V3_CONTROLLER_ARCHITECTURE,
                    balance=0.8,
                ),
                candidate(
                    "g0-c3",
                    score=5.0,
                    architecture=MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
                ),
            ],
            limit=3,
        )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in nominees],
            ["g0-c0", "g0-c1", "g0-c2"],
        )

    def test_mind_v3_archive_uses_fixture_selection_parent_lane(self) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _archive_from_candidates,
            _archive_parent_candidates,
        )

        def candidate(
            candidate_id: str,
            *,
            score: float,
            fixture_selection: dict[str, object] | None = None,
        ) -> dict[str, object]:
            payload = {
                "candidate_id": candidate_id,
                "candidate_index": int(candidate_id.rsplit("c", 1)[-1]),
                "parent_candidate_id": None,
                "generation_index": 0,
                "score": score,
                "alive_agents_mean": 8.0,
                "births_mean": 2.0,
                "deaths_mean": 1.0,
                "alive_agent_ticks_per_tick_mean": 4.0,
                "resource_event_rate": 0.2,
                "movement_event_rate": 0.1,
                "heuristic_action_source_count": 0,
                "controller_metadata": {"marker": candidate_id},
            }
            if fixture_selection is not None:
                payload["fixture_selection"] = fixture_selection
            return payload

        archive = _archive_from_candidates(
            [
                candidate("g0-c0", score=100.0),
                candidate(
                    "g0-c1",
                    score=20.0,
                    fixture_selection={
                        "fixture_gate_passed": False,
                        "passed_horizon_count": 1,
                        "first_horizon_passed": True,
                        "blocker_count": 2,
                        "score_delta": -1.0,
                    },
                ),
            ]
        )

        self.assertEqual(
            archive["elites"]["balanced"]["candidate_id"],
            "g0-c0",
        )
        self.assertEqual(
            archive["elites"]["fixture_selection"]["candidate_id"],
            "g0-c1",
        )
        self.assertEqual(
            _archive_parent_candidates(archive)[0]["candidate_id"],
            "g0-c1",
        )

    def test_mind_v3_fixture_rerank_combines_horizon_gate_blockers(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _combined_fixture_horizon_gate,
        )

        gate = _combined_fixture_horizon_gate(
            [
                {
                    "ticks": 80,
                    "fixture_gate": {
                        "passed": True,
                        "suite": "basic",
                        "blockers": [],
                        "per_fixture": {},
                    },
                },
                {
                    "ticks": 120,
                    "fixture_gate": {
                        "passed": False,
                        "suite": "basic",
                        "blockers": [
                            {
                                "fixture": "carrion_only",
                                "reason": "fixture_energy_viability_floor",
                                "metric": "energy_viability_share_mean",
                                "value": 0.0,
                                "floor": 0.2,
                            }
                        ],
                        "per_fixture": {},
                    },
                },
            ]
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["horizon_ticks"], [80, 120])
        self.assertEqual(gate["blockers"][0]["ticks"], 120)
        self.assertEqual(
            gate["blockers"][0]["reason"],
            "fixture_energy_viability_floor",
        )

    def test_mind_v3_fixture_rerank_candidate_evaluates_all_horizons(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _evaluate_fixture_rerank_candidate,
        )

        def suite_for_ticks(ticks: int) -> dict[str, object]:
            energy = 0.5 if ticks == 80 else 0.0
            return {
                "fixtures": [
                    {
                        "fixture": "carrion_only",
                        "comparison": {
                            "mind_v3": {
                                "runs": [],
                                "aggregate": {
                                    "alive_agents_mean": 1.0,
                                    "births_mean": 1.0,
                                    "terminal_energy_requirement_satisfaction_mean": energy,
                                    "terminal_energy_hydration_balance_mean": energy,
                                    "terminal_balanced_reproduction_readiness_mean": energy,
                                    "reproduction_failure_attribution": {
                                        "biologically_ready_agents_mean": 0.0,
                                        "terminal_viability_shares_mean": {
                                            "energy": energy,
                                            "hydration": energy,
                                            "health": 1.0,
                                            "matched_diet": energy,
                                        },
                                    },
                                },
                            }
                        },
                    }
                ]
            }

        def gate_for_config(
            *,
            fixture_suite: object,
            fixture_config: dict[str, object],
        ) -> dict[str, object]:
            ticks = int(fixture_config["ticks"])
            passed = ticks == 80
            return {
                "passed": passed,
                "suite": "basic",
                "blockers": []
                if passed
                else [
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_energy_viability_floor",
                    }
                ],
                "per_fixture": {},
            }

        with (
            patch(
                "evolution_sim.cli.mind_v3_evolve.run_mind_v3_fixture_suite",
                side_effect=lambda **kwargs: suite_for_ticks(int(kwargs["ticks"])),
            ) as run_suite,
            patch(
                "evolution_sim.cli.mind_v3_evolve.mind_v3_fixture_gate_status",
                side_effect=gate_for_config,
            ),
        ):
            runtime = _evaluate_fixture_rerank_candidate(
                candidate={
                    "candidate_id": "c0",
                    "score": 1.0,
                    "controller_metadata": {
                        "schema_version": "mind_v3_controller_metadata_v1"
                    },
                },
                prefilter_rank=0,
                holdout_seeds=[],
                ticks=80,
                rollout_workers=1,
                fixture_config={
                    "suite": "basic",
                    "seeds": [13],
                    "ticks": 80,
                },
                fixture_ticks=[80, 120],
            )

        self.assertEqual(
            [call.kwargs["ticks"] for call in run_suite.call_args_list],
            [80, 120],
        )
        self.assertFalse(runtime["fixture_gate"]["passed"])
        self.assertEqual(
            runtime["entry"]["fixture_horizon_summary"]["ticks"],
            [80, 120],
        )
        self.assertEqual(
            runtime["entry"]["fixture_horizon_summary"][
                "carrion_only_terminal_energy_viability_share_min"
            ],
            0.0,
        )

    def test_mind_v3_fixture_rerank_selection_uses_multi_horizon_carrion_readiness(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        short_pass_long_starves = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {
                        "fixture": "carrion_only",
                        "reason": "fixture_energy_viability_floor",
                        "ticks": 120,
                    }
                ],
            },
            "fixture_summary": {
                "alive_agents_mean": 4.0,
                "births_mean": 4.0,
                "mixed_stable_births_mean": 2.0,
                "terminal_reproduction_viability_min": 0.5,
                "terminal_energy_hydration_balance_min": 0.5,
            },
            "fixture_horizon_summary": {
                "carrion_only_terminal_energy_requirement_satisfaction_min": 0.1,
                "carrion_only_terminal_energy_viability_share_min": 0.0,
                "carrion_only_terminal_hydration_viability_share_min": 0.0,
                "carrion_only_terminal_matched_diet_viability_share_min": 0.25,
                "carrion_only_animal_resource_consumption_events_min": 10.0,
                "carrion_only_animal_resource_gained_energy_min": 2.0,
                "carrion_only_dominant_requested_action_share_max": 0.9,
                "carrion_only_alive_agents_min": 0.0,
                "carrion_only_births_min": 2.0,
                "terminal_energy_hydration_balance_min": 0.0,
                "terminal_reproduction_viability_min": 0.0,
                "mixed_stable_births_min": 2.0,
                "births_min": 2.0,
                "alive_agents_min": 0.0,
            },
            "holdout_aggregate": {
                "alive_agents_mean": 20.0,
                "births_mean": 8.0,
                "terminal_energy_viability_share_mean": 0.8,
                "terminal_energy_requirement_satisfaction_mean": 0.8,
                "terminal_hydration_viability_share_mean": 0.8,
                "terminal_health_viability_share_mean": 0.8,
                "terminal_matched_diet_viability_share_mean": 0.8,
                "dominant_requested_action_share": 0.4,
            },
        }
        weaker_search_less_bad_long_horizon = {
            **short_pass_long_starves,
            "prefilter_rank": 1,
            "search_score": 10.0,
            "fixture_horizon_summary": {
                **short_pass_long_starves["fixture_horizon_summary"],
                "carrion_only_terminal_energy_requirement_satisfaction_min": 0.45,
                "carrion_only_terminal_energy_viability_share_min": 0.25,
                "carrion_only_terminal_hydration_viability_share_min": 0.25,
                "carrion_only_terminal_matched_diet_viability_share_min": 0.5,
                "carrion_only_dominant_requested_action_share_max": 0.55,
                "carrion_only_alive_agents_min": 1.0,
                "terminal_energy_hydration_balance_min": 0.25,
                "terminal_reproduction_viability_min": 0.25,
            },
            "holdout_aggregate": {
                **short_pass_long_starves["holdout_aggregate"],
                "alive_agents_mean": 4.0,
                "births_mean": 1.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(weaker_search_less_bad_long_horizon),
            _fixture_rerank_selection_key(short_pass_long_starves),
        )

    def test_mind_v3_fixture_rerank_selection_uses_carrion_alive_ticks(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        base = {
            "prefilter_rank": 0,
            "search_score": 1.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {"fixture": "carrion_only", "reason": "fixture_alive"}
                ],
            },
            "fixture_summary": {},
            "fixture_horizon_summary": {
                "passed_horizon_count": 0,
                "carrion_only_terminal_energy_requirement_satisfaction_min": 0.2,
                "carrion_only_terminal_energy_viability_share_min": 0.0,
                "carrion_only_terminal_hydration_viability_share_min": 0.0,
                "carrion_only_terminal_matched_diet_viability_share_min": 0.25,
                "carrion_only_animal_resource_consumption_events_min": 8.0,
                "carrion_only_animal_resource_gained_energy_min": 2.0,
                "carrion_only_dominant_requested_action_share_max": 0.5,
                "carrion_only_alive_agents_min": 0.0,
                "carrion_only_births_min": 2.0,
                "terminal_energy_hydration_balance_min": 0.0,
                "terminal_reproduction_viability_min": 0.0,
                "mixed_stable_births_min": 1.0,
                "births_min": 2.0,
                "alive_agents_min": 0.0,
            },
            "holdout_aggregate": {},
        }
        short_survival = {
            **base,
            "fixture_horizon_summary": {
                **base["fixture_horizon_summary"],
                "carrion_only_alive_agent_ticks_per_tick_min": 0.5,
            },
        }
        longer_survival = {
            **base,
            "fixture_horizon_summary": {
                **base["fixture_horizon_summary"],
                "carrion_only_alive_agent_ticks_per_tick_min": 2.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(longer_survival),
            _fixture_rerank_selection_key(short_survival),
        )

    def test_mind_v3_fixture_rerank_prefers_partial_horizon_pass_coverage(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_selection_key

        fewer_blockers_no_horizon_pass = {
            "prefilter_rank": 0,
            "search_score": 100.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {"fixture": "carrion_only", "reason": "fixture_energy"},
                    {"fixture": "carrion_only", "reason": "fixture_alive"},
                ],
            },
            "fixture_horizons": [
                {"ticks": 80, "fixture_gate": {"passed": False}},
                {"ticks": 120, "fixture_gate": {"passed": False}},
            ],
            "fixture_horizon_summary": {
                "passed_horizon_count": 0,
                "carrion_only_terminal_energy_requirement_satisfaction_min": 0.45,
                "carrion_only_terminal_energy_viability_share_min": 0.0,
                "carrion_only_terminal_hydration_viability_share_min": 0.5,
                "carrion_only_terminal_matched_diet_viability_share_min": 0.5,
                "carrion_only_animal_resource_consumption_events_min": 16.0,
                "carrion_only_animal_resource_gained_energy_min": 4.0,
                "carrion_only_dominant_requested_action_share_max": 0.47,
                "carrion_only_alive_agents_min": 0.5,
                "carrion_only_births_min": 5.0,
                "terminal_energy_hydration_balance_min": 0.2,
                "terminal_reproduction_viability_min": 0.0,
                "mixed_stable_births_min": 6.0,
                "births_min": 11.0,
                "alive_agents_min": 14.0,
            },
            "fixture_summary": {},
            "holdout_aggregate": {
                "alive_agents_mean": 12.0,
                "births_mean": 5.0,
                "terminal_energy_viability_share_mean": 0.5,
                "terminal_energy_requirement_satisfaction_mean": 0.7,
                "terminal_hydration_viability_share_mean": 0.5,
                "terminal_health_viability_share_mean": 0.8,
                "terminal_matched_diet_viability_share_mean": 0.8,
                "dominant_requested_action_share": 0.45,
            },
        }
        more_blockers_short_horizon_pass = {
            **fewer_blockers_no_horizon_pass,
            "prefilter_rank": 1,
            "search_score": 50.0,
            "fixture_gate": {
                "passed": False,
                "blockers": [
                    {"fixture": "carrion_only", "reason": "fixture_energy"},
                    {"fixture": "carrion_only", "reason": "fixture_hydration"},
                    {"fixture": "carrion_only", "reason": "fixture_matched"},
                ],
            },
            "fixture_horizons": [
                {"ticks": 80, "fixture_gate": {"passed": True}},
                {"ticks": 120, "fixture_gate": {"passed": False}},
            ],
            "fixture_horizon_summary": {
                **fewer_blockers_no_horizon_pass["fixture_horizon_summary"],
                "passed_horizon_count": 1,
                "carrion_only_terminal_energy_requirement_satisfaction_min": 0.24,
            },
            "holdout_aggregate": {
                **fewer_blockers_no_horizon_pass["holdout_aggregate"],
                "alive_agents_mean": 8.0,
                "births_mean": 2.0,
            },
        }

        self.assertGreater(
            _fixture_rerank_selection_key(more_blockers_short_horizon_pass),
            _fixture_rerank_selection_key(fewer_blockers_no_horizon_pass),
        )

    def test_mind_v3_fixture_repair_builds_promotion_safe_bridge(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _fixture_promotion_bridge_repair_candidates,
        )

        def runtime(
            *,
            candidate_id: str,
            prefilter_rank: int,
            first_horizon_passed: bool,
            blocker_count: int,
            carrion_energy: float,
            holdout_alive: float,
            holdout_births: float,
        ) -> dict[str, object]:
            return {
                "candidate": {
                    "candidate_id": candidate_id,
                    "score": 10.0,
                    "founder_template_pool": [
                        {
                            "schema_version": "mind_v3_controller_metadata_v1",
                            "specialization_profile": f"{candidate_id}-{index}",
                            "action_head_weights": [float(index)],
                            "action_head_bias": [float(index)],
                        }
                        for index in range(3)
                    ],
                },
                "fixture_gate": {
                    "passed": False,
                    "blockers": [
                        {"fixture": "carrion_only", "reason": f"b{index}"}
                        for index in range(blocker_count)
                    ],
                },
                "entry": {
                    "candidate_id": candidate_id,
                    "prefilter_rank": prefilter_rank,
                    "search_score": 10.0,
                    "fixture_gate": {
                        "passed": False,
                        "blockers": [
                            {"fixture": "carrion_only", "reason": f"b{index}"}
                            for index in range(blocker_count)
                        ],
                    },
                    "fixture_horizons": [
                        {
                            "ticks": 80,
                            "fixture_gate": {"passed": first_horizon_passed},
                        },
                        {"ticks": 120, "fixture_gate": {"passed": False}},
                    ],
                    "fixture_horizon_summary": {
                        "passed_horizon_count": 1 if first_horizon_passed else 0,
                        "carrion_only_terminal_energy_requirement_satisfaction_min": (
                            carrion_energy
                        ),
                        "carrion_only_terminal_energy_viability_share_min": 0.0,
                        "carrion_only_terminal_hydration_viability_share_min": 0.0,
                        "carrion_only_terminal_matched_diet_viability_share_min": 0.25,
                        "carrion_only_alive_agents_min": 1.0,
                        "carrion_only_births_min": 4.0,
                        "carrion_only_animal_resource_consumption_events_min": 10.0,
                        "carrion_only_animal_resource_gained_energy_min": 2.0,
                    },
                    "fixture_summary": {},
                    "holdout_aggregate": {
                        "alive_agents_mean": holdout_alive,
                        "births_mean": holdout_births,
                        "terminal_energy_viability_share_mean": 0.5,
                        "terminal_energy_requirement_satisfaction_mean": 0.7,
                        "terminal_hydration_viability_share_mean": 0.5,
                        "terminal_health_viability_share_mean": 0.8,
                        "terminal_matched_diet_viability_share_mean": 0.8,
                        "dominant_requested_action_share": 0.45,
                    },
                },
            }

        bridges = _fixture_promotion_bridge_repair_candidates(
            [
                runtime(
                    candidate_id="safe80",
                    prefilter_rank=0,
                    first_horizon_passed=True,
                    blocker_count=3,
                    carrion_energy=0.23,
                    holdout_alive=13.0,
                    holdout_births=5.0,
                ),
                runtime(
                    candidate_id="unsafe120",
                    prefilter_rank=1,
                    first_horizon_passed=False,
                    blocker_count=2,
                    carrion_energy=0.45,
                    holdout_alive=11.5,
                    holdout_births=4.5,
                ),
                runtime(
                    candidate_id="unsafeweaker",
                    prefilter_rank=2,
                    first_horizon_passed=False,
                    blocker_count=4,
                    carrion_energy=0.2,
                    holdout_alive=16.0,
                    holdout_births=7.0,
                ),
            ]
        )

        self.assertEqual(len(bridges), 3)
        bridge_payloads = [bridge for _, bridge in bridges]
        self.assertEqual(
            [
                bridge["fixture_repair"]["donor_template_limit"]
                for bridge in bridge_payloads
            ],
            [1, 2, 4],
        )
        for bridge in bridge_payloads:
            self.assertEqual(bridge["parent_candidate_id"], "safe80")
            self.assertEqual(
                bridge["fixture_repair"]["primary_candidate_id"],
                "safe80",
            )
            self.assertEqual(
                bridge["fixture_repair"]["donor_candidate_id"],
                "unsafe120",
            )
            self.assertEqual(
                bridge["fixture_repair"]["donor_selection_reason"],
                "promotion_safe_bridge",
            )
        self.assertEqual(
            bridge_payloads[-1]["fixture_repair"]["donor_template_count"],
            3,
        )

    def test_mind_v3_fixture_rerank_ticks_deduplicates_and_adds_active_horizon(
        self,
    ) -> None:
        from argparse import Namespace

        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_ticks_from_args

        ticks = _fixture_rerank_ticks_from_args(
            Namespace(fixture_rerank_ticks="120,80,120"),
            default_ticks=80,
        )

        self.assertEqual(ticks, [80, 120])
        self.assertEqual(
            _fixture_rerank_ticks_from_args(
                Namespace(fixture_rerank_ticks="120"),
                default_ticks=80,
            ),
            [80, 120],
        )

    def test_mind_v3_fixture_summary_records_observed_animal_resource_intake(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_fixture_summary

        suite = {
            "fixtures": [
                {
                    "fixture": "carrion_only",
                    "comparison": {
                        "mind_v3": {
                            "runs": [
                                {
                                    "ticks": 80,
                                    "trajectory_record_count": 160,
                                    "fresh_kill_end": {
                                        "consumption_events": 1,
                                        "gained_energy": 0.2,
                                    },
                                    "carcass_end": {
                                        "consumption_events": 3,
                                        "gained_energy": 0.6,
                                    },
                                },
                                {
                                    "ticks": 80,
                                    "trajectory_record_count": 80,
                                    "fresh_kill_end": {
                                        "consumption_events": 0,
                                        "gained_energy": 0.0,
                                    },
                                    "carcass_end": {
                                        "consumption_events": 1,
                                        "gained_energy": 0.2,
                                    },
                                },
                            ],
                            "aggregate": {
                                "alive_agents_mean": 1.0,
                                "births_mean": 1.5,
                                "dominant_requested_action": "eat",
                                "dominant_requested_action_share": 0.8,
                                "terminal_energy_requirement_satisfaction_mean": 0.35,
                                "terminal_energy_hydration_balance_mean": 0.35,
                                "terminal_balanced_reproduction_readiness_mean": 0.2,
                                "reproduction_failure_attribution": {
                                    "biologically_ready_agents_mean": 0.0,
                                    "terminal_viability_shares_mean": {
                                        "energy": 0.0,
                                        "hydration": 0.5,
                                        "health": 0.0,
                                        "matched_diet": 0.5,
                                    },
                                },
                            },
                        }
                    },
                }
            ]
        }

        summary = _fixture_rerank_fixture_summary(suite)
        carrion = summary["per_fixture"]["carrion_only"]

        self.assertEqual(
            carrion["animal_resource_consumption_events_mean"],
            2.5,
        )
        self.assertEqual(
            summary["carrion_only_animal_resource_gained_energy_mean"],
            0.5,
        )
        self.assertEqual(
            carrion["terminal_energy_requirement_satisfaction_mean"],
            0.35,
        )
        self.assertEqual(carrion["alive_agent_ticks_per_tick_mean"], 1.5)
        self.assertEqual(
            summary["carrion_only_alive_agent_ticks_per_tick_mean"],
            1.5,
        )

    def test_mind_v3_fixture_repair_builds_composite_template_pool(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _fixture_repaired_candidate,
        )

        primary = {
            "candidate_id": "g1-c9",
            "parent_candidate_id": None,
            "score": 10.0,
            "controller_metadata": {
                "architecture": "a",
                "specialization_profile": "grazer",
                "action_head_weights": {"eat": [1.0]},
                "action_head_bias": {"eat": 0.1},
            },
            "founder_template_pool": [
                {
                    "architecture": "a",
                    "specialization_profile": "grazer",
                    "action_head_weights": {"eat": [1.0]},
                    "action_head_bias": {"eat": 0.1},
                }
            ],
        }
        donor = {
            "candidate_id": "g0-c4",
            "controller_metadata": {
                "architecture": "a",
                "specialization_profile": "scavenger",
                "action_head_weights": {"eat": [0.2]},
                "action_head_bias": {"eat": -0.1},
            },
            "founder_template_pool": [
                {
                    "architecture": "a",
                    "specialization_profile": "scavenger",
                    "action_head_weights": {"eat": [0.2]},
                    "action_head_bias": {"eat": -0.1},
                }
            ],
        }

        repaired = _fixture_repaired_candidate(primary, donor_candidate=donor)

        self.assertEqual(
            repaired["candidate_id"],
            "g1-c9+repair-g0-c4",
        )
        self.assertEqual(repaired["parent_candidate_id"], "g1-c9")
        self.assertEqual(repaired["founder_template_pool_size"], 2)
        self.assertEqual(
            repaired["fixture_repair"]["primary_candidate_id"],
            "g1-c9",
        )
        self.assertEqual(
            repaired["fixture_repair"]["donor_candidate_id"],
            "g0-c4",
        )
        self.assertEqual(
            repaired["founder_template_pool_specialization_profile_counts"],
            {"grazer": 1, "scavenger": 1},
        )

    def test_mind_v3_best_candidate_preserves_repaired_template_pool(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _candidate_with_founder_template_pool,
        )

        repaired = {
            "candidate_id": "g1-c9+repair-g0-c4",
            "controller_metadata": {"specialization_profile": "primary"},
            "fixture_repair": {"policy": "fixture_blocker_composite"},
            "founder_template_pool": [
                {
                    "architecture": "a",
                    "specialization_profile": "primary",
                    "action_head_weights": {"eat": [1.0]},
                    "action_head_bias": {"eat": 0.1},
                },
                {
                    "architecture": "a",
                    "specialization_profile": "donor",
                    "action_head_weights": {"eat": [0.2]},
                    "action_head_bias": {"eat": -0.1},
                },
            ],
            "founder_template_pool_size": 2,
        }

        enriched = _candidate_with_founder_template_pool(
            repaired,
            archive={"elites": {}},
            candidates=[],
        )

        self.assertEqual(enriched["founder_template_pool_size"], 2)
        self.assertEqual(
            enriched["founder_template_pool_specialization_profile_counts"],
            {"donor": 1, "primary": 1},
        )

    def test_mind_v3_archive_records_behavior_niche_elites(self) -> None:
        from evolution_sim.cli.mind_v3_evolve import _archive_from_candidates

        def candidate(
            candidate_id: str,
            *,
            score: float,
            alive: float,
            terrain: str,
            role: str,
            meat_mode: str,
            marker: str,
        ) -> dict[str, object]:
            return {
                "candidate_id": candidate_id,
                "candidate_index": int(candidate_id.split("-c")[-1]),
                "score": score,
                "alive_agents_mean": alive,
                "births_mean": 0.0,
                "deaths_mean": 1.0,
                "alive_agent_ticks_per_tick_mean": alive,
                "resource_event_rate": 0.0,
                "movement_event_rate": 0.0,
                "controller_metadata": {"marker": marker},
                "controller_lineage_elites": [
                    {
                        "dominant_terrain": terrain,
                        "trophic_role": role,
                        "meat_mode": meat_mode,
                        "controller_metadata": {"marker": marker},
                    }
                ],
            }

        archive = _archive_from_candidates(
            [
                candidate(
                    "g0-c0",
                    score=1.0,
                    alive=1.0,
                    terrain="forest",
                    role="herbivore",
                    meat_mode="none",
                    marker="weak_forest",
                ),
                candidate(
                    "g0-c1",
                    score=2.0,
                    alive=2.0,
                    terrain="forest",
                    role="herbivore",
                    meat_mode="none",
                    marker="strong_forest",
                ),
                candidate(
                    "g0-c2",
                    score=0.5,
                    alive=0.5,
                    terrain="wetland",
                    role="omnivore",
                    meat_mode="mixed",
                    marker="wetland",
                ),
            ]
        )

        self.assertEqual(archive["niche_count"], 2)
        niches = archive["behavior_niches"]
        forest = niches[
            "terrain=forest|trophic_role=herbivore|meat_mode=none"
        ]
        wetland = niches[
            "terrain=wetland|trophic_role=omnivore|meat_mode=mixed"
        ]
        self.assertEqual(forest["candidate"]["candidate_id"], "g0-c1")
        self.assertEqual(
            forest["lineage_elite"]["controller_metadata"]["marker"],
            "strong_forest",
        )
        self.assertEqual(wetland["candidate"]["candidate_id"], "g0-c2")

    def test_mind_v3_archive_preserves_scavenger_lane_elite(self) -> None:
        from evolution_sim.cli.mind_v3_evolve import _archive_from_candidates

        def candidate(
            candidate_id: str,
            *,
            score: float,
            marker: str,
            profile: str,
            meat_mode: str,
        ) -> dict[str, object]:
            return {
                "candidate_id": candidate_id,
                "candidate_index": int(candidate_id.split("-c")[-1]),
                "score": score,
                "alive_agents_mean": 3.0,
                "births_mean": 1.0,
                "deaths_mean": 1.0,
                "alive_agent_ticks_per_tick_mean": 3.0,
                "resource_event_rate": 0.2,
                "movement_event_rate": 0.1,
                "terminal_energy_hydration_balance_mean": 0.4,
                "terminal_balanced_reproduction_readiness_mean": 0.4,
                "terminal_matched_diet_viability_share_mean": 0.4,
                "controller_metadata": {
                    "marker": marker,
                    "specialization_profile": profile,
                },
                "behavior_descriptors": {
                    "meat_mode_counts": {meat_mode: 2},
                    "specialization_profile_counts": {profile: 2},
                },
                "controller_lineage_elites": [
                    {
                        "alive": True,
                        "record_count": 10,
                        "dominant_terrain": "rocky",
                        "trophic_role": "carnivore",
                        "meat_mode": meat_mode,
                        "controller_metadata": {
                            "marker": marker,
                            "specialization_profile": profile,
                        },
                    }
                ],
            }

        archive = _archive_from_candidates(
            [
                candidate(
                    "g0-c0",
                    score=10.0,
                    marker="global",
                    profile="forager",
                    meat_mode="none",
                ),
                candidate(
                    "g0-c1",
                    score=2.0,
                    marker="scavenger",
                    profile="scavenger",
                    meat_mode="scavenger",
                ),
            ]
        )

        self.assertIn("scavenger_lane", archive["elites"])
        self.assertEqual(
            archive["elites"]["scavenger_lane"]["controller_metadata"]["marker"],
            "scavenger",
        )

    def test_mind_v3_archive_parents_include_behavior_niche_templates(self) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _archive_from_candidates,
            _archive_parent_candidates,
        )

        candidate = {
            "candidate_id": "g0-c0",
            "candidate_index": 0,
            "score": 1.0,
            "alive_agents_mean": 1.0,
            "births_mean": 0.0,
            "deaths_mean": 1.0,
            "alive_agent_ticks_per_tick_mean": 1.0,
            "resource_event_rate": 0.0,
            "movement_event_rate": 0.0,
            "controller_metadata": {"marker": "balanced"},
            "controller_lineage_elites": [
                {
                    "dominant_terrain": "forest",
                    "trophic_role": "herbivore",
                    "meat_mode": "none",
                    "controller_metadata": {"marker": "forest"},
                },
                {
                    "dominant_terrain": "wetland",
                    "trophic_role": "omnivore",
                    "meat_mode": "mixed",
                    "controller_metadata": {"marker": "wetland"},
                },
            ],
        }

        archive = _archive_from_candidates([candidate])
        parent_markers = [
            dict(parent["controller_metadata"]).get("marker")
            for parent in _archive_parent_candidates(archive)
        ]

        self.assertIn("balanced", parent_markers)
        self.assertIn("forest", parent_markers)
        self.assertIn("wetland", parent_markers)

    def test_mind_v3_fixture_rerank_pool_includes_scavenger_lane_nominee(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_candidate_pool

        candidates = [
            {
                "candidate_id": "g0-c0",
                "candidate_index": 0,
                "score": 10.0,
                "alive_agents_mean": 10.0,
                "births_mean": 3.0,
                "controller_metadata": {"specialization_profile": "forager"},
            },
            {
                "candidate_id": "g0-c1",
                "candidate_index": 1,
                "score": 9.0,
                "alive_agents_mean": 9.0,
                "births_mean": 3.0,
                "controller_metadata": {"specialization_profile": "forager"},
            },
            {
                "candidate_id": "g0-c2",
                "candidate_index": 2,
                "score": 1.0,
                "alive_agents_mean": 2.0,
                "births_mean": 0.0,
                "controller_metadata": {"specialization_profile": "scavenger"},
                "behavior_descriptors": {
                    "meat_mode_counts": {"scavenger": 1},
                    "specialization_profile_counts": {"scavenger": 1},
                },
                "controller_lineage_elites": [
                    {
                        "alive": True,
                        "record_count": 5,
                        "dominant_terrain": "rocky",
                        "trophic_role": "carnivore",
                        "meat_mode": "scavenger",
                        "controller_metadata": {
                            "specialization_profile": "scavenger"
                        },
                    }
                ],
            },
        ]

        selected = _fixture_rerank_candidate_pool(candidates, limit=2)

        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected],
            ["g0-c0", "g0-c1", "g0-c2"],
        )

    def test_mind_v3_fixture_rerank_pool_includes_warm_start_nominee(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _fixture_rerank_candidate_pool

        candidates = [
            {
                "candidate_id": "g0-c0",
                "candidate_index": 0,
                "score": 10.0,
                "alive_agents_mean": 10.0,
                "births_mean": 3.0,
                "controller_metadata": {"specialization_profile": "forager"},
            },
            {
                "candidate_id": "g0-c1",
                "candidate_index": 1,
                "score": 9.0,
                "alive_agents_mean": 9.0,
                "births_mean": 3.0,
                "controller_metadata": {"specialization_profile": "forager"},
            },
            {
                "candidate_id": "g0-c2",
                "candidate_index": 2,
                "score": 1.0,
                "alive_agents_mean": 1.0,
                "births_mean": 0.0,
                "controller_metadata": {"specialization_profile": "forager"},
            },
            {
                "candidate_id": "g0-c3",
                "candidate_index": 3,
                "score": 0.5,
                "alive_agents_mean": 0.5,
                "births_mean": 0.0,
                "controller_metadata": {"specialization_profile": "scavenger"},
                "warm_start": {
                    "policy": "fixture_archive_report_warm_start_v1",
                },
            },
        ]

        selected = _fixture_rerank_candidate_pool(candidates, limit=2)

        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected],
            ["g0-c0", "g0-c1", "g0-c3"],
        )

    def test_mind_v3_best_candidate_gets_archive_diverse_founder_pool(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _archive_from_candidates,
            _candidate_with_founder_template_pool,
        )

        def metadata(profile: str, marker: str) -> dict[str, object]:
            return {
                "specialization_profile": profile,
                "source_marker": marker,
            }

        def candidate(
            candidate_id: str,
            *,
            score: float,
            alive: float,
            terrain: str,
            profile: str,
            marker: str,
        ) -> dict[str, object]:
            controller = metadata(profile, marker)
            return {
                "candidate_id": candidate_id,
                "candidate_index": int(candidate_id.split("-c")[-1]),
                "parent_candidate_id": None,
                "generation_index": 0,
                "score": score,
                "alive_agents_mean": alive,
                "births_mean": 0.0,
                "deaths_mean": 1.0,
                "alive_agent_ticks_per_tick_mean": alive,
                "resource_event_rate": 0.0,
                "movement_event_rate": 0.0,
                "heuristic_action_source_count": 0,
                "controller_metadata": controller,
                "controller_lineage_elites": [
                    {
                        "alive": True,
                        "reproduced_count": 0,
                        "record_count": 12,
                        "dominant_terrain": terrain,
                        "trophic_role": "herbivore",
                        "meat_mode": "none",
                        "controller_metadata": controller,
                    }
                ],
            }

        candidates = [
            candidate(
                "g0-c0",
                score=3.0,
                alive=3.0,
                terrain="wetland",
                profile="hydration_seeker",
                marker="best",
            ),
            candidate(
                "g0-c1",
                score=2.0,
                alive=2.0,
                terrain="forest",
                profile="forager",
                marker="forest",
            ),
            candidate(
                "g0-c2",
                score=1.5,
                alive=2.1,
                terrain="plain",
                profile="reproducer",
                marker="plain",
            ),
            candidate(
                "g0-c3",
                score=1.0,
                alive=1.0,
                terrain="desert",
                profile="predator_scavenger",
                marker="weak_predator",
            ),
        ]
        archive = _archive_from_candidates([candidates[0]])

        enriched = _candidate_with_founder_template_pool(
            candidates[0],
            archive=archive,
            candidates=candidates,
        )

        self.assertEqual(
            enriched["founder_template_pool_policy"],
            "archive_diverse_founder_template_pool_with_identity_diagnostics_v2",
        )
        self.assertEqual(
            enriched["founder_template_pool_min_alive_agents_mean"],
            2.0,
        )
        self.assertEqual(enriched["founder_template_pool_size"], 3)
        self.assertEqual(
            enriched["founder_template_pool_distinct_fingerprint_count"],
            3,
        )
        self.assertEqual(len(enriched["founder_template_pool_fingerprints"]), 3)
        self.assertEqual(
            enriched["founder_template_pool_specialization_profile_counts"],
            {"forager": 1, "hydration_seeker": 1, "reproducer": 1},
        )
        self.assertEqual(
            {
                metadata["source_marker"]
                for metadata in enriched["founder_template_pool"]
            },
            {"best", "forest", "plain"},
        )

    def test_mind_v3_search_founders_cover_specialization_profiles(self) -> None:
        from random import Random

        from evolution_sim.cli.mind_v3_evolve import _founder_candidates
        from evolution_sim.mind.evolution import MIND_V3_SPECIALIZATION_PROFILES

        candidates = _founder_candidates(
            population_size=len(MIND_V3_SPECIALIZATION_PROFILES),
            rng=Random(311),
        )

        self.assertEqual(
            [
                candidate["controller_metadata"]["specialization_profile"]
                for candidate in candidates
            ],
            list(MIND_V3_SPECIALIZATION_PROFILES),
        )

    def test_mind_v3_warm_start_reports_inject_template_pool_candidates(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.cli.mind_v3_evolve import (
            _founder_candidates,
            _inject_warm_start_candidates,
            _warm_start_candidates_from_reports,
        )
        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        best = founder_mind_v3_metadata(
            agent_id=1,
            rng=Random(1),
            specialization_profile="scavenger",
        )
        donor = founder_mind_v3_metadata(
            agent_id=2,
            rng=Random(2),
            specialization_profile="hydration_seeker",
        )

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "source.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "candidate_id": "g1-c7",
                            "controller_metadata": best,
                            "founder_template_pool": [best, donor],
                        },
                    }
                ),
                encoding="utf-8",
            )
            warm = _warm_start_candidates_from_reports(
                [report_path],
                candidate_limit=1,
                population_size=3,
                start_index=2,
            )

        candidates = _inject_warm_start_candidates(
            _founder_candidates(population_size=3, rng=Random(311)),
            warm_start_candidates=warm,
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[-1]["candidate_id"], "g0-c2")
        self.assertEqual(
            candidates[-1]["parent_candidate_id"],
            "g1-c7",
        )
        self.assertEqual(
            candidates[-1]["warm_start"]["policy"],
            "fixture_archive_report_warm_start_v1",
        )
        self.assertEqual(len(candidates[-1]["founder_template_pool"]), 2)

    def test_mind_v3_warm_start_reports_interleave_source_reports(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.cli.mind_v3_evolve import (
            _warm_start_candidates_from_reports,
        )
        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        first = founder_mind_v3_metadata(
            agent_id=1,
            rng=Random(1),
            specialization_profile="scavenger",
        )
        second = founder_mind_v3_metadata(
            agent_id=2,
            rng=Random(2),
            specialization_profile="predator_scavenger",
        )

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first_report = tmp_path / "first.json"
            second_report = tmp_path / "second.json"
            for report_path, candidate_id, metadata in (
                (first_report, "first-best", first),
                (second_report, "second-best", second),
            ):
                report_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "mind_v3_evolution_search_v1",
                            "best_candidate": {
                                "candidate_id": candidate_id,
                                "controller_metadata": metadata,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            warm = _warm_start_candidates_from_reports(
                [first_report, second_report],
                candidate_limit=2,
                population_size=4,
                start_index=2,
            )

        self.assertEqual(
            [candidate["candidate_id"] for candidate in warm],
            ["g0-c2", "g0-c3"],
        )
        self.assertEqual(
            [candidate["parent_candidate_id"] for candidate in warm],
            ["first-best", "second-best"],
        )
        self.assertEqual(
            [
                candidate["warm_start"]["source_report"].rsplit("/", 1)[-1]
                for candidate in warm
            ],
            ["first.json", "second.json"],
        )

    def test_mind_v3_warm_start_candidate_evaluates_full_template_pool(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.cli.mind_v3_evolve import _candidate_evaluation_template
        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        first = founder_mind_v3_metadata(agent_id=1, rng=Random(1))
        second = founder_mind_v3_metadata(agent_id=2, rng=Random(2))

        selected = _candidate_evaluation_template(
            {
                "candidate_id": "g0-c2",
                "controller_metadata": first,
                "founder_template_pool": [first, second],
                "warm_start": {
                    "policy": "fixture_archive_report_warm_start_v1",
                },
            }
        )

        self.assertIsInstance(selected, list)
        self.assertEqual(len(selected), 2)

    def test_mind_v3_evolve_cli_resumes_from_checkpoint_report(self) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            checkpoint_path = tmp_path / "checkpoint.json"
            resumed_path = tmp_path / "resumed.json"
            first_argv = [
                "mind_v3_evolve",
                "--seeds",
                "5",
                "--ticks",
                "10",
                "--population-size",
                "2",
                "--generations",
                "1",
                "--output",
                str(checkpoint_path),
            ]
            resumed_argv = [
                "mind_v3_evolve",
                "--resume-from",
                str(checkpoint_path),
                "--generations",
                "2",
                "--output",
                str(resumed_path),
            ]
            with (
                patch("sys.argv", first_argv),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()
            with (
                patch("sys.argv", resumed_argv),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            resumed = json.loads(resumed_path.read_text(encoding="utf-8"))

        self.assertEqual(len(checkpoint["generations"]), 1)
        self.assertEqual(len(resumed["generations"]), 2)
        self.assertEqual(
            resumed["generations"][0]["best_candidate_id"],
            checkpoint["generations"][0]["best_candidate_id"],
        )
        self.assertEqual(resumed["resume"]["completed_generations"], 2)
        self.assertEqual(resumed["search"]["resumed_from"], str(checkpoint_path))

    def test_mind_v3_evolve_candidate_promotes_runtime_lineage_controller(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.cli.mind_v3_evolve import _evaluate_candidate
        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        founder = founder_mind_v3_metadata(agent_id=1, rng=Random(1))
        survivor = founder_mind_v3_metadata(agent_id=2, rng=Random(2))
        reproducer = founder_mind_v3_metadata(agent_id=3, rng=Random(3))

        with patch(
            "evolution_sim.cli.mind_v3_evolve._run_candidate",
            return_value={
                "seed": 5,
                "ticks": 12,
                "ticks_executed": 12,
                "alive_agents": 2,
                "births": 1,
                "deaths": 18,
                "trajectory_record_count": 24,
                "alive_agent_ticks": 24,
                "alive_agent_ticks_per_tick": 2.0,
                "resource_event_count": 3,
                "resource_event_rate": 0.125,
                "movement_event_count": 4,
                "movement_event_rate": 0.1667,
                "unique_requested_actions": 3,
                "heuristic_action_source_count": 0,
                "action_source_counts": {
                    "mind_v3_autonomous_evolution_policy_v1": 24,
                },
                "policy_id_counts": {
                    "mind_v3_autonomous_evolution_policy": 24,
                },
                "behavior_descriptors": {
                    "terrain_occupancy": {"forest": 7, "wetland": 3},
                    "dominant_terrain": "forest",
                    "trophic_role_counts": {"herbivore": 2},
                    "meat_mode_counts": {"none": 2},
                },
                "controller_lineage_elites": [
                    {
                        "selection_reason": "survivor",
                        "agent_id": 2,
                        "lineage_id": 2,
                        "alive": True,
                        "reproduced_count": 0,
                        "last_tick": 11,
                        "dominant_terrain": "forest",
                        "trophic_role": "herbivore",
                        "meat_mode": "none",
                        "controller_metadata": survivor,
                    },
                    {
                        "selection_reason": "reproducer",
                        "agent_id": 3,
                        "lineage_id": 3,
                        "alive": False,
                        "reproduced_count": 1,
                        "last_tick": 10,
                        "dominant_terrain": "wetland",
                        "trophic_role": "omnivore",
                        "meat_mode": "mixed",
                        "controller_metadata": reproducer,
                    },
                ],
            },
        ):
            evaluated = _evaluate_candidate(
                {
                    "candidate_id": "g0-c0",
                    "parent_candidate_id": None,
                    "controller_metadata": founder,
                },
                generation_index=0,
                seeds=[5],
                ticks=12,
            )

        self.assertEqual(
            evaluated["controller_selection_policy"],
            "runtime_lineage_elite_controller_selection_v1",
        )
        self.assertEqual(evaluated["founder_template_metadata"], founder)
        self.assertEqual(evaluated["controller_metadata"], survivor)
        self.assertEqual(
            evaluated["controller_lineage_elites"][0]["selection_reason"],
            "survivor",
        )
        self.assertEqual(
            evaluated["behavior_descriptors"]["terrain_occupancy"],
            {"forest": 7, "wetland": 3},
        )

    def test_mind_v3_evolve_next_generation_uses_lineage_elite_templates(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.cli.mind_v3_evolve import _next_generation

        selected = {
            "schema_version": "mind_v3_controller_metadata_v1",
            "state_size": 0,
            "source_marker": "selected",
        }
        survivor = {
            "schema_version": "mind_v3_controller_metadata_v1",
            "state_size": 0,
            "source_marker": "survivor",
        }
        reproducer = {
            "schema_version": "mind_v3_controller_metadata_v1",
            "state_size": 0,
            "source_marker": "reproducer",
        }

        def fake_inherit_mind_v3_metadata(**kwargs: object) -> dict[str, object]:
            parent = dict(kwargs["primary_parent_metadata"])
            return {"inherited_from": parent["source_marker"]}

        with patch(
            "evolution_sim.cli.mind_v3_evolve.inherit_mind_v3_metadata",
            side_effect=fake_inherit_mind_v3_metadata,
        ):
            children = _next_generation(
                [
                    {
                        "candidate_id": "g0-c0",
                        "controller_metadata": selected,
                        "controller_lineage_elites": [
                            {"controller_metadata": survivor},
                            {"controller_metadata": reproducer},
                        ],
                    }
                ],
                generation_index=1,
                population_size=3,
                rng=Random(5),
            )

        self.assertEqual(children[0]["controller_metadata"], selected)
        self.assertEqual(children[1]["controller_metadata"], {"inherited_from": "survivor"})
        self.assertEqual(children[2]["controller_metadata"], {"inherited_from": "reproducer"})

    def test_mind_v3_evolve_cli_writes_holdout_gated_curriculum_report(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-curriculum.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evolve",
                        "--seeds",
                        "5",
                        "--holdout-seeds",
                        "13",
                        "--curriculum-ticks",
                        "8,10",
                        "--curriculum-min-holdout-alive",
                        "0",
                        "--curriculum-min-holdout-births",
                        "0",
                        "--curriculum-min-holdout-energy-viability",
                        "0",
                        "--curriculum-min-holdout-hydration-viability",
                        "0",
                        "--curriculum-min-holdout-health-viability",
                        "0",
                        "--curriculum-min-holdout-matched-diet-viability",
                        "0",
                        "--curriculum-min-holdout-biologically-ready",
                        "0",
                        "--curriculum-min-holdout-ready-agent-tick-share",
                        "0",
                        "--curriculum-min-holdout-ready-pair-tick-share",
                        "0",
                        "--population-size",
                        "2",
                        "--generations",
                        "1",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["schema_version"], "mind_v3_evolution_search_v1")
        self.assertEqual(
            report["curriculum"]["policy"],
            "holdout_gated_tick_curriculum_v1",
        )
        self.assertEqual(report["curriculum"]["tick_schedule"], [8, 10])
        self.assertEqual(report["curriculum"]["completed_stage_count"], 2)
        self.assertEqual(report["curriculum"]["stopped_reason"], "complete")
        self.assertEqual(report["curriculum"]["min_holdout_energy_viability"], 0.0)
        self.assertEqual(
            report["curriculum"]["min_holdout_hydration_viability"],
            0.0,
        )
        self.assertEqual(report["curriculum"]["min_holdout_health_viability"], 0.0)
        self.assertEqual(
            report["curriculum"]["min_holdout_matched_diet_viability"],
            0.0,
        )
        self.assertEqual(
            report["curriculum"]["min_holdout_biologically_ready"],
            0.0,
        )
        self.assertEqual(
            report["curriculum"]["min_holdout_ready_agent_tick_share"],
            0.0,
        )
        self.assertEqual(
            report["curriculum"]["min_holdout_ready_pair_tick_share"],
            0.0,
        )
        self.assertEqual(len(report["stages"]), 2)
        self.assertEqual(
            [stage["curriculum_stage"]["ticks"] for stage in report["stages"]],
            [8, 10],
        )
        self.assertIn(
            "holdout_terminal_energy_viability_share_mean",
            report["stages"][0]["curriculum_stage"],
        )
        self.assertIn(
            "holdout_terminal_hydration_viability_share_mean",
            report["stages"][0]["curriculum_stage"],
        )
        self.assertIn(
            "holdout_reproduction_ready_agent_tick_share_mean",
            report["stages"][0]["curriculum_stage"],
        )
        self.assertIn("controller_metadata", report["best_candidate"])
        self.assertEqual(
            report["holdout_evaluation"]["aggregate"][
                "heuristic_action_source_count"
            ],
            0,
        )

    def test_mind_v3_evolve_cli_stops_curriculum_on_holdout_alive_floor(
        self,
    ) -> None:
        from evolution_sim.cli import mind_v3_evolve

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-v3-curriculum-stop.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evolve",
                        "--seeds",
                        "5",
                        "--holdout-seeds",
                        "13",
                        "--curriculum-ticks",
                        "8,10",
                        "--curriculum-min-holdout-alive",
                        "999",
                        "--population-size",
                        "2",
                        "--generations",
                        "1",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evolve.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["curriculum"]["stopped_reason"],
            "holdout_alive_floor",
        )
        self.assertEqual(report["curriculum"]["completed_stage_count"], 1)
        self.assertFalse(report["curriculum"]["selected_stage_passed"])
        self.assertEqual(len(report["stages"]), 1)
        self.assertEqual(report["stages"][0]["curriculum_stage"]["stage_index"], 0)
        self.assertFalse(report["stages"][0]["curriculum_stage"]["passed"])
        self.assertIn("controller_metadata", report["best_candidate"])

    def test_mind_v3_holdout_evaluation_reports_behavior_niche_count(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _evaluate_holdout

        def run(seed: int, terrain: str, role: str, mode: str) -> dict[str, object]:
            return {
                "seed": seed,
                "ticks": 12,
                "ticks_executed": 12,
                "alive_agents": 2,
                "births": 0,
                "deaths": 18,
                "trajectory_record_count": 24,
                "alive_agent_ticks": 24,
                "alive_agent_ticks_per_tick": 2.0,
                "resource_event_count": 8,
                "resource_event_rate": 0.3333,
                "movement_event_count": 4,
                "movement_event_rate": 0.1667,
                "unique_requested_actions": 3,
                "heuristic_action_source_count": 0,
                "action_source_counts": {
                    "mind_v3_autonomous_evolution_policy_v1": 24,
                },
                "policy_id_counts": {
                    "mind_v3_autonomous_evolution_policy": 24,
                },
                "behavior_descriptors": {
                    "policy": "mind_v3_behavior_descriptor_v1",
                    "terrain_occupancy": {terrain: 24},
                    "dominant_terrain_counts": {terrain: 1},
                    "trophic_role_counts": {role: 1},
                    "meat_mode_counts": {mode: 1},
                    "requested_action_counts": {"eat": 12, "drink": 8, "move_east": 4},
                    "action_source_counts": {
                        "mind_v3_autonomous_evolution_policy_v1": 24,
                    },
                    "lineage_count": 1,
                    "runtime_species_count": 1,
                    "runtime_ecotype_count": 1,
                },
                "controller_lineage_elites": [
                    {
                        "dominant_terrain": terrain,
                        "trophic_role": role,
                        "meat_mode": mode,
                    }
                ],
            }

        with patch(
            "evolution_sim.cli.mind_v3_evolve._run_candidate",
            side_effect=[
                run(13, "forest", "herbivore", "none"),
                run(17, "wetland", "omnivore", "mixed"),
            ],
        ):
            holdout = _evaluate_holdout(
                seeds=[13, 17],
                ticks=12,
                metadata={"schema_version": "mind_v3_controller_metadata_v1"},
            )

        aggregate = holdout["aggregate"]
        self.assertEqual(aggregate["behavior_niche_count"], 2)
        self.assertEqual(
            aggregate["behavior_niche_keys"],
            [
                "terrain=forest|trophic_role=herbivore|meat_mode=none",
                "terrain=wetland|trophic_role=omnivore|meat_mode=mixed",
            ],
        )
        self.assertEqual(
            aggregate["behavior_descriptors"]["terrain_occupancy"],
            {"forest": 24, "wetland": 24},
        )

    def test_mind_v3_holdout_evaluation_reports_action_collapse_diagnostics(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _evaluate_holdout

        def run(
            *,
            seed: int,
            requested_actions: dict[str, int],
            resolved_actions: dict[str, int],
            elites: list[dict[str, object]],
        ) -> dict[str, object]:
            record_count = sum(requested_actions.values())
            return {
                "seed": seed,
                "ticks": 12,
                "ticks_executed": 12,
                "alive_agents": 2,
                "births": 0,
                "deaths": 18,
                "trajectory_record_count": record_count,
                "alive_agent_ticks": record_count,
                "alive_agent_ticks_per_tick": 2.0,
                "resource_event_count": 8,
                "resource_event_rate": 0.3333,
                "movement_event_count": 4,
                "movement_event_rate": 0.1667,
                "unique_requested_actions": len(requested_actions),
                "heuristic_action_source_count": 0,
                "requested_action_counts": requested_actions,
                "resolved_action_counts": resolved_actions,
                "action_source_counts": {
                    "mind_v3_autonomous_evolution_policy_v1": record_count,
                },
                "policy_id_counts": {
                    "mind_v3_autonomous_evolution_policy": record_count,
                },
                "behavior_descriptors": {
                    "policy": "mind_v3_behavior_descriptor_v1",
                    "terrain_occupancy": {"plain": record_count},
                    "dominant_terrain_counts": {"plain": 1},
                    "trophic_role_counts": {"herbivore": 1},
                    "meat_mode_counts": {"none": 1},
                    "requested_action_counts": requested_actions,
                    "action_source_counts": {
                        "mind_v3_autonomous_evolution_policy_v1": record_count,
                    },
                    "lineage_count": 1,
                    "runtime_species_count": 1,
                    "runtime_ecotype_count": 1,
                },
                "controller_lineage_elites": elites,
            }

        with patch(
            "evolution_sim.cli.mind_v3_evolve._run_candidate",
            side_effect=[
                run(
                    seed=13,
                    requested_actions={"eat": 8, "drink": 2},
                    resolved_actions={"eat": 8, "drink": 2},
                    elites=[
                        {
                            "alive": True,
                            "reproduced_count": 0,
                            "movement_event_rate": 0.0,
                            "dominant_terrain": "forest",
                            "trophic_role": "herbivore",
                            "meat_mode": "none",
                        },
                        {
                            "alive": False,
                            "reproduced_count": 0,
                            "movement_event_rate": 0.0,
                            "dominant_terrain": "wetland",
                            "trophic_role": "omnivore",
                            "meat_mode": "mixed",
                        },
                    ],
                ),
                run(
                    seed=17,
                    requested_actions={"eat": 7, "move_east": 3},
                    resolved_actions={"eat": 7, "move_east": 3},
                    elites=[
                        {
                            "alive": False,
                            "reproduced_count": 0,
                            "movement_event_rate": 0.25,
                            "dominant_terrain": "plain",
                            "trophic_role": "carnivore",
                            "meat_mode": "hunter",
                        },
                    ],
                ),
            ],
        ):
            holdout = _evaluate_holdout(
                seeds=[13, 17],
                ticks=12,
                metadata={"schema_version": "mind_v3_controller_metadata_v1"},
            )

        aggregate = holdout["aggregate"]
        self.assertEqual(
            aggregate["requested_action_counts"],
            {"drink": 2, "eat": 15, "move_east": 3},
        )
        self.assertEqual(
            aggregate["resolved_action_counts"],
            {"drink": 2, "eat": 15, "move_east": 3},
        )
        self.assertEqual(aggregate["dominant_requested_action"], "eat")
        self.assertEqual(aggregate["dominant_requested_action_count"], 15)
        self.assertEqual(aggregate["dominant_requested_action_share"], 0.75)
        self.assertEqual(aggregate["behavior_niche_count"], 3)
        self.assertEqual(aggregate["active_behavior_niche_count"], 2)
        self.assertEqual(
            aggregate["active_behavior_niche_keys"],
            [
                "terrain=forest|trophic_role=herbivore|meat_mode=none",
                "terrain=plain|trophic_role=carnivore|meat_mode=hunter",
            ],
        )

    def test_mind_v3_curriculum_status_blocks_low_holdout_behavior(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _curriculum_stage_status

        status = _curriculum_stage_status(
            stage_report={
                "holdout_evaluation": {
                    "aggregate": {
                        "alive_agents_mean": 4.0,
                        "births_mean": 1.0,
                        "movement_event_rate": 0.0,
                        "unique_requested_actions_mean": 1.0,
                        "behavior_niche_count": 1,
                    }
                }
            },
            stage_index=0,
            ticks=80,
            min_holdout_alive=1.0,
            min_holdout_births=0.0,
            min_holdout_movement_rate=0.01,
            min_holdout_unique_actions=2.0,
            min_holdout_behavior_niches=2,
        )

        self.assertFalse(status["passed"])
        self.assertEqual(status["stopped_reason"], "holdout_movement_floor")
        self.assertEqual(status["holdout_behavior_niche_count"], 1)
        self.assertEqual(status["min_holdout_behavior_niches"], 2)

    def test_mind_v3_curriculum_status_blocks_dominant_action_collapse(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _curriculum_stage_status

        status = _curriculum_stage_status(
            stage_report={
                "holdout_evaluation": {
                    "aggregate": {
                        "alive_agents_mean": 4.0,
                        "births_mean": 1.0,
                        "movement_event_rate": 0.05,
                        "unique_requested_actions_mean": 3.0,
                        "dominant_requested_action": "eat",
                        "dominant_requested_action_share": 0.96,
                        "behavior_niche_count": 3,
                        "active_behavior_niche_count": 2,
                    }
                }
            },
            stage_index=0,
            ticks=80,
            min_holdout_alive=1.0,
            min_holdout_births=0.0,
            min_holdout_movement_rate=0.01,
            min_holdout_unique_actions=2.0,
            min_holdout_behavior_niches=2,
            max_holdout_dominant_action_share=0.9,
            min_holdout_active_behavior_niches=1,
        )

        self.assertFalse(status["passed"])
        self.assertEqual(
            status["stopped_reason"],
            "holdout_dominant_action_share_ceiling",
        )
        self.assertEqual(status["holdout_dominant_requested_action"], "eat")
        self.assertEqual(status["holdout_dominant_requested_action_share"], 0.96)
        self.assertEqual(status["max_holdout_dominant_action_share"], 0.9)

    def test_mind_v3_curriculum_status_blocks_static_only_niches(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _curriculum_stage_status

        status = _curriculum_stage_status(
            stage_report={
                "holdout_evaluation": {
                    "aggregate": {
                        "alive_agents_mean": 4.0,
                        "births_mean": 1.0,
                        "movement_event_rate": 0.05,
                        "unique_requested_actions_mean": 3.0,
                        "dominant_requested_action": "eat",
                        "dominant_requested_action_share": 0.7,
                        "behavior_niche_count": 6,
                        "active_behavior_niche_count": 1,
                    }
                }
            },
            stage_index=0,
            ticks=80,
            min_holdout_alive=1.0,
            min_holdout_births=0.0,
            min_holdout_movement_rate=0.01,
            min_holdout_unique_actions=2.0,
            min_holdout_behavior_niches=2,
            max_holdout_dominant_action_share=0.9,
            min_holdout_active_behavior_niches=2,
        )

        self.assertFalse(status["passed"])
        self.assertEqual(
            status["stopped_reason"],
            "holdout_active_behavior_niche_floor",
        )
        self.assertEqual(status["holdout_behavior_niche_count"], 6)
        self.assertEqual(status["holdout_active_behavior_niche_count"], 1)
        self.assertEqual(status["min_holdout_active_behavior_niches"], 2)

    def test_mind_v3_curriculum_status_blocks_low_reproduction_viability(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _curriculum_stage_status

        base_aggregate = {
            "alive_agents_mean": 4.0,
            "births_mean": 1.0,
            "movement_event_rate": 0.05,
            "unique_requested_actions_mean": 3.0,
            "dominant_requested_action": "drink",
            "dominant_requested_action_share": 0.5,
            "behavior_niche_count": 6,
            "active_behavior_niche_count": 3,
            "terminal_energy_viability_share_mean": 0.25,
            "terminal_hydration_viability_share_mean": 0.25,
            "terminal_health_viability_share_mean": 0.25,
            "terminal_matched_diet_viability_share_mean": 0.25,
            "biologically_reproduction_ready_agents_mean": 0.25,
            "reproduction_ready_agent_tick_share_mean": 0.2,
            "reproduction_ready_pair_tick_share_mean": 0.1,
        }
        cases = [
            (
                {"terminal_energy_viability_share_mean": 0.0},
                "holdout_energy_viability_floor",
            ),
            (
                {"terminal_hydration_viability_share_mean": 0.0},
                "holdout_hydration_viability_floor",
            ),
            (
                {"terminal_health_viability_share_mean": 0.0},
                "holdout_health_viability_floor",
            ),
            (
                {"terminal_matched_diet_viability_share_mean": 0.0},
                "holdout_matched_diet_viability_floor",
            ),
            (
                {"biologically_reproduction_ready_agents_mean": 0.0},
                "holdout_biological_reproduction_readiness_floor",
            ),
            (
                {"reproduction_ready_agent_tick_share_mean": 0.0},
                "holdout_sustained_ready_agent_tick_floor",
            ),
            (
                {"reproduction_ready_pair_tick_share_mean": 0.0},
                "holdout_sustained_ready_pair_tick_floor",
            ),
        ]

        for overrides, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                aggregate = {**base_aggregate, **overrides}
                status = _curriculum_stage_status(
                    stage_report={
                        "holdout_evaluation": {"aggregate": aggregate}
                    },
                    stage_index=0,
                    ticks=80,
                    min_holdout_alive=1.0,
                    min_holdout_births=0.0,
                    min_holdout_movement_rate=0.01,
                    min_holdout_unique_actions=2.0,
                    min_holdout_behavior_niches=2,
                    max_holdout_dominant_action_share=0.9,
                    min_holdout_active_behavior_niches=2,
                    min_holdout_energy_viability=0.1,
                    min_holdout_hydration_viability=0.1,
                    min_holdout_health_viability=0.1,
                    min_holdout_matched_diet_viability=0.1,
                    min_holdout_biologically_ready=0.1,
                    min_holdout_ready_agent_tick_share=0.1,
                    min_holdout_ready_pair_tick_share=0.05,
                )

                self.assertFalse(status["passed"])
                self.assertEqual(status["stopped_reason"], expected_reason)
                self.assertEqual(
                    status["min_holdout_energy_viability"],
                    0.1,
                )
                self.assertEqual(
                    status[
                        "holdout_terminal_matched_diet_viability_share_mean"
                    ],
                    aggregate["terminal_matched_diet_viability_share_mean"],
                )
                self.assertEqual(
                    status[
                        "holdout_terminal_hydration_viability_share_mean"
                    ],
                    aggregate["terminal_hydration_viability_share_mean"],
                )
                self.assertEqual(
                    status[
                        "holdout_biologically_reproduction_ready_agents_mean"
                    ],
                    aggregate["biologically_reproduction_ready_agents_mean"],
                )

    def test_mind_v3_curriculum_status_blocks_fixture_gate_failure(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _curriculum_stage_status

        status = _curriculum_stage_status(
            stage_report={
                "holdout_evaluation": {
                    "aggregate": {
                        "alive_agents_mean": 4.0,
                        "births_mean": 1.0,
                        "movement_event_rate": 0.05,
                        "unique_requested_actions_mean": 3.0,
                        "dominant_requested_action": "eat",
                        "dominant_requested_action_share": 0.4,
                        "behavior_niche_count": 4,
                        "active_behavior_niche_count": 2,
                        "terminal_energy_viability_share_mean": 1.0,
                        "terminal_health_viability_share_mean": 1.0,
                        "terminal_matched_diet_viability_share_mean": 1.0,
                        "biologically_reproduction_ready_agents_mean": 1.0,
                    }
                },
                "fixture_gate": {
                    "passed": False,
                    "blockers": [
                        {
                            "fixture": "mixed_stable",
                            "reason": "fixture_mixed_stable_birth_floor",
                        }
                    ],
                },
            },
            stage_index=0,
            ticks=80,
            min_holdout_alive=1.0,
            min_holdout_births=0.0,
            min_holdout_movement_rate=0.01,
            min_holdout_unique_actions=2.0,
            min_holdout_behavior_niches=2,
            max_holdout_dominant_action_share=0.9,
            min_holdout_active_behavior_niches=2,
            min_holdout_energy_viability=0.1,
            min_holdout_health_viability=0.1,
            min_holdout_matched_diet_viability=0.1,
            min_holdout_biologically_ready=0.1,
        )

        self.assertFalse(status["passed"])
        self.assertEqual(status["stopped_reason"], "fixture_gate_floor")
        self.assertFalse(status["fixture_gate_passed"])
        self.assertEqual(
            status["fixture_gate_blockers"][0]["reason"],
            "fixture_mixed_stable_birth_floor",
        )

    def test_mind_v3_evolve_score_prefers_survival_over_resource_loop(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        resource_loop_with_birth = {
            "alive_agents_mean": 0.3333,
            "births_mean": 0.3333,
            "deaths_mean": 20.0,
            "alive_agent_ticks_per_tick_mean": 10.8528,
            "resource_event_rate": 0.9839,
            "movement_event_rate": 0.0077,
            "unique_requested_actions_mean": 5.3333,
        }
        sustained_survival_without_birth = {
            "alive_agents_mean": 4.0,
            "births_mean": 0.0,
            "deaths_mean": 16.0,
            "alive_agent_ticks_per_tick_mean": 8.0,
            "resource_event_rate": 0.2,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 5.0,
        }

        loop_score = _score_components_total(
            _score_components(resource_loop_with_birth)
        )
        survival_score = _score_components_total(
            _score_components(sustained_survival_without_birth)
        )

        self.assertGreater(survival_score, loop_score)

    def test_mind_v3_evolve_score_prefers_terminal_reproduction_viability(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _run_reproduction_viability,
            _score_components,
            _score_components_total,
        )

        starving_summary = {
            "alive_agents": 4,
            "reproduction_end": {
                "ready_agents": 0,
                "biologically_ready_agents": 0,
                "biological_blocker_counts": {
                    "energy": 4,
                    "hydration": 4,
                    "health": 4,
                    "matched_diet": 4,
                },
                "energy_readiness_by_meat_mode": {
                    "none": {
                        "alive_agents": 4,
                        "energy_shortfall_agents": 4,
                        "energy_total": 4.0,
                        "energy_required_total": 12.0,
                        "energy_gap_total": 8.0,
                    }
                },
            },
        }
        viable_summary = {
            "alive_agents": 4,
            "reproduction_end": {
                "ready_agents": 1,
                "biologically_ready_agents": 2,
                "biological_blocker_counts": {
                    "energy": 0,
                    "hydration": 0,
                    "health": 1,
                    "matched_diet": 1,
                },
                "biologically_ready_group_count": 2,
                "ready_group_count": 1,
                "energy_readiness_by_meat_mode": {
                    "none": {
                        "alive_agents": 4,
                        "energy_shortfall_agents": 0,
                        "energy_total": 14.0,
                        "energy_required_total": 12.0,
                        "energy_gap_total": 0.0,
                    }
                },
            },
        }
        starving_viability = _run_reproduction_viability(starving_summary)
        viable_viability = _run_reproduction_viability(viable_summary)
        base = {
            "alive_agents_mean": 4.0,
            "births_mean": 0.0,
            "deaths_mean": 16.0,
            "alive_agent_ticks_per_tick_mean": 8.0,
            "resource_event_rate": 0.2,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 5.0,
        }

        starving_score = _score_components_total(
            _score_components(
                {
                    **base,
                    "terminal_energy_viability_share_mean": starving_viability[
                        "terminal_energy_viability_share"
                    ],
                    "terminal_energy_requirement_satisfaction_mean": (
                        starving_viability[
                            "terminal_energy_requirement_satisfaction"
                        ]
                    ),
                    "terminal_hydration_viability_share_mean": starving_viability[
                        "terminal_hydration_viability_share"
                    ],
                    "terminal_health_viability_share_mean": starving_viability[
                        "terminal_health_viability_share"
                    ],
                    "terminal_matched_diet_viability_share_mean": starving_viability[
                        "terminal_matched_diet_viability_share"
                    ],
                    "reproduction_ready_agents_mean": starving_viability[
                        "reproduction_ready_agents"
                    ],
                    "biologically_reproduction_ready_agents_mean": starving_viability[
                        "biologically_reproduction_ready_agents"
                    ],
                    "terminal_biologically_ready_group_count_mean": (
                        starving_viability[
                            "terminal_biologically_ready_group_count"
                        ]
                    ),
                    "terminal_ready_group_count_mean": starving_viability[
                        "terminal_ready_group_count"
                    ],
                    "terminal_reproduction_viability_run_share": 0.0,
                    "birth_positive_run_share": 0.0,
                }
            )
        )
        viable_score = _score_components_total(
            _score_components(
                {
                    **base,
                    "terminal_energy_viability_share_mean": viable_viability[
                        "terminal_energy_viability_share"
                    ],
                    "terminal_energy_requirement_satisfaction_mean": (
                        viable_viability[
                            "terminal_energy_requirement_satisfaction"
                        ]
                    ),
                    "terminal_hydration_viability_share_mean": viable_viability[
                        "terminal_hydration_viability_share"
                    ],
                    "terminal_health_viability_share_mean": viable_viability[
                        "terminal_health_viability_share"
                    ],
                    "terminal_matched_diet_viability_share_mean": viable_viability[
                        "terminal_matched_diet_viability_share"
                    ],
                    "reproduction_ready_agents_mean": viable_viability[
                        "reproduction_ready_agents"
                    ],
                    "biologically_reproduction_ready_agents_mean": viable_viability[
                        "biologically_reproduction_ready_agents"
                    ],
                    "terminal_biologically_ready_group_count_mean": (
                        viable_viability[
                            "terminal_biologically_ready_group_count"
                        ]
                    ),
                    "terminal_ready_group_count_mean": viable_viability[
                        "terminal_ready_group_count"
                    ],
                    "terminal_reproduction_viability_run_share": 1.0,
                    "birth_positive_run_share": 0.0,
                }
            )
        )

        self.assertEqual(starving_viability["terminal_energy_viability_share"], 0.0)
        self.assertEqual(viable_viability["terminal_energy_viability_share"], 1.0)
        self.assertEqual(
            starving_viability["terminal_energy_requirement_satisfaction"],
            0.3333,
        )
        self.assertEqual(
            viable_viability["terminal_energy_requirement_satisfaction"],
            1.0,
        )
        self.assertEqual(viable_viability["terminal_hydration_viability_share"], 1.0)
        self.assertEqual(viable_viability["terminal_ready_group_count"], 1)
        self.assertGreater(viable_score, starving_score)

    def test_mind_v3_reproduction_viability_uses_real_energy_requirements(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import _run_reproduction_viability

        viability = _run_reproduction_viability(
            {
                "alive_agents": 3,
                "reproduction_end": {
                    "ready_agents": 0,
                    "biologically_ready_agents": 0,
                    "biological_blocker_counts": {
                        "energy": 2,
                        "hydration": 0,
                        "health": 0,
                        "matched_diet": 0,
                    },
                    "energy_readiness_by_meat_mode": {
                        "none": {
                            "alive_agents": 1,
                            "energy_shortfall_agents": 0,
                            "energy_total": 4.0,
                            "energy_required_total": 3.0,
                            "energy_gap_total": 0.0,
                        },
                        "scavenger": {
                            "alive_agents": 2,
                            "energy_shortfall_agents": 2,
                            "energy_total": 4.0,
                            "energy_required_total": 8.0,
                            "energy_gap_total": 4.0,
                        },
                    },
                },
            }
        )

        self.assertEqual(viability["terminal_energy_viability_share"], 0.3333)
        self.assertEqual(
            viability["terminal_energy_requirement_satisfaction"],
            0.6364,
        )
        self.assertEqual(viability["terminal_energy_shortfall_share"], 0.6667)
        self.assertEqual(viability["terminal_energy_required_total"], 11.0)
        self.assertEqual(viability["terminal_energy_gap_total"], 4.0)

    def test_mind_v3_evolve_score_penalizes_alive_reproduction_dead_end(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        alive_dead_end = {
            "alive_agents_mean": 4.0,
            "births_mean": 0.0,
            "deaths_mean": 16.0,
            "alive_agent_ticks_per_tick_mean": 8.0,
            "resource_event_rate": 0.3,
            "movement_event_rate": 0.05,
            "unique_requested_actions_mean": 5.0,
            "terminal_energy_viability_share_mean": 0.0,
            "terminal_hydration_viability_share_mean": 0.5,
            "terminal_health_viability_share_mean": 0.5,
            "terminal_matched_diet_viability_share_mean": 0.0,
        }
        lower_alive_viable = {
            **alive_dead_end,
            "alive_agents_mean": 2.0,
            "deaths_mean": 18.0,
            "terminal_energy_viability_share_mean": 0.5,
            "terminal_hydration_viability_share_mean": 0.5,
            "terminal_health_viability_share_mean": 0.75,
            "terminal_matched_diet_viability_share_mean": 0.5,
            "terminal_reproduction_viability_run_share": 1.0,
        }

        dead_end_components = _score_components(alive_dead_end)
        viable_components = _score_components(lower_alive_viable)

        self.assertLess(
            dead_end_components["terminal_reproduction_dead_end"],
            0.0,
        )
        self.assertGreater(
            viable_components["terminal_reproduction_viability"],
            0.0,
        )
        self.assertGreater(
            _score_components_total(viable_components),
            _score_components_total(dead_end_components),
        )

    def test_mind_v3_evolve_score_penalizes_seed_brittle_births(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        brittle_birth = {
            "alive_agents_mean": 1.5,
            "births_mean": 0.5,
            "deaths_mean": 19.0,
            "alive_agent_ticks_per_tick_mean": 12.0,
            "resource_event_rate": 0.7,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 7.0,
            "active_behavior_niche_count": 5,
            "terminal_energy_viability_share_mean": 0.25,
            "terminal_hydration_viability_share_mean": 0.25,
            "terminal_health_viability_share_mean": 0.75,
            "terminal_matched_diet_viability_share_mean": 1.0,
            "terminal_reproduction_viability_run_share": 0.5,
            "birth_positive_run_share": 0.5,
        }
        robust_viability = {
            **brittle_birth,
            "alive_agents_mean": 2.0,
            "births_mean": 0.0,
            "deaths_mean": 18.0,
            "terminal_matched_diet_viability_share_mean": 0.25,
            "terminal_reproduction_viability_run_share": 1.0,
            "birth_positive_run_share": 0.0,
        }

        brittle_components = _score_components(brittle_birth)
        robust_components = _score_components(robust_viability)

        self.assertLess(brittle_components["seed_brittle_birth"], 0.0)
        self.assertLess(
            brittle_components["terminal_reproduction_brittleness"],
            0.0,
        )
        self.assertGreater(
            robust_components["terminal_reproduction_viability_coverage"],
            brittle_components["terminal_reproduction_viability_coverage"],
        )
        self.assertGreater(
            _score_components_total(robust_components),
            _score_components_total(brittle_components),
        )

    def test_mind_v3_sustained_readiness_metrics_use_observed_outcomes(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _run_sustained_readiness_metrics,
        )

        metrics = _run_sustained_readiness_metrics(
            [
                {
                    "tick": 0,
                    "agent_id": 1,
                    "after": {
                        "alive": True,
                        "energy_ratio": 0.9,
                        "hydration_ratio": 0.8,
                        "health_ratio": 0.8,
                    },
                    "outcome": {"reproduction_ready_after": True},
                },
                {
                    "tick": 0,
                    "agent_id": 2,
                    "after": {
                        "alive": True,
                        "energy_ratio": 0.9,
                        "hydration_ratio": 0.8,
                        "health_ratio": 0.8,
                    },
                    "outcome": {"reproduction_ready_after": True},
                },
                {
                    "tick": 1,
                    "agent_id": 1,
                    "after": {
                        "alive": True,
                        "energy_ratio": 0.36,
                        "hydration_ratio": 0.72,
                        "health_ratio": 0.72,
                    },
                    "outcome": {"reproduction_ready_after": False},
                },
                {
                    "tick": 2,
                    "agent_id": 3,
                    "after": {"alive": False},
                    "outcome": {"reproduction_ready_after": True},
                },
            ],
            ticks_executed=4,
        )

        self.assertEqual(metrics["reproduction_ready_agent_tick_count"], 2)
        self.assertEqual(metrics["reproduction_ready_distinct_agent_count"], 2)
        self.assertEqual(metrics["reproduction_ready_tick_count"], 1)
        self.assertEqual(metrics["reproduction_ready_pair_tick_count"], 1)
        self.assertEqual(metrics["reproduction_ready_agent_tick_share"], 0.5)
        self.assertEqual(metrics["reproduction_ready_pair_tick_share"], 0.25)
        self.assertEqual(metrics["core_ready_agent_tick_count"], 2)
        self.assertEqual(metrics["core_ready_pair_tick_count"], 1)
        self.assertEqual(metrics["core_blocker_agent_tick_counts"]["energy"], 1)
        self.assertEqual(metrics["core_blocker_agent_tick_shares"]["energy"], 0.3333)
        self.assertEqual(
            metrics["primary_core_blocker_agent_tick_counts"]["energy"],
            1,
        )
        self.assertEqual(metrics["primary_core_blocker"], "energy")
        self.assertGreater(metrics["core_readiness_agent_tick_mean"], 0.8)

    def test_mind_v3_evolve_score_prefers_sustained_ready_pairs(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        transient_birth = {
            "alive_agents_mean": 4.0,
            "births_mean": 2.0,
            "deaths_mean": 12.0,
            "alive_agent_ticks_per_tick_mean": 10.0,
            "resource_event_rate": 0.3,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 5.0,
            "terminal_energy_viability_share_mean": 0.5,
            "terminal_hydration_viability_share_mean": 0.5,
            "terminal_health_viability_share_mean": 0.5,
            "terminal_matched_diet_viability_share_mean": 0.5,
            "terminal_reproduction_viability_run_share": 1.0,
            "birth_positive_run_share": 1.0,
            "reproduction_ready_agent_tick_share_mean": 0.0,
            "reproduction_ready_tick_share_mean": 0.0,
            "reproduction_ready_pair_tick_share_mean": 0.0,
        }
        sustained_ready = {
            **transient_birth,
            "births_mean": 1.0,
            "reproduction_ready_agent_tick_share_mean": 0.2,
            "reproduction_ready_tick_share_mean": 0.35,
            "reproduction_ready_pair_tick_share_mean": 0.1,
        }

        transient_components = _score_components(transient_birth)
        sustained_components = _score_components(sustained_ready)

        self.assertLess(transient_components["sustained_ready_dead_end"], 0.0)
        self.assertLess(
            transient_components["sustained_ready_pair_dead_end"],
            0.0,
        )
        self.assertGreater(
            sustained_components["sustained_reproduction_readiness"],
            transient_components["sustained_reproduction_readiness"],
        )
        self.assertGreater(
            _score_components_total(sustained_components),
            _score_components_total(transient_components),
        )

    def test_mind_v3_evolve_score_prefers_real_energy_satisfaction(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        low_real_energy = {
            "alive_agents_mean": 6.0,
            "births_mean": 1.0,
            "deaths_mean": 10.0,
            "alive_agent_ticks_per_tick_mean": 10.0,
            "resource_event_rate": 0.3,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 5.0,
            "terminal_energy_viability_share_mean": 0.5,
            "terminal_energy_requirement_satisfaction_mean": 0.2,
            "terminal_hydration_viability_share_mean": 0.8,
            "terminal_health_viability_share_mean": 0.8,
            "terminal_matched_diet_viability_share_mean": 0.8,
            "terminal_reproduction_viability_run_share": 1.0,
            "birth_positive_run_share": 1.0,
        }
        higher_real_energy = {
            **low_real_energy,
            "terminal_energy_requirement_satisfaction_mean": 0.8,
        }

        low_components = _score_components(low_real_energy)
        high_components = _score_components(higher_real_energy)

        self.assertGreater(
            high_components["terminal_energy_requirement_satisfaction"],
            low_components["terminal_energy_requirement_satisfaction"],
        )
        self.assertGreater(
            _score_components_total(high_components),
            _score_components_total(low_components),
        )

    def test_mind_v3_evolve_score_credits_observed_animal_resource_use(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        plant_only = {
            "alive_agents_mean": 6.0,
            "births_mean": 1.0,
            "deaths_mean": 10.0,
            "alive_agent_ticks_per_tick_mean": 10.0,
            "resource_event_rate": 0.4,
            "animal_resource_event_rate": 0.0,
            "carcass_event_rate": 0.0,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 5.0,
            "terminal_energy_viability_share_mean": 0.5,
            "terminal_energy_requirement_satisfaction_mean": 0.5,
            "terminal_hydration_viability_share_mean": 0.5,
            "terminal_health_viability_share_mean": 0.5,
            "terminal_matched_diet_viability_share_mean": 0.5,
            "terminal_reproduction_viability_run_share": 1.0,
            "birth_positive_run_share": 1.0,
        }
        animal_resource_user = {
            **plant_only,
            "animal_resource_event_rate": 0.2,
            "carcass_event_rate": 0.15,
        }

        plant_components = _score_components(plant_only)
        animal_components = _score_components(animal_resource_user)

        self.assertGreater(
            animal_components["animal_resource_events"],
            plant_components["animal_resource_events"],
        )
        self.assertGreater(
            animal_components["carrion_resource_events"],
            plant_components["carrion_resource_events"],
        )
        self.assertGreater(
            _score_components_total(animal_components),
            _score_components_total(plant_components),
        )

    def test_mind_v3_evolve_score_prefers_balanced_energy_hydration(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        energy_overcorrected = {
            "alive_agents_mean": 6.0,
            "births_mean": 2.0,
            "deaths_mean": 10.0,
            "alive_agent_ticks_per_tick_mean": 10.0,
            "resource_event_rate": 0.3,
            "movement_event_rate": 0.1,
            "unique_requested_actions_mean": 5.0,
            "terminal_energy_viability_share_mean": 0.8,
            "terminal_energy_requirement_satisfaction_mean": 0.9,
            "terminal_hydration_viability_share_mean": 0.25,
            "terminal_health_viability_share_mean": 0.8,
            "terminal_matched_diet_viability_share_mean": 0.8,
            "terminal_reproduction_viability_run_share": 1.0,
            "birth_positive_run_share": 1.0,
        }
        balanced = {
            **energy_overcorrected,
            "terminal_energy_requirement_satisfaction_mean": 0.68,
            "terminal_hydration_viability_share_mean": 0.68,
        }

        energy_components = _score_components(energy_overcorrected)
        balanced_components = _score_components(balanced)

        self.assertGreater(
            balanced_components["terminal_energy_hydration_balance"],
            energy_components["terminal_energy_hydration_balance"],
        )
        self.assertGreater(
            balanced_components["terminal_balanced_reproduction_readiness"],
            energy_components["terminal_balanced_reproduction_readiness"],
        )
        self.assertGreater(
            _score_components_total(balanced_components),
            _score_components_total(energy_components),
        )

    def test_mind_v3_evolve_score_penalizes_dominant_action_collapse(
        self,
    ) -> None:
        from evolution_sim.cli.mind_v3_evolve import (
            _score_components,
            _score_components_total,
        )

        collapsed = {
            "alive_agents_mean": 4.0,
            "births_mean": 0.0,
            "deaths_mean": 16.0,
            "alive_agent_ticks_per_tick_mean": 8.0,
            "resource_event_rate": 0.6,
            "movement_event_rate": 0.05,
            "unique_requested_actions_mean": 4.0,
            "dominant_requested_action_share": 1.0,
            "active_behavior_niche_count": 1,
        }
        diverse = {
            **collapsed,
            "dominant_requested_action_share": 0.55,
            "active_behavior_niche_count": 5,
        }

        collapsed_components = _score_components(collapsed)
        diverse_components = _score_components(diverse)

        self.assertLess(collapsed_components["dominant_action_collapse"], 0.0)
        self.assertEqual(diverse_components["dominant_action_collapse"], 0.0)
        self.assertGreater(
            diverse_components["active_behavior_niches"],
            collapsed_components["active_behavior_niches"],
        )
        self.assertGreater(
            _score_components_total(diverse_components),
            _score_components_total(collapsed_components),
        )

    def test_mind_v3_evaluate_cli_accepts_evolved_founder_template(
        self,
    ) -> None:
        from random import Random

        from evolution_sim.cli import mind_v3_evaluate
        from evolution_sim.mind.evolution import founder_mind_v3_metadata

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_path = tmp_path / "template-report.json"
            report_path = tmp_path / "mind-v3-eval.json"
            template_path.write_text(
                json.dumps(
                    {
                        "schema_version": "mind_v3_evolution_search_v1",
                        "best_candidate": {
                            "controller_metadata": founder_mind_v3_metadata(
                                agent_id=1,
                                rng=Random(3),
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_v3_evaluate",
                        "--seeds",
                        "5",
                        "--ticks",
                        "20",
                        "--founder-template",
                        str(template_path),
                        "--output",
                        str(report_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_v3_evaluate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(
            report["policy"]["founder_template_source"],
            str(template_path),
        )
        self.assertEqual(
            report["comparison"]["mind_v3"]["aggregate"][
                "heuristic_action_source_count"
            ],
            0,
        )

    def test_mind_policy_eval_cli_compares_runtime_modes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "trajectory.jsonl.gz"
            artifact_path = tmp_path / "mind-artifact.json"
            report_path = tmp_path / "mind-eval-report.json"
            ledger_path = tmp_path / "mind-eval-ledger.jsonl"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_policy_eval",
                        "--artifact",
                        str(artifact_path),
                        "--enable-mind",
                        "--seed",
                        "7",
                        "--ticks",
                        "2",
                        "--compare-runtime-mode",
                        "autonomous-online",
                        "--output",
                        str(report_path),
                        "--experiment-ledger-output",
                        str(ledger_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_policy_eval.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(
            report["protocol"]["runtime_mode"],
            "guarded",
        )
        self.assertEqual(len(report["runtime_mode_evaluation_matrix"]), 1)
        self.assertEqual(
            report["runtime_mode_evaluation_matrix"][0]["runtime_mode"],
            "autonomous-online",
        )
        self.assertIn(
            "policy_update_trace",
            report["runtime_mode_evaluation_matrix"][0]["evaluation"]["learned"][
                "aggregate"
            ],
        )
        self.assertEqual(ledger_rows, [report["experiment_ledger_entry"]])
        self.assertEqual(
            report["experiment_ledger_entry"]["runtime_mode_comparisons"][0][
                "runtime_mode"
            ],
            "autonomous-online",
        )
        self.assertIn(
            "alive_delta",
            report["experiment_ledger_entry"]["runtime_mode_comparisons"][0],
        )
        self.assertIn(
            "births_delta",
            report["experiment_ledger_entry"]["runtime_mode_comparisons"][0],
        )

    def test_mind_policy_eval_online_primary_uses_fresh_policy_factory(self) -> None:
        class StubPolicy:
            policy_id = "stub_mind_policy"

        compare_calls: list[dict[str, object]] = []

        def compare_stub(**kwargs: object) -> dict[str, object]:
            compare_calls.append(dict(kwargs))
            return {
                "protocol": {},
                "learned": {
                    "aggregate": {
                        "policy_diagnostics": {
                            "guard_intervention_rate": 0.0,
                            "heuristic_delegate_rate": 0.0,
                        }
                    }
                },
                "comparison": {
                    "alive_agents_mean_delta": 0.0,
                    "births_mean_delta": 0.0,
                },
                "mind_v1_gates": {
                    "status": "pass",
                    "blockers": [],
                    "warnings": [],
                },
            }

        with TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "mind-eval-report.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "mind_policy_eval",
                        "--artifact",
                        str(Path(tmpdir) / "mind-artifact.json"),
                        "--enable-mind",
                        "--seed",
                        "7",
                        "--ticks",
                        "2",
                        "--mind-runtime-mode",
                        "autonomous-online",
                        "--output",
                        str(report_path),
                    ],
                ),
                patch(
                    "evolution_sim.cli.mind_policy_eval.load_learned_policy",
                    return_value=StubPolicy(),
                ) as load_policy,
                patch(
                    "evolution_sim.cli.mind_policy_eval.compare_heuristic_and_learned",
                    side_effect=compare_stub,
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_policy_eval.main()

        self.assertEqual(len(compare_calls), 1)
        self.assertNotIn("learned_policy", compare_calls[0])
        self.assertIn("learned_policy_factory", compare_calls[0])
        self.assertEqual(compare_calls[0]["learned_policy_name"], "stub_mind_policy")
        load_policy.assert_called_once()

    def test_load_learned_policy_autonomous_runtime_mode_disables_heuristic_fallback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_path = tmp_path / "trajectory.jsonl.gz"
            artifact_path = tmp_path / "mind-artifact.json"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            guarded = load_learned_policy(artifact_path, enable_mind=True)
            autonomous = load_learned_policy(
                artifact_path,
                enable_mind=True,
                runtime_mode="autonomous",
            )

        self.assertTrue(guarded.heuristic_guard)
        self.assertTrue(guarded.heuristic_delegate)
        self.assertFalse(autonomous.heuristic_guard)
        self.assertFalse(autonomous.heuristic_delegate)
        self.assertIn("heuristic_free_autonomous_controller_v1", autonomous.policy_version)

    def test_online_adaptive_mind_policy_updates_action_scores_from_reward(self) -> None:
        base_policy = LearnedPolicy(
            action_scores={
                action: (0.1 if action == "eat" else 0.0)
                for action in ACTION_NAMES
            },
            heuristic_guard=False,
            heuristic_delegate=False,
            policy_version="test_policy",
        )
        policy = OnlineAdaptiveMindPolicy(
            base_policy=base_policy,
            learning_rate=0.5,
        )
        action_mask = {action: action in {"eat", "stay"} for action in ACTION_NAMES}

        first_decision = policy.decide({}, action_mask)
        self.assertEqual(first_decision.requested_action, "eat")
        policy.observe_transition(
            {
                "resolved_action": "eat",
                "reward": {"total": -1.0},
                "policy_decision_diagnostics": first_decision.diagnostics,
            }
        )
        second_decision = policy.decide({}, action_mask)

        self.assertEqual(policy.online_update_count, 1)
        self.assertEqual(second_decision.requested_action, "stay")
        self.assertIn(
            "in_run_contextual_bandit_adapter_v1",
            second_decision.source,
        )

    def test_online_adaptive_mind_policy_returns_replayable_update_trace(self) -> None:
        base_policy = LearnedPolicy(
            action_scores={
                action: (0.1 if action == "eat" else 0.0)
                for action in ACTION_NAMES
            },
            heuristic_guard=False,
            heuristic_delegate=False,
        )
        policy = OnlineAdaptiveMindPolicy(
            base_policy=base_policy,
            learning_rate=0.5,
        )
        action_mask = {action: action in {"eat", "stay"} for action in ACTION_NAMES}
        decision = policy.decide({}, action_mask)

        trace = policy.observe_transition(
            {
                "resolved_action": "eat",
                "reward": {"total": -1.0},
                "policy_decision_diagnostics": decision.diagnostics,
            }
        )

        self.assertEqual(trace["schema_version"], "mind_policy_update_trace_v1")
        self.assertEqual(trace["policy"], "in_run_contextual_bandit_adapter_v1")
        self.assertEqual(trace["context_key"], "global")
        self.assertEqual(trace["action"], "eat")
        self.assertEqual(trace["previous_adjustment"], 0.0)
        self.assertEqual(trace["updated_adjustment"], -0.5)
        self.assertEqual(replay_online_update_traces([trace]), {("global", "eat"): -0.5})

    def test_replay_online_update_traces_rejects_invalid_adjustment_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "adjustment bounds"):
            replay_online_update_traces(
                [
                    {
                        "schema_version": "mind_policy_update_trace_v1",
                        "policy": "in_run_contextual_bandit_adapter_v1",
                        "update_index": 1,
                        "context_key": "global",
                        "action": "eat",
                        "reward_total": 1.0,
                        "reward_signal": 1.0,
                        "previous_adjustment": 0.0,
                        "updated_adjustment": 0.0,
                        "learning_rate": 0.5,
                        "min_adjustment": 1.0,
                        "max_adjustment": -1.0,
                    }
                ]
            )

    def test_torch_online_feedback_ignores_mind_v3_update_traces(self) -> None:
        class Transition:
            action = "eat"
            policy_update_trace = {
                "schema_version": "mind_v3_controller_update_trace_v1",
                "policy": "bounded_reward_modulated_controller_update_v1",
                "credit_assignment": (
                    "policy_valid_requested_action_horizon_eligibility_trace_v3"
                ),
                "update_index": 1,
                "agent_id": 1,
                "action": "eat",
                "requested_action": "eat",
                "resolved_action": "eat",
                "credited_actions": ["eat"],
                "reward_total": 1.0,
                "reward_signal": 1.0,
                "heuristic_free": True,
            }

        self.assertEqual(
            mind_torch_trainer._online_update_feedback_record(
                Transition(),
                action_index={action: index for index, action in enumerate(ACTION_NAMES)},
            ),
            (-1, 0.0),
        )

    def test_online_adaptive_mind_policy_ignores_non_policy_feedback(self) -> None:
        base_policy = LearnedPolicy(
            action_scores={action: 0.0 for action in ACTION_NAMES},
            heuristic_guard=False,
            heuristic_delegate=False,
        )
        policy = OnlineAdaptiveMindPolicy(base_policy=base_policy)

        policy.observe_transition(
            {
                "action_source": "passive",
                "resolved_action": "stay",
                "reward": {"total": -1.0},
            }
        )

        self.assertEqual(policy.online_update_count, 0)
        self.assertEqual(policy.action_adjustments, {})

    def test_world_feeds_finalized_transition_records_to_online_policy(self) -> None:
        class FeedbackPolicy:
            policy_id = "test_online_feedback_policy"
            policy_version = "test_online_feedback_policy_v1"

            def __init__(self) -> None:
                self.records: list[dict[str, object]] = []

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                action = "stay" if action_mask.get("stay", False) else next(
                    action for action, available in action_mask.items() if available
                )
                return ActionDecision(
                    requested_action=action,
                    source=self.policy_id,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    diagnostics={
                        "learned_action": action,
                        "online_context_key": "test_context",
                    },
                )

            def observe_transition(self, record: dict[str, object]) -> None:
                self.records.append(record)

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "online-feedback.jsonl.gz"
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="online_feedback",
            )
            policy = FeedbackPolicy()
            SimulationWorld(
                WorldConfig(seed=7, max_ticks=2),
                policy=policy,
            ).run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )

        self.assertGreater(len(policy.records), 0)
        first_record = policy.records[0]
        self.assertIn("reward", first_record)
        self.assertEqual(
            first_record["policy_decision_diagnostics"]["online_context_key"],
            "test_context",
        )

    def test_collect_trajectory_cli_writes_learned_policy_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            training_path = tmp_path / "training.jsonl.gz"
            artifact_path = tmp_path / "mind-artifact.json"
            learned_path = tmp_path / "learned.jsonl.gz"
            self._write_tiny_trajectory(training_path)
            dataset = load_trajectory_jsonl(training_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )
            write_model_artifact(artifact_path, baseline.to_artifact())

            stdout = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "collect_trajectory",
                        "--seed",
                        "8",
                        "--ticks",
                        "2",
                        "--output",
                        str(learned_path),
                        "--split-id",
                        "learned-online-probe",
                        "--mind-artifact",
                        str(artifact_path),
                        "--enable-mind",
                    ],
                ),
                patch("sys.stdout", stdout),
                patch("sys.stderr", io.StringIO()),
            ):
                collect_trajectory.main()

            learned_dataset = load_trajectory_jsonl(learned_path)

        self.assertIn("mind_policy=mind_v1_learned_policy", stdout.getvalue())
        self.assertEqual(
            learned_dataset.header["provenance"]["split_id"],
            "learned-online-probe",
        )
        self.assertTrue(
            any(
                record["policy_id"] == "mind_v1_learned_policy"
                for record in learned_dataset.records
            )
        )

    def test_collect_trajectory_cli_requires_explicit_mind_enable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "mind-artifact.json"
            with (
                patch(
                    "sys.argv",
                    [
                        "collect_trajectory",
                        "--mind-artifact",
                        str(artifact_path),
                    ],
                ),
                self.assertRaisesRegex(SystemExit, "--mind-artifact requires --enable-mind"),
            ):
                collect_trajectory.main()

    def test_behavior_cloning_artifact_diagnostics_report_imitation_and_coverage(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)
            baseline = train_behavior_cloning_baseline(
                dataset.records,
                provenance=dataset_provenance(dataset),
            )

            artifact = baseline.to_artifact()
            diagnostics = build_artifact_diagnostics(artifact, [dataset])

        self.assertIn("action_score_metadata", artifact["model"])
        self.assertIn("conditional_action_metadata", artifact["model"])
        self.assertEqual(
            artifact["model"]["action_score_metadata"]["record_count"],
            dataset.record_count,
        )
        self.assertEqual(diagnostics["record_count"], dataset.record_count)
        self.assertGreaterEqual(diagnostics["imitation"]["top1_accuracy"], 0.0)
        self.assertLessEqual(diagnostics["imitation"]["top1_accuracy"], 1.0)
        self.assertEqual(
            sum(diagnostics["action_distribution"]["label_counts"].values()),
            dataset.record_count,
        )
        self.assertEqual(
            sum(diagnostics["action_distribution"]["predicted_counts"].values()),
            dataset.record_count,
        )
        self.assertGreaterEqual(
            diagnostics["action_distribution"]["prediction_label_tvd"],
            0.0,
        )
        self.assertLessEqual(
            diagnostics["action_distribution"]["prediction_label_tvd"],
            1.0,
        )
        confusion = diagnostics["action_distribution"]["confusion"]
        self.assertIn("matrix", confusion)
        self.assertIn("top_misclassifications", confusion)
        self.assertIn("per_action", confusion)
        self.assertEqual(
            sum(
                sum(row.values())
                for row in confusion["matrix"].values()
            ),
            dataset.record_count,
        )
        self.assertIn("eat", confusion["per_action"])
        self.assertIn("precision", confusion["per_action"]["eat"])
        self.assertIn("recall", confusion["per_action"]["eat"])
        self.assertIn("match_depth_counts", diagnostics["contextual_coverage"])
        self.assertEqual(
            sum(diagnostics["contextual_coverage"]["match_depth_counts"].values()),
            dataset.record_count,
        )
        self.assertIn("support_bucket_counts", diagnostics["contextual_coverage"])
        self.assertIn(
            "score_margin_bucket_counts",
            diagnostics["contextual_coverage"],
        )
        self.assertIn("reward_calibration", diagnostics)
        self.assertIn(
            "by_predicted_action",
            diagnostics["reward_calibration"],
        )
        self.assertIn(
            "by_label_action",
            diagnostics["reward_calibration"],
        )
        self.assertIn(
            "by_score_margin_bucket",
            diagnostics["reward_calibration"],
        )
        self.assertIn(
            "eat",
            diagnostics["reward_calibration"]["by_label_action"],
        )
        self.assertIn(
            "mean_reward",
            diagnostics["reward_calibration"]["by_label_action"]["eat"],
        )
        self.assertGreaterEqual(
            diagnostics["contextual_coverage"]["matched_record_rate"],
            0.0,
        )

    def test_mind_gate_cli_writes_seed_bank_report_and_artifact(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"
            ledger_path = tmp_path / "mind-experiment-ledger.jsonl"
            trajectory_dir = tmp_path / "trajectories"
            extra_trajectory_path = tmp_path / "learned-extra.jsonl.gz"
            self._write_tiny_trajectory(extra_trajectory_path, seed=9)

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--trajectory-dir",
                        str(trajectory_dir),
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--experiment-ledger-output",
                        str(ledger_path),
                        "--extra-trajectory",
                        str(extra_trajectory_path),
                        "--compare-runtime-mode",
                        "autonomous-online",
                        "--max-alive-agents-mean-regression",
                        "0.5",
                        "--max-births-mean-regression",
                        "0.25",
                        "--max-invalid-action-rate",
                        "0.03",
                        "--min-viable-run-share",
                        "0.5",
                        "--min-births-per-run-mean",
                        "0.0",
                        "--min-plant-energy-available-per-land-tile",
                        "0.0",
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_gate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)
            self.assertTrue(report["complete"])
            self.assertEqual(report["protocol"]["train_seeds"], [7])
            self.assertEqual(
                report["protocol"]["extra_trajectory_paths"],
                [str(extra_trajectory_path)],
            )
            self.assertEqual(report["protocol"]["validation_seeds"], [8])
            self.assertEqual(
                report["protocol"]["criteria"],
                {
                    "max_alive_agents_mean_regression": 0.5,
                    "max_alive_agents_per_seed_regression": 2.0,
                    "max_births_mean_regression": 0.25,
                    "max_births_per_seed_regression": 1.0,
                    "max_invalid_action_rate": 0.03,
                    "max_guard_intervention_rate": 0.45,
                    "max_total_heuristic_fallback_rate": 1.0,
                    "max_guard_intervention_rate_by_group": 0.5,
                    "min_guard_intervention_rate_reduction": 0.0,
                    "min_viable_run_share": 0.5,
                    "min_births_per_run_mean": 0.0,
                    "min_plant_energy_available_per_land_tile": 0.0,
                },
            )
            self.assertEqual(
                report["evaluation"]["protocol"]["gate_criteria"],
                report["protocol"]["criteria"],
            )
            self.assertEqual(report["artifact"]["path"], str(artifact_path))
            self.assertEqual(
                report["experiment_ledger_entry"]["trainer"],
                "contextual-prior",
            )
            self.assertEqual(
                report["experiment_ledger_entry"]["report_path"],
                str(report_path),
            )
            self.assertIn(
                "strict_control_target_passed",
                report["experiment_ledger_entry"],
            )
            self.assertIn("min_alive_delta", report["experiment_ledger_entry"])
            self.assertIn("min_births_delta", report["experiment_ledger_entry"])
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(ledger_rows, [report["experiment_ledger_entry"]])
            self.assertGreater(report["artifact"]["trained_record_count"], 0)
            self.assertEqual(
                report["artifact"]["trained_record_count"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertEqual(report["artifact"]["conditional_min_records"], 3)
            self.assertEqual(
                report["artifact"]["conditional_score_policy"],
                artifact["model"]["conditional_score_policy"],
            )
            self.assertEqual(
                report["artifact"]["conditional_prior_correction_exponent"],
                artifact["model"]["conditional_prior_correction_exponent"],
            )
            self.assertEqual(
                report["artifact"]["conditional_score_smoothing_alpha"],
                artifact["model"]["conditional_score_smoothing_alpha"],
            )
            self.assertEqual(report["protocol"]["trainer"], "contextual-prior")
            self.assertEqual(
                report["protocol"]["runtime_mode_comparisons"],
                ["autonomous-online"],
            )
            self.assertEqual(report["artifact"]["trainer"], "contextual-prior")
            self.assertEqual(
                report["artifact"]["sample_weight_policy"],
                artifact["model"]["sample_weight_policy"],
            )
            self.assertEqual(
                report["artifact"]["sample_weight_total"],
                artifact["model"]["sample_weight_total"],
            )
            self.assertEqual(
                report["artifact"]["reward_advantage_blend_policy"],
                artifact["model"].get("reward_advantage_blend_policy"),
            )
            self.assertEqual(
                report["artifact"]["reward_advantage_blend_weight"],
                artifact["model"].get("reward_advantage_blend_weight"),
            )
            self.assertEqual(
                report["artifact"]["value_estimation_policy"],
                artifact["model"].get("value_estimation_policy"),
            )
            self.assertEqual(
                report["artifact"]["value_score_blend_policy"],
                artifact["model"].get("value_score_blend_policy"),
            )
            self.assertEqual(
                report["artifact"]["value_score_blend_weight"],
                artifact["model"].get("value_score_blend_weight"),
            )
            self.assertEqual(
                report["artifact"]["value_supported_deviation_policy"],
                artifact["model"].get("value_supported_deviation_policy"),
            )
            self.assertEqual(
                report["artifact"]["value_supported_deviation_min_support"],
                artifact["model"].get("value_supported_deviation_min_support"),
            )
            self.assertEqual(
                report["artifact"]["heuristic_confidence_threshold"],
                artifact["model"]["heuristic_confidence_threshold"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_override_min_margin"],
                artifact["model"]["heuristic_override_min_margin"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_delegate_policy"],
                artifact["model"]["heuristic_delegate_policy"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_delegate_max_training_score_margin"],
                artifact["model"]["heuristic_delegate_max_training_score_margin"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_safe_local_eat_min_score"],
                artifact["model"]["heuristic_safe_local_eat_min_score"],
            )
            self.assertEqual(
                report["artifact"]["heuristic_safe_plant_move_min_score"],
                artifact["model"]["heuristic_safe_plant_move_min_score"],
            )
            self.assertEqual(
                report["artifact_diagnostics"]["train"]["record_count"],
                report["artifact"]["trained_record_count"],
            )
            self.assertGreater(
                report["artifact_diagnostics"]["held_out"]["record_count"],
                0,
            )
            self.assertIn("imitation", report["artifact_diagnostics"]["train"])
            self.assertIn("imitation", report["artifact_diagnostics"]["held_out"])
            self.assertIn(
                "contextual_coverage",
                report["artifact_diagnostics"]["train"],
            )
            self.assertIn(report["readiness"]["status"], {"pass", "review", "fail"})
            self.assertEqual(len(report["trajectory_collection"]), 1)
            self.assertEqual(len(report["extra_trajectory_collection"]), 1)
            self.assertEqual(len(report["artifact_diagnostic_collection"]), 1)
            self.assertEqual(len(report["runtime_mode_evaluation_matrix"]), 1)
            self.assertEqual(
                report["runtime_mode_evaluation_matrix"][0]["runtime_mode"],
                "autonomous-online",
            )
            self.assertIn(
                "autonomous-online",
                report["runtime_mode_readiness"],
            )
            self.assertIn(
                "runtime_mode_comparisons",
                report["experiment_ledger_entry"],
            )
            self.assertIn(
                "alive_delta",
                report["experiment_ledger_entry"]["runtime_mode_comparisons"][0],
            )
            self.assertIn(
                "births_delta",
                report["experiment_ledger_entry"]["runtime_mode_comparisons"][0],
            )
            self.assertEqual(
                report["experiment_ledger_entry"]["extra_trajectory_paths"],
                [str(extra_trajectory_path)],
            )
            self.assertEqual(
                report["protocol"]["artifact_diagnostics_execution"],
                {
                    "policy": (
                        "mind_gate_artifact_diagnostics_sharded_process_pool_v2"
                    ),
                    "requested_workers": 1,
                    "effective_workers": 1,
                    "diagnostic_split_count": 2,
                    "diagnostic_task_count": 3,
                    "shard_policy": "trajectory_dataset_shards_v1",
                },
            )
            self.assertEqual(
                report["protocol"]["evaluation_execution"],
                {
                    "policy": "mind_gate_matrix_process_pool_evaluation_v1",
                    "requested_workers": 1,
                    "effective_workers": 1,
                    "evaluation_entry_count": 2,
                },
            )
            phase_timings = report["timings"]["phase_wall_seconds"]
            for phase in (
                "trajectory_prepare",
                "training_dataset_load",
                "training",
                "artifact_write",
                "artifact_diagnostics_prepare",
                "artifact_diagnostics",
                "evaluation",
            ):
                self.assertIn(phase, phase_timings)
                self.assertIsInstance(phase_timings[phase], float)

    def test_mind_gate_writes_artifact_before_artifact_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            artifact_path = tmp_path / "mind-gate-artifact.json"
            trajectory_dir = tmp_path / "trajectories"

            with (
                patch(
                    "evolution_sim.cli.mind_gate.build_artifact_diagnostics",
                    side_effect=RuntimeError("diagnostics boom"),
                ),
                self.assertRaisesRegex(RuntimeError, "diagnostics boom"),
            ):
                mind_gate.run_mind_gate(
                    train_seeds=[7],
                    validation_seeds=[8],
                    ticks=2,
                    validation_ticks=[2],
                    trajectory_dir=trajectory_dir,
                    artifact_output=artifact_path,
                    report_output=tmp_path / "report.json",
                    experiment_ledger_output=None,
                )

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)

    def test_mind_gate_can_skip_artifact_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"

            with patch(
                "evolution_sim.cli.mind_gate.build_artifact_diagnostics",
                side_effect=AssertionError("artifact diagnostics should be skipped"),
            ):
                report = mind_gate.run_mind_gate(
                    train_seeds=[7],
                    validation_seeds=[8],
                    ticks=2,
                    validation_ticks=[2],
                    trajectory_dir=tmp_path / "trajectories",
                    artifact_output=artifact_path,
                    report_output=report_path,
                    experiment_ledger_output=None,
                    skip_artifact_diagnostics=True,
                )

            self.assertTrue(artifact_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(report["protocol"]["artifact_diagnostics_skipped"])
            self.assertEqual(report["artifact_diagnostic_collection"], [])
            self.assertEqual(
                report["artifact_diagnostics"],
                {
                    "skipped": True,
                    "reason": "skip_artifact_diagnostics",
                },
            )
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))[
                    "artifact_diagnostics"
                ],
                report["artifact_diagnostics"],
            )
            self.assertEqual(
                report["protocol"]["evaluation_execution"],
                {
                    "policy": "mind_gate_matrix_process_pool_evaluation_v1",
                    "requested_workers": 1,
                    "effective_workers": 1,
                    "evaluation_entry_count": 1,
                },
            )
            self.assertEqual(
                report["protocol"]["artifact_diagnostics_execution"],
                {
                    "policy": (
                        "mind_gate_artifact_diagnostics_sharded_process_pool_v2"
                    ),
                    "requested_workers": 1,
                    "effective_workers": 0,
                    "diagnostic_split_count": 0,
                    "diagnostic_task_count": 0,
                    "shard_policy": "trajectory_dataset_shards_v1",
                },
            )

    def test_mind_gate_evaluation_execution_caps_workers_to_matrix_size(self) -> None:
        metadata = mind_gate._mind_gate_evaluation_execution_metadata(
            requested_workers=8,
            evaluation_entry_count=3,
        )

        self.assertEqual(
            metadata,
            {
                "policy": "mind_gate_matrix_process_pool_evaluation_v1",
                "requested_workers": 8,
                "effective_workers": 3,
                "evaluation_entry_count": 3,
            },
        )

    def test_mind_gate_artifact_diagnostics_execution_caps_workers(self) -> None:
        metadata = mind_gate._mind_gate_artifact_diagnostics_execution_metadata(
            requested_workers=8,
            diagnostic_split_count=2,
            diagnostic_task_count=6,
        )

        self.assertEqual(
            metadata,
            {
                "policy": "mind_gate_artifact_diagnostics_sharded_process_pool_v2",
                "requested_workers": 8,
                "effective_workers": 6,
                "diagnostic_split_count": 2,
                "diagnostic_task_count": 6,
                "shard_policy": "trajectory_dataset_shards_v1",
            },
        )

    def test_mind_gate_artifact_diagnostics_execution_has_no_workers_when_skipped(
        self,
    ) -> None:
        metadata = mind_gate._mind_gate_artifact_diagnostics_execution_metadata(
            requested_workers=8,
            diagnostic_split_count=0,
            diagnostic_task_count=0,
        )

        self.assertEqual(
            metadata,
            {
                "policy": "mind_gate_artifact_diagnostics_sharded_process_pool_v2",
                "requested_workers": 8,
                "effective_workers": 0,
                "diagnostic_split_count": 0,
                "diagnostic_task_count": 0,
                "shard_policy": "trajectory_dataset_shards_v1",
            },
        )

    def test_mind_gate_parallel_artifact_diagnostics_match_serial(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_dir = tmp_path / "trajectories"
            serial_report = mind_gate.run_mind_gate(
                train_seeds=[7, 9],
                validation_seeds=[8, 10],
                ticks=2,
                validation_ticks=[2],
                trajectory_dir=trajectory_dir,
                artifact_output=tmp_path / "serial-artifact.json",
                report_output=tmp_path / "serial-report.json",
                experiment_ledger_output=None,
                artifact_diagnostics_workers=1,
            )
            parallel_report = mind_gate.run_mind_gate(
                train_seeds=[7, 9],
                validation_seeds=[8, 10],
                ticks=2,
                validation_ticks=[2],
                trajectory_dir=trajectory_dir,
                artifact_output=tmp_path / "parallel-artifact.json",
                report_output=tmp_path / "parallel-report.json",
                experiment_ledger_output=None,
                reuse_trajectories=True,
                artifact_diagnostics_workers=4,
            )

        self.assertEqual(
            parallel_report["protocol"]["artifact_diagnostics_execution"],
            {
                "policy": "mind_gate_artifact_diagnostics_sharded_process_pool_v2",
                "requested_workers": 4,
                "effective_workers": 4,
                "diagnostic_split_count": 2,
                "diagnostic_task_count": 4,
                "shard_policy": "trajectory_dataset_shards_v1",
            },
        )
        self.assertEqual(
            parallel_report["artifact_diagnostics"],
            serial_report["artifact_diagnostics"],
        )

    def test_mind_gate_rejects_reused_trajectory_seed_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            trajectory_dir = tmp_path / "trajectories"
            trajectory_dir.mkdir()
            mislabeled_path = trajectory_dir / "seed8-ticks2.jsonl.gz"
            self._write_tiny_trajectory(mislabeled_path, seed=7)

            with self.assertRaisesRegex(ValueError, "seed 8"):
                mind_gate.run_mind_gate(
                    train_seeds=[8],
                    validation_seeds=[9],
                    ticks=2,
                    validation_ticks=[2],
                    trajectory_dir=trajectory_dir,
                    artifact_output=tmp_path / "artifact.json",
                    report_output=tmp_path / "report.json",
                    reuse_trajectories=True,
                )

    def test_mind_gate_rejects_unreplayable_extra_update_trace(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            extra_path = tmp_path / "bad-extra.jsonl.gz"
            self._write_tiny_trajectory(extra_path, seed=9)
            with gzip.open(extra_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            for row_index, previous_adjustment in ((1, 0.0), (2, 0.0)):
                rows[row_index]["record"]["policy_update_trace"] = {
                    "schema_version": "mind_policy_update_trace_v1",
                    "policy": "in_run_contextual_bandit_adapter_v1",
                    "update_index": row_index,
                    "context_key": "global",
                    "action": "eat",
                    "reward_total": 1.0,
                    "reward_signal": 1.0,
                    "previous_adjustment": previous_adjustment,
                    "updated_adjustment": 0.05,
                    "learning_rate": 0.05,
                    "min_adjustment": -0.5,
                    "max_adjustment": 0.5,
                }
            with gzip.open(extra_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            with self.assertRaisesRegex(ValueError, "previous_adjustment"):
                mind_gate.run_mind_gate(
                    train_seeds=[7],
                    validation_seeds=[8],
                    ticks=2,
                    validation_ticks=[2],
                    trajectory_dir=tmp_path / "trajectories",
                    artifact_output=tmp_path / "artifact.json",
                    report_output=tmp_path / "report.json",
                    extra_trajectories=[extra_path],
                )

    def test_mind_gate_default_validation_seeds_are_held_out_seed_bank(self) -> None:
        self.assertEqual(mind_gate.DEFAULT_VALIDATION_SEEDS, (5, 13, 19, 29))
        self.assertTrue(
            set(mind_gate.DEFAULT_VALIDATION_SEEDS).isdisjoint(
                mind_gate.DEFAULT_TRAIN_SEEDS
            )
        )

    def test_mind_extended_gate_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:gate:extended"]

        self.assertIn("evolution_sim.cli.mind_gate", script)
        self.assertIn("--validation-seeds 5,13,19,29,37,41", script)
        self.assertIn("--validation-ticks 120,180", script)
        self.assertIn("output/mind/mind-v1-gate-extended-report.json", script)

    def test_mind_diagnostics_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:diagnostics"]

        self.assertIn("evolution_sim.cli.mind_artifact_diagnostics", script)

    def test_mind_v3_evolution_search_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:v3:evolve"]

        self.assertIn("evolution_sim.cli.mind_v3_evolve", script)

    def test_mind_strict_promotion_gate_has_npm_entrypoint(self) -> None:
        package = json.loads(Path("package.json").read_text(encoding="utf-8"))
        script = package["scripts"]["sim:mind:gate:strict"]

        self.assertIn("evolution_sim.cli.mind_gate", script)
        self.assertIn("--reuse-trajectories", script)
        self.assertIn("--trainer advantage-blended-contextual-prior", script)
        self.assertIn("--validation-seeds 5,13,19,29,37,41", script)
        self.assertIn("--validation-ticks 120,180", script)
        self.assertIn("--max-alive-agents-per-seed-regression 0", script)
        self.assertIn("--max-births-per-seed-regression 0", script)
        self.assertIn("--max-guard-intervention-rate 0.1189", script)
        self.assertIn("--max-total-heuristic-fallback-rate 0.4779", script)
        self.assertIn("--fail-on-blockers", script)

    def test_mind_gate_cli_fail_on_blockers_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"
            ledger_path = tmp_path / "mind-gate-ledger.jsonl"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--experiment-ledger-output",
                        str(ledger_path),
                        "--min-viable-run-share",
                        "2.0",
                        "--fail-on-blockers",
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    mind_gate.main()

            self.assertEqual(raised.exception.code, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["readiness"]["status"], "fail")

    def test_mind_gate_cli_evaluates_validation_tick_matrix(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"
            ledger_path = tmp_path / "mind-gate-ledger.jsonl"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--validation-ticks",
                        "2,3",
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--experiment-ledger-output",
                        str(ledger_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                mind_gate.main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["protocol"]["validation_ticks"], [2, 3])
            self.assertEqual(
                [entry["ticks"] for entry in report["evaluation_matrix"]],
                [2, 3],
            )
            self.assertEqual(report["evaluation"]["protocol"]["ticks"], 2)
            self.assertIn(report["readiness"]["status"], {"pass", "review", "fail"})

    def test_mind_gate_cli_fail_on_review_exits_nonzero(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mind-gate-report.json"
            artifact_path = tmp_path / "mind-gate-artifact.json"
            ledger_path = tmp_path / "mind-gate-ledger.jsonl"

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_gate",
                        "--train-seed",
                        "7",
                        "--validation-seed",
                        "8",
                        "--ticks",
                        "2",
                        "--artifact-output",
                        str(artifact_path),
                        "--output",
                        str(report_path),
                        "--experiment-ledger-output",
                        str(ledger_path),
                        "--min-births-per-run-mean",
                        "999.0",
                        "--max-guard-intervention-rate",
                        "1.0",
                        "--max-guard-intervention-rate-by-group",
                        "1.0",
                        "--fail-on-review",
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
                patch("sys.stderr", io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    mind_gate.main()

            self.assertEqual(raised.exception.code, 1)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["readiness"]["status"], "review")

    def test_mind_train_cli_accepts_seed_bank_trajectories(self) -> None:
        with TemporaryDirectory() as tmpdir:
            first_trajectory_path = Path(tmpdir) / "trajectory-seed7.jsonl.gz"
            second_trajectory_path = Path(tmpdir) / "trajectory-seed8.jsonl.gz"
            artifact_path = Path(tmpdir) / "bc-seed-bank-artifact.json"
            self._write_tiny_trajectory(first_trajectory_path, seed=7)
            self._write_tiny_trajectory(second_trajectory_path, seed=8)

            with (
                patch(
                    "sys.argv",
                    [
                        "mind_train",
                        "--trajectory",
                        str(first_trajectory_path),
                        "--trajectory",
                        str(second_trajectory_path),
                        "--output",
                        str(artifact_path),
                    ],
                ),
                patch("sys.stdout", io.StringIO()),
            ):
                mind_train.main()

            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_model_artifact_manifest(artifact)
            provenance = artifact["manifest"]["provenance"]
            self.assertEqual(provenance["source_seeds"], [7, 8])
            self.assertEqual(provenance["source_dataset_count"], 2)
            self.assertEqual(
                provenance["record_count"],
                artifact["manifest"]["trained_record_count"],
            )
            self.assertEqual(
                provenance["trajectory_paths"],
                [str(first_trajectory_path), str(second_trajectory_path)],
            )
            self.assertGreater(len(artifact["model"]["conditional_action_scores"]), 0)

    def test_combined_dataset_provenance_rejects_mismatched_contracts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            first_trajectory_path = Path(tmpdir) / "trajectory-seed7.jsonl.gz"
            second_trajectory_path = Path(tmpdir) / "trajectory-stale.jsonl.gz"
            self._write_tiny_trajectory(first_trajectory_path, seed=7)
            with gzip.open(first_trajectory_path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            rows[-1]["provenance"]["contract_digest"] = "stale"
            with gzip.open(second_trajectory_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            first_dataset = load_trajectory_jsonl(first_trajectory_path)
            second_dataset = load_trajectory_jsonl(second_trajectory_path)

            with self.assertRaisesRegex(TrajectoryDatasetError, "contract digests"):
                combined_dataset_provenance([first_dataset, second_dataset])

    def test_learned_policy_obeys_action_mask(self) -> None:
        policy = LearnedPolicy(action_scores={"eat": 1.0, "stay": 0.1})

        blocked = policy.decide({}, {"stay": True, "eat": False})
        allowed = policy.decide({}, {"stay": True, "eat": True})

        self.assertEqual(blocked.requested_action, "stay")
        self.assertEqual(allowed.requested_action, "eat")

    def test_feature_policy_keeps_no_signal_local_patch_tokens_neutral(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
            },
            "local_patch": [
                {"dx": 0, "dy": 0, "terrain": "plain", "food": 0.0},
                {"dx": 1, "dy": 0, "terrain": "plain", "food": 0.0},
                {"dx": -1, "dy": 0, "terrain": "plain", "food": 0.0},
            ],
            "navigation": {},
        }

        feature_key = feature_keys_from_observation(
            observation,
            {"stay": True, "eat": True},
        )[0]

        self.assertIn("foodc0:waterc0:carrionc0:preyc0:riskc0", feature_key)

    def test_learned_policy_uses_contextual_scores_before_global_prior(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.2,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
            },
            "local_patch": [
                {"dx": 0, "dy": 0, "food": 0.2, "fresh_kill_energy": 0.0, "carcass_energy": 0.0}
            ],
            "navigation": {},
        }
        action_mask = {"stay": True, "eat": True}
        contextual_key = feature_keys_from_observation(observation, action_mask)[0]
        policy = LearnedPolicy(
            action_scores={"eat": 1.0, "stay": 0.1},
            conditional_action_scores={contextual_key: {"eat": 0.0, "stay": 1.0}},
        )

        decision = policy.decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "stay")

    def test_learned_policy_reports_score_support_and_margin_diagnostics(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "food": 0.2,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                }
            ],
            "navigation": {},
        }
        action_mask = {"stay": True, "eat": True}
        contextual_key = feature_keys_from_observation(observation, action_mask)[0]
        policy = LearnedPolicy(
            action_scores={"eat": 0.2, "stay": 0.8},
            action_score_metadata={
                "record_count": 100,
                "score_margin": 0.6,
            },
            conditional_action_scores={contextual_key: {"eat": 0.75, "stay": 0.25}},
            conditional_action_metadata={
                contextual_key: {
                    "record_count": 14,
                    "score_margin": 0.5,
                }
            },
        )

        decision = policy.decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "eat")
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertEqual(diagnostics["score_source"], "conditional")
        self.assertEqual(diagnostics["score_feature_key"], contextual_key)
        self.assertEqual(diagnostics["score_match_depth"], 0)
        self.assertEqual(diagnostics["score_support"], 14)
        self.assertAlmostEqual(float(diagnostics["learned_score_margin"]), 0.5)
        self.assertAlmostEqual(float(diagnostics["training_score_margin"]), 0.5)

    def test_heuristic_guard_conserves_low_energy_agent_before_learned_move(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.42,
                "hydration_ratio": 0.7,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        action_mask = {"stay": True, "move_north": True}
        policy = LearnedPolicy(
            action_scores={"move_north": 1.0, "stay": 0.1},
            heuristic_guard=True,
        )

        decision = policy.decide(observation, action_mask)

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_guard_defers_low_confidence_learned_action(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.42, "stay": 0.4},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_guard_defers_small_margin_learned_disagreement(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.58, "stay": 0.52},
            heuristic_guard=True,
            heuristic_override_min_margin=0.1,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_delegate_handles_low_margin_action_prior(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.58, "stay": 0.52},
            action_score_metadata={"record_count": 20, "score_margin": 0.1},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=0.1,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertTrue(decision.diagnostics["heuristic_delegate_used"])
        self.assertEqual(
            decision.diagnostics["heuristic_delegate_reason"],
            "low_confidence_action_prior",
        )

    def test_heuristic_delegate_uses_conservative_delegate_margin(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 0.7, "stay": 0.3},
            action_score_metadata={
                "record_count": 20,
                "score_margin": 0.4,
                "delegate_score_margin": 0.05,
            },
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=0.1,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertTrue(diagnostics["heuristic_delegate_used"])
        self.assertAlmostEqual(float(diagnostics["training_score_margin"]), 0.05)
        self.assertEqual(
            diagnostics["heuristic_delegate_reason"],
            "low_confidence_action_prior",
        )

    def test_neural_policy_delegates_using_live_score_margin(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"stay": 1.0},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 1.0,
            },
            neural_network={"patched": True},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
        )

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                {action: 0.0 for action in ACTION_NAMES} | {"eat": 0.5, "stay": 0.5},
                {action: 0.0 for action in ACTION_NAMES},
                0.0,
            ),
        ):
            decision = policy.decide(observation, {"eat": True, "stay": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertTrue(diagnostics["heuristic_delegate_used"])
        self.assertAlmostEqual(float(diagnostics["training_score_margin"]), 0.0)

    def test_neural_policy_renormalizes_scores_over_action_mask_for_delegate_margin(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"stay": 1.0},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 1.0,
            },
            neural_network={"patched": True},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.0,
            heuristic_override_min_margin=0.0,
        )
        raw_scores = {action: 0.0 for action in ACTION_NAMES}
        raw_scores.update({"attack_north": 0.9, "eat": 0.08, "stay": 0.02})

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                raw_scores,
                {action: 0.0 for action in ACTION_NAMES},
                0.0,
            ),
        ):
            decision = policy.decide(observation, {"eat": True, "stay": True})

        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertNotIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertFalse(diagnostics["heuristic_delegate_used"])
        self.assertAlmostEqual(float(diagnostics["learned_score"]), 0.8)
        self.assertAlmostEqual(float(diagnostics["learned_runner_up_score"]), 0.2)
        self.assertAlmostEqual(float(diagnostics["training_score_margin"]), 0.6)

    def test_neural_policy_anchors_actor_ranking_to_contextual_prior(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.8},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={
                action: 0.0 for action in ACTION_NAMES
            }
            | {"eat": 0.05, "move_east": 0.95},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 0.9,
            },
            neural_network={"patched": True},
            neural_actor_prior_policy="contextual_prior_score_anchor_v1",
            neural_actor_prior_blend_weight=0.9,
        )
        raw_scores = {action: 0.0 for action in ACTION_NAMES}
        raw_scores.update({"eat": 0.8, "move_east": 0.2})

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                raw_scores,
                {action: 0.0 for action in ACTION_NAMES},
                0.0,
            ),
        ):
            decision = policy.decide(
                observation,
                {"eat": True, "move_east": True},
            )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertEqual(
            diagnostics["score_source"],
            "neural_actor_critic:contextual_prior_score_anchor_v1",
        )
        self.assertAlmostEqual(float(diagnostics["learned_score"]), 0.875)
        self.assertAlmostEqual(float(diagnostics["learned_runner_up_score"]), 0.125)

    def test_neural_policy_anchors_actor_ranking_to_advantage_blended_prior(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.9,
                "hydration_ratio": 0.9,
                "health_ratio": 1.0,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "neighbors": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.8},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={
                action: 0.0 for action in ACTION_NAMES
            }
            | {"eat": 0.05, "move_east": 0.95},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 0.9,
            },
            neural_network={"patched": True},
            neural_actor_prior_policy=(
                "advantage_blended_contextual_prior_score_anchor_v1"
            ),
            neural_actor_prior_blend_weight=0.9,
        )
        raw_scores = {action: 0.0 for action in ACTION_NAMES}
        raw_scores.update({"eat": 0.8, "move_east": 0.2})

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                raw_scores,
                {action: 0.0 for action in ACTION_NAMES},
                0.0,
            ),
        ):
            decision = policy.decide(
                observation,
                {"eat": True, "move_east": True},
            )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertEqual(
            diagnostics["score_source"],
            (
                "neural_actor_critic:"
                "advantage_blended_contextual_prior_score_anchor_v1"
            ),
        )
        self.assertAlmostEqual(float(diagnostics["learned_score"]), 0.875)
        self.assertAlmostEqual(float(diagnostics["learned_runner_up_score"]), 0.125)

    def test_value_supported_deviation_bypasses_delegate_and_guard(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.32,
                    "vegetation": 0.32,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.72, "stay": 0.01},
            action_score_metadata={"record_count": 64, "delegate_score_margin": 0.04},
            action_value_estimates={"eat": 0.12, "stay": 0.02},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
        )

        decision = policy.decide(observation, {"eat": True, "stay": True})

        self.assertEqual(decision.requested_action, "eat")
        self.assertNotIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertFalse(diagnostics["heuristic_delegate_used"])
        self.assertTrue(diagnostics["safe_deviation_used"])
        self.assertEqual(
            diagnostics["safe_deviation_reason"],
            "value_supported_local_resource_eat",
        )
        self.assertAlmostEqual(float(diagnostics["learned_action_value"]), 0.12)
        self.assertAlmostEqual(float(diagnostics["heuristic_action_value"]), 0.02)

    def test_neural_value_supported_resource_action_bypasses_fallback(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.9,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.32,
                    "vegetation": 0.32,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"stay": 1.0},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 1.0,
            },
            neural_network={"patched": True},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
            value_supported_deviation_min_score_margin=0.25,
            value_supported_deviation_min_predicted_advantage=0.24,
        )

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                {action: 0.0 for action in ACTION_NAMES}
                | {"eat": 0.74, "move_east": 0.20},
                {action: 0.0 for action in ACTION_NAMES}
                | {"eat": 0.34, "move_east": 0.04},
                0.10,
            ),
        ):
            decision = policy.decide(
                observation,
                {"eat": True, "move_east": True, "stay": True},
            )

        self.assertEqual(decision.requested_action, "eat")
        self.assertNotIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertFalse(diagnostics["heuristic_delegate_used"])
        self.assertTrue(diagnostics["safe_deviation_used"])
        self.assertEqual(
            diagnostics["safe_deviation_reason"],
            "value_supported_neural_resource_action",
        )
        self.assertAlmostEqual(
            float(diagnostics["learned_action_predicted_advantage"]),
            0.24,
        )

    def test_neural_value_supported_resource_action_allows_prior_blend_source(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.9,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.32,
                    "vegetation": 0.32,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={action: 0.0 for action in ACTION_NAMES} | {"eat": 1.0},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 1.0,
            },
            neural_network={"patched": True},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
            value_supported_deviation_min_score_margin=0.25,
            value_supported_deviation_min_predicted_advantage=0.24,
            neural_actor_prior_policy="contextual_prior_score_anchor_v1",
            neural_actor_prior_blend_weight=0.1,
        )

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                {action: 0.0 for action in ACTION_NAMES}
                | {"eat": 0.74, "move_east": 0.20},
                {action: 0.0 for action in ACTION_NAMES}
                | {"eat": 0.34, "move_east": 0.04},
                0.10,
            ),
        ):
            decision = policy.decide(
                observation,
                {"eat": True, "move_east": True, "stay": True},
            )

        self.assertEqual(decision.requested_action, "eat")
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertEqual(
            diagnostics["score_source"],
            "neural_actor_critic:contextual_prior_score_anchor_v1",
        )
        self.assertTrue(diagnostics["safe_deviation_used"])
        self.assertEqual(
            diagnostics["safe_deviation_reason"],
            "value_supported_neural_resource_action",
        )

    def test_neural_value_supported_resource_action_requires_advantage(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.9,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.32,
                    "vegetation": 0.32,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"stay": 1.0},
            action_score_metadata={
                "record_count": 100,
                "delegate_score_margin": 1.0,
            },
            neural_network={"patched": True},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
            value_supported_deviation_min_score_margin=0.25,
            value_supported_deviation_min_predicted_advantage=0.24,
        )

        with patch(
            "evolution_sim.mind.learned_policy.score_neural_actor_critic",
            return_value=(
                {action: 0.0 for action in ACTION_NAMES}
                | {"eat": 0.74, "move_east": 0.20},
                {action: 0.0 for action in ACTION_NAMES}
                | {"eat": 0.34, "move_east": 0.04},
                0.12,
            ),
        ):
            decision = policy.decide(
                observation,
                {"eat": True, "move_east": True, "stay": True},
            )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertIn(
            "observation_heuristic",
            decision.source,
        )
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertFalse(diagnostics["heuristic_delegate_used"])
        self.assertFalse(diagnostics["safe_deviation_used"])

    def test_value_supported_deviation_requires_context_support(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.32,
                    "vegetation": 0.32,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.72, "stay": 0.01},
            action_score_metadata={"record_count": 12, "delegate_score_margin": 0.04},
            action_value_estimates={"eat": 0.12, "stay": 0.02},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
        )

        decision = policy.decide(observation, {"eat": True, "stay": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn(
            "observation_heuristic_confidence_delegate_v1",
            decision.source,
        )
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertTrue(diagnostics["heuristic_delegate_used"])
        self.assertFalse(diagnostics["safe_deviation_used"])

    def test_value_supported_deviation_rejects_local_eat_over_move(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.32,
                    "vegetation": 0.32,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.72, "move_east": 0.2, "stay": 0.01},
            action_score_metadata={"record_count": 64, "delegate_score_margin": 0.04},
            action_value_estimates={"eat": 0.12, "move_east": 0.02, "stay": 0.01},
            heuristic_guard=True,
            heuristic_delegate=True,
            heuristic_delegate_max_training_score_margin=0.25,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertIn("observation_heuristic_confidence_delegate_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertTrue(diagnostics["heuristic_delegate_used"])
        self.assertFalse(diagnostics["safe_deviation_used"])
        self.assertEqual(diagnostics["heuristic_action"], "move_east")

    def test_value_supported_deviation_allows_local_eat_over_conservation(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "carnivore",
                "meat_mode": "hunter",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.2,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.72, "stay": 0.01},
            action_score_metadata={"record_count": 64, "delegate_score_margin": 0.4},
            action_value_estimates={"eat": 0.12, "stay": 0.02},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            value_supported_deviation_policy="positive_value_safe_deviation_v1",
            value_supported_deviation_min_support=32,
            value_supported_deviation_min_value_margin=0.04,
            value_supported_deviation_min_learned_value=0.02,
        )

        decision = policy.decide(observation, {"eat": True, "stay": True})

        self.assertEqual(decision.requested_action, "eat")
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        diagnostics = decision.diagnostics or {}
        self.assertTrue(diagnostics["safe_deviation_used"])
        self.assertEqual(
            diagnostics["safe_deviation_reason"],
            "value_supported_local_resource_eat",
        )
        self.assertEqual(diagnostics["heuristic_action"], "stay")

    def test_heuristic_guard_preserves_explicit_conservation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.7,
                "hydration_ratio": 0.8,
                "health_ratio": 1.0,
                "trophic_role": "carnivore",
                "meat_mode": "hunter",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.0,
                    "vegetation": 0.0,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                }
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_north": 1.0, "stay": 0.0},
            heuristic_guard=True,
        )

        decision = policy.decide(observation, {"stay": True, "move_north": True})

        self.assertEqual(decision.requested_action, "stay")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)

    def test_heuristic_guard_allows_safe_local_eat_deviation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.28,
                    "vegetation": 0.3,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.3, "move_east": 0.2, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_local_eat_min_score=0.25,
            heuristic_safe_local_eat_min_food=0.25,
            heuristic_safe_local_eat_min_plant_ratio=0.5,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "eat")
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertTrue(decision.diagnostics["safe_deviation_used"])
        self.assertEqual(
            decision.diagnostics["safe_deviation_reason"],
            "local_resource_eat",
        )
        self.assertEqual(decision.diagnostics["heuristic_action"], "move_east")

    def test_heuristic_guard_rejects_inferior_local_eat_deviation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.28,
                    "vegetation": 0.3,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.3, "move_east": 0.2, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_local_eat_min_score=0.25,
            heuristic_safe_local_eat_min_food=0.25,
            heuristic_safe_local_eat_min_plant_ratio=1.0,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertFalse(decision.diagnostics["safe_deviation_used"])

    def test_heuristic_guard_rejects_meat_specialist_plant_only_local_eat_deviation(
        self,
    ) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.95,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "carnivore",
                "meat_mode": "hunter",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.45,
                    "vegetation": 0.45,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"eat": 0.3, "move_east": 0.2, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_local_eat_min_score=0.25,
            heuristic_safe_local_eat_min_food=0.25,
            heuristic_safe_local_eat_min_plant_ratio=0.5,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertNotEqual(decision.requested_action, "eat")
        self.assertIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertFalse(decision.diagnostics["safe_deviation_used"])

    def test_heuristic_guard_allows_safe_plant_move_deviation(self) -> None:
        observation = {
            "self": {
                "energy_ratio": 0.82,
                "hydration_ratio": 0.86,
                "health_ratio": 0.9,
                "trophic_role": "herbivore",
                "meat_mode": "none",
                "matched_diet_ratio": 1.0,
            },
            "local_patch": [
                {
                    "dx": 0,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "self",
                    "water_access_reason": "none",
                    "food": 0.25,
                    "vegetation": 0.3,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
                {
                    "dx": 1,
                    "dy": 0,
                    "in_bounds": True,
                    "terrain": "plain",
                    "occupant": "none",
                    "water_access_reason": "none",
                    "food": 0.5,
                    "vegetation": 0.5,
                    "recovery_debt": 0.0,
                    "fresh_kill_energy": 0.0,
                    "carcass_energy": 0.0,
                    "hazard_level": 0.0,
                    "prey_biomass": 0.0,
                    "carrion_signal": 0.0,
                    "predator_risk": 0.0,
                },
            ],
            "navigation": {
                "water": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "plant": {"dx": 1, "dy": 0, "distance": 1, "strength": 0.5},
                "carrion": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
                "prey": {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0},
            },
        }
        policy = LearnedPolicy(
            action_scores={"move_east": 0.4, "eat": 0.3, "stay": 0.0},
            heuristic_guard=True,
            heuristic_confidence_threshold=0.5,
            heuristic_override_min_margin=1.0,
            heuristic_safe_plant_move_min_score=0.4,
            heuristic_safe_plant_move_min_strength=0.3,
            heuristic_safe_plant_move_max_local_food_ratio=0.9,
            heuristic_safe_plant_move_max_distance=3,
        )

        decision = policy.decide(
            observation,
            {"eat": True, "move_east": True, "stay": True},
        )

        self.assertEqual(decision.requested_action, "move_east")
        self.assertNotIn("observation_heuristic_safety_floor_v1", decision.source)
        self.assertIsNotNone(decision.diagnostics)
        self.assertTrue(decision.diagnostics["safe_deviation_used"])
        self.assertEqual(
            decision.diagnostics["safe_deviation_reason"],
            "stronger_plant_navigation",
        )

    def test_policy_evaluation_compares_heuristic_and_learned_on_summary_only_seeds(self) -> None:
        policy = LearnedPolicy(action_scores={"stay": 1.0})

        report = compare_heuristic_and_learned(
            learned_policy=policy,
            seeds=[7],
            ticks=2,
        )

        self.assertEqual(report["protocol"]["mode"], RunMode.SUMMARY_ONLY.value)
        self.assertIn("trajectory", report["learned"]["aggregate"])
        self.assertIn("mind_v1_gates", report)
        self.assertIn(report["mind_v1_gates"]["status"], {"pass", "review", "fail"})
        self.assertIn("heuristic", report)
        self.assertIn("learned", report)
        self.assertIn("by_trophic_role", report["comparison"])
        self.assertIn("by_meat_mode", report["comparison"])
        self.assertIn(
            "policy_diagnostics_by_trophic_role",
            report["comparison"],
        )
        self.assertIn(
            "policy_diagnostics_by_meat_mode",
            report["comparison"],
        )
        role_comparison = report["comparison"]["by_trophic_role"]
        self.assertGreaterEqual(len(role_comparison), 1)
        role_delta = next(iter(role_comparison.values()))
        self.assertEqual(
            role_delta["total_delta"],
            role_delta["learned_total"] - role_delta["heuristic_total"],
        )
        mode_comparison = report["comparison"]["policy_diagnostics_by_meat_mode"]
        self.assertGreaterEqual(len(mode_comparison), 1)
        mode_delta = next(iter(mode_comparison.values()))
        self.assertIn("mean_reward_delta", mode_delta)
        self.assertIn("guard_intervention_rate_delta", mode_delta)
        learned_diagnostics = report["learned"]["aggregate"]["policy_diagnostics"]
        self.assertIn("guard_intervention_rate", learned_diagnostics)
        self.assertIn("heuristic_delegate_rate", learned_diagnostics)
        self.assertIn("action_source_counts", learned_diagnostics)
        self.assertIn("by_trophic_role", learned_diagnostics)
        self.assertIn("by_meat_mode", learned_diagnostics)
        self.assertIn("heuristic_delegate_by_score_source", learned_diagnostics)
        self.assertIn("heuristic_delegate_by_support_bucket", learned_diagnostics)
        self.assertIn("heuristic_delegate_by_score_margin_bucket", learned_diagnostics)
        self.assertIn("top_delegated_contexts", learned_diagnostics)
        json.dumps(report)

    def test_mind_evaluation_uses_public_reporting_helpers(self) -> None:
        source = Path("python/evolution_sim/mind/evaluation.py").read_text(
            encoding="utf-8",
        )

        self.assertNotIn("from evolution_sim.cli.evaluate import _", source)

    def test_policy_diagnostics_report_guard_actions_and_contexts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        records = [dict(dataset.records[0]), dict(dataset.records[1])]
        records[0]["action_source"] = (
            "mind_v1_learned_policy:observation_heuristic_safety_floor_v1"
        )
        records[1]["action_source"] = "mind_v1_learned_policy"

        diagnostics = build_policy_diagnostics(records)

        guarded_action = str(records[0]["requested_action"])
        self.assertEqual(diagnostics["guard_intervention_count"], 1)
        self.assertIn(guarded_action, diagnostics["guard_intervention_by_action"])
        self.assertEqual(
            diagnostics["guard_intervention_by_action"][guarded_action][
                "guard_intervention_count"
            ],
            1,
        )
        self.assertGreaterEqual(len(diagnostics["top_guarded_contexts"]), 1)
        top_context = diagnostics["top_guarded_contexts"][0]
        self.assertIn("feature_key", top_context)
        self.assertEqual(top_context["guard_intervention_count"], 1)
        self.assertIn("action_counts", top_context)

    def test_policy_diagnostics_report_heuristic_delegate_share(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        record = dict(dataset.records[0])
        record["action_source"] = (
            "mind_v1_learned_policy:"
            "observation_heuristic_confidence_delegate_v1"
        )

        diagnostics = build_policy_diagnostics(
            [record],
            decision_diagnostics=[
                {
                    "heuristic_delegate_used": True,
                    "learned_action": "eat",
                    "heuristic_action": record["requested_action"],
                    "score_source": "conditional",
                    "score_support": 0,
                    "learned_score_margin": 0.12,
                }
            ],
        )

        requested_action = str(record["requested_action"])
        self.assertEqual(diagnostics["guard_intervention_count"], 0)
        self.assertEqual(diagnostics["heuristic_delegate_count"], 1)
        self.assertEqual(diagnostics["heuristic_delegate_rate"], 1.0)
        self.assertEqual(
            diagnostics["heuristic_delegate_by_action"][requested_action][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_suppressed_learned_action"]["eat"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_by_score_source"]["conditional"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_by_support_bucket"]["0"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["heuristic_delegate_by_score_margin_bucket"]["0.10-0.24"][
                "heuristic_delegate_count"
            ],
            1,
        )
        self.assertGreaterEqual(len(diagnostics["top_delegated_contexts"]), 1)
        top_context = diagnostics["top_delegated_contexts"][0]
        self.assertIn("feature_key", top_context)
        self.assertEqual(top_context["heuristic_delegate_count"], 1)
        self.assertIn("action_counts", top_context)
        self.assertEqual(
            sum(
                int(group["heuristic_delegate_count"])
                for group in diagnostics["by_trophic_role"].values()
            ),
            1,
        )
        self.assertEqual(
            sum(
                int(group["heuristic_delegate_count"])
                for group in diagnostics["by_meat_mode"].values()
            ),
            1,
        )

    def test_policy_evaluation_reports_guard_suppressed_learned_actions(self) -> None:
        class AlwaysGuardedPolicy:
            policy_id = "mind_v1_learned_policy"
            policy_version = "mind_v1_learned_policy_test"

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                return ActionDecision(
                    requested_action="stay",
                    source=(
                        "mind_v1_learned_policy:"
                        "observation_heuristic_safety_floor_v1"
                    ),
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    diagnostics={
                        "guard_used": True,
                        "learned_action": "eat",
                        "heuristic_action": "stay",
                        "score_source": "conditional",
                        "score_support": 14,
                        "learned_score_margin": 0.5,
                    },
                )

        report = compare_heuristic_and_learned(
            learned_policy=AlwaysGuardedPolicy(),
            seeds=[7],
            ticks=1,
        )

        diagnostics = report["learned"]["aggregate"]["policy_diagnostics"]
        self.assertIn("guard_suppressed_learned_action", diagnostics)
        self.assertEqual(
            diagnostics["guard_suppressed_learned_action"]["eat"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertEqual(
            diagnostics["guard_intervention_by_score_source"]["conditional"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertEqual(
            diagnostics["guard_intervention_by_support_bucket"]["10-31"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertEqual(
            diagnostics["guard_intervention_by_score_margin_bucket"]["0.50-0.99"][
                "guard_intervention_count"
            ],
            diagnostics["guard_intervention_count"],
        )
        self.assertNotIn(
            "policy_decision_diagnostics",
            report["learned"]["runs"][0]["trajectory"],
        )

    def test_trajectory_writer_can_persist_policy_decision_diagnostics_opt_in(
        self,
    ) -> None:
        class AlwaysDelegatedPolicy:
            policy_id = "mind_v2_neural_policy"
            policy_version = "mind_v2_neural_policy_test"

            def decide(
                self,
                observation: dict[str, object],
                action_mask: dict[str, bool],
            ) -> ActionDecision:
                return ActionDecision(
                    requested_action="stay",
                    source=(
                        "mind_v2_neural_policy:"
                        "observation_heuristic_confidence_delegate_v1"
                    ),
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    diagnostics={
                        "guard_used": False,
                        "heuristic_delegate_used": True,
                        "learned_action": "eat",
                        "heuristic_action": "stay",
                        "score_source": "neural_actor",
                        "score_support": 0,
                        "learned_score_margin": 0.1,
                    },
                )

        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="learned-rollout",
                include_policy_decision_diagnostics=True,
            )
            SimulationWorld(
                WorldConfig(seed=7, max_ticks=1),
                policy=AlwaysDelegatedPolicy(),
            ).run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )
            dataset = load_trajectory_jsonl(trajectory_path)

        self.assertEqual(
            dataset.header["optional_record_fields"],
            ["policy_decision_diagnostics"],
        )
        diagnostic_records = [
            record
            for record in dataset.records
            if isinstance(record.get("policy_decision_diagnostics"), dict)
        ]
        self.assertGreater(len(diagnostic_records), 0)
        diagnostics = diagnostic_records[0]["policy_decision_diagnostics"]
        self.assertEqual(diagnostics["learned_action"], "eat")
        transitions = build_trajectory_transitions(dataset.records)
        diagnostic_transitions = [
            transition
            for transition in transitions
            if transition.policy_decision_diagnostics is not None
        ]
        self.assertGreater(len(diagnostic_transitions), 0)
        self.assertIn(
            "observation_heuristic_confidence_delegate_v1",
            diagnostic_transitions[0].action_source,
        )

    def test_trajectory_writer_can_persist_policy_update_trace_opt_in(self) -> None:
        base_policy = LearnedPolicy(
            action_scores={
                action: (0.1 if action == "eat" else 0.0)
                for action in ACTION_NAMES
            },
            heuristic_guard=False,
            heuristic_delegate=False,
        )
        policy = OnlineAdaptiveMindPolicy(
            base_policy=base_policy,
            learning_rate=0.05,
        )
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "online-trace.jsonl.gz"
            writer = JsonlTrajectoryWriter(
                trajectory_path,
                source_seeds=[7],
                split_id="online-trace",
                include_policy_decision_diagnostics=True,
                include_policy_update_trace=True,
            )
            SimulationWorld(
                WorldConfig(seed=7, max_ticks=1),
                policy=policy,
            ).run(
                mode=RunMode.SUMMARY_ONLY,
                trajectory_sink=writer,
            )
            dataset = load_trajectory_jsonl(trajectory_path)

        self.assertEqual(
            dataset.header["optional_record_fields"],
            ["policy_decision_diagnostics", "policy_update_trace"],
        )
        traces = [
            record["policy_update_trace"]
            for record in dataset.records
            if isinstance(record.get("policy_update_trace"), dict)
        ]
        self.assertGreater(len(traces), 0)
        self.assertEqual(
            traces[0]["schema_version"],
            "mind_policy_update_trace_v1",
        )
        self.assertTrue(replay_online_update_traces(traces))
        transitions = build_trajectory_transitions(dataset.records)
        traced_transitions = [
            transition
            for transition in transitions
            if transition.policy_update_trace is not None
        ]
        self.assertGreater(len(traced_transitions), 0)
        self.assertEqual(
            traced_transitions[0].policy_update_trace["schema_version"],
            "mind_policy_update_trace_v1",
        )

    def test_policy_diagnostics_separates_zero_support_bucket(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        record = dict(dataset.records[0])
        record["action_source"] = (
            "mind_v1_learned_policy:observation_heuristic_safety_floor_v1"
        )
        record["policy_decision_diagnostics"] = {
            "guard_used": True,
            "score_support": 0,
        }

        diagnostics = build_policy_diagnostics(
            [record],
            decision_diagnostics=[record["policy_decision_diagnostics"]],
        )

        self.assertIn("0", diagnostics["guard_intervention_by_support_bucket"])
        self.assertNotIn("1-9", diagnostics["guard_intervention_by_support_bucket"])

    def test_policy_diagnostics_reports_safe_deviation_share(self) -> None:
        with TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
            self._write_tiny_trajectory(trajectory_path)
            dataset = load_trajectory_jsonl(trajectory_path)

        record = dict(dataset.records[0])
        record["action_source"] = "mind_v1_learned_policy"
        diagnostic = {
            "guard_used": False,
            "safe_deviation_used": True,
            "learned_action": "eat",
        }

        diagnostics = build_policy_diagnostics(
            [record],
            decision_diagnostics=[diagnostic],
        )

        self.assertEqual(diagnostics["safe_deviation_count"], 1)
        self.assertEqual(diagnostics["safe_deviation_rate"], 1.0)
        self.assertEqual(
            diagnostics["safe_deviation_by_action"]["eat"]["safe_deviation_count"],
            1,
        )

    def test_policy_evaluation_reports_paired_seed_deltas(self) -> None:
        policy = LearnedPolicy(action_scores={"stay": 1.0})

        report = compare_heuristic_and_learned(
            learned_policy=policy,
            seeds=[7],
            ticks=2,
        )

        per_seed = report["comparison"]["per_seed"]
        self.assertEqual(len(per_seed), 1)
        self.assertEqual(per_seed[0]["seed"], 7)
        self.assertEqual(
            per_seed[0]["alive_agents_delta"],
            per_seed[0]["learned_alive_agents"] - per_seed[0]["heuristic_alive_agents"],
        )
        self.assertEqual(
            per_seed[0]["births_delta"],
            per_seed[0]["learned_births"] - per_seed[0]["heuristic_births"],
        )

    def test_policy_evaluation_applies_mind_gate_criteria(self) -> None:
        policy = LearnedPolicy(action_scores={"stay": 1.0})

        report = compare_heuristic_and_learned(
            learned_policy=policy,
            seeds=[7],
            ticks=2,
            gate_criteria={"min_viable_run_share": 2.0},
        )

        self.assertEqual(report["protocol"]["gate_criteria"]["min_viable_run_share"], 2.0)
        self.assertEqual(report["mind_v1_gates"]["status"], "fail")
        self.assertEqual(
            report["mind_v1_gates"]["blockers"][0]["field"],
            "alive_agents",
        )

    def test_mind_gate_blocks_per_seed_alive_regression(self) -> None:
        learned_report = {
            "runs": [
                {
                    "seed": 1,
                    "alive_agents": 7,
                    "births": 8,
                    "land_tile_count": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                }
            ],
            "aggregate": {
                "births": {"mean": 8.0},
                "trajectory": {
                    "invalid_observation_action_rate": 0.0,
                },
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.0,
                    "by_trophic_role": {},
                    "by_meat_mode": {},
                },
            },
        }
        heuristic_report = {
            "runs": [{"seed": 1, "alive_agents": 10, "births": 8}],
            "aggregate": {"alive_agents": {"mean": 10.0}, "births": {"mean": 8.0}},
        }

        gate = build_mind_v1_gate_report(
            learned_report,
            baseline_report=heuristic_report,
            max_alive_agents_mean_regression=10.0,
            max_alive_agents_per_seed_regression=2.0,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "alive_agents.per_seed_delta",
        )

    def test_mind_gate_warns_on_per_seed_birth_regression(self) -> None:
        learned_report = {
            "runs": [
                {
                    "seed": 1,
                    "alive_agents": 10,
                    "births": 5,
                    "land_tile_count": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                }
            ],
            "aggregate": {
                "births": {"mean": 5.0},
                "trajectory": {
                    "invalid_observation_action_rate": 0.0,
                },
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.0,
                    "by_trophic_role": {},
                    "by_meat_mode": {},
                },
            },
        }
        heuristic_report = {
            "runs": [{"seed": 1, "alive_agents": 10, "births": 7}],
            "aggregate": {"alive_agents": {"mean": 10.0}, "births": {"mean": 7.0}},
        }

        gate = build_mind_v1_gate_report(
            learned_report,
            baseline_report=heuristic_report,
            max_births_mean_regression=10.0,
            max_births_per_seed_regression=1.0,
        )

        self.assertEqual(gate["status"], "review")
        self.assertEqual(
            gate["warnings"][0]["field"],
            "births.per_seed_delta",
        )

    def test_mind_gate_uses_policy_visible_invalid_action_rate(self) -> None:
        report = {
            "runs": [
                {
                    "alive_agents": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                    "land_tile_count": 1,
                }
            ],
            "aggregate": {
                "births": {"mean": 0.0},
                "trajectory": {
                    "invalid_action_rate": 0.0,
                    "invalid_observation_action_rate": 0.0,
                    "invalid_resolution_action_rate": 0.5,
                },
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.0,
                    "by_trophic_role": {},
                    "by_meat_mode": {},
                },
            },
        }

        gate = build_mind_v1_gate_report(report)

        self.assertEqual(gate["status"], "pass")

    def test_mind_gate_blocks_high_guard_intervention_rate(self) -> None:
        report = {
            "runs": [
                {
                    "alive_agents": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                    "land_tile_count": 1,
                }
            ],
            "aggregate": {
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.75,
                    "by_trophic_role": {
                        "herbivore": {"guard_intervention_rate": 0.75}
                    },
                    "by_meat_mode": {
                        "none": {"guard_intervention_rate": 0.75}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=0.5,
            max_guard_intervention_rate_by_group=0.6,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.guard_intervention_rate",
        )

    def test_mind_gate_blocks_total_heuristic_fallback_rate(self) -> None:
        report = {
            "runs": [
                {
                    "alive_agents": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                    "land_tile_count": 1,
                }
            ],
            "aggregate": {
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.2,
                    "heuristic_delegate_rate": 0.35,
                    "by_trophic_role": {
                        "herbivore": {"guard_intervention_rate": 0.2}
                    },
                    "by_meat_mode": {
                        "none": {"guard_intervention_rate": 0.2}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=1.0,
            max_guard_intervention_rate_by_group=1.0,
            max_total_heuristic_fallback_rate=0.5,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.total_heuristic_fallback_rate",
        )

    def test_mind_gate_blocks_guard_group_cap(self) -> None:
        report = {
            "runs": [
                {
                    "alive_agents": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                    "land_tile_count": 1,
                }
            ],
            "aggregate": {
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.4,
                    "by_trophic_role": {
                        "carnivore": {"guard_intervention_rate": 0.6}
                    },
                    "by_meat_mode": {
                        "hunter": {"guard_intervention_rate": 0.4}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=0.45,
            max_guard_intervention_rate_by_group=0.5,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.by_trophic_role.carnivore.guard_intervention_rate",
        )

    def test_mind_gate_blocks_insufficient_guard_reduction(self) -> None:
        report = {
            "runs": [
                {
                    "alive_agents": 1,
                    "resource_pressure": {
                        "plant_budget": {"energy_available_at_end": 1.0},
                    },
                    "land_tile_count": 1,
                }
            ],
            "aggregate": {
                "births": {"mean": 1.0},
                "trajectory": {"invalid_observation_action_rate": 0.0},
                "policy_diagnostics": {
                    "guard_intervention_rate": 0.42,
                    "by_trophic_role": {
                        "herbivore": {"guard_intervention_rate": 0.42}
                    },
                    "by_meat_mode": {
                        "none": {"guard_intervention_rate": 0.42}
                    },
                },
            },
        }

        gate = build_mind_v1_gate_report(
            report,
            max_guard_intervention_rate=1.0,
            max_guard_intervention_rate_by_group=1.0,
            min_guard_intervention_rate_reduction=0.1,
            reference_guard_intervention_rate=0.45,
        )

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(
            gate["blockers"][0]["field"],
            "policy_diagnostics.guard_intervention_rate_reduction",
        )


if __name__ == "__main__":
    unittest.main()
