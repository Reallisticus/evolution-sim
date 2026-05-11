from __future__ import annotations

from typing import Any

from evolution_sim.env.contracts import SUMMARY_SCHEMA_VERSION
from evolution_sim.env.runtime.action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract,
)
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    observation_contract,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reproduction import REPRODUCTIVE_GROUP_CONTRACT_VERSION
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_RECORD_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
    reward_contract,
)
from evolution_sim.mind.neural import (
    NEURAL_ACTOR_CRITIC_MODEL_TYPE,
    NEURAL_ACTOR_CRITIC_TRAINER,
    NEURAL_ARCHITECTURE,
    NEURAL_BACKEND,
    NEURAL_HIDDEN_UNITS,
    NEURAL_TRAINING_POLICY,
    TORCH_NEURAL_ACTOR_CRITIC_MODEL_TYPE,
    TORCH_NEURAL_ACTOR_CRITIC_TRAINER,
    TORCH_ADVANTAGE_ACTOR_CRITIC_MODEL_TYPE,
    TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER,
    TORCH_ADVANTAGE_TRAINING_POLICY,
    TORCH_DISCRETE_IQL_MODEL_TYPE,
    TORCH_DISCRETE_IQL_TRAINER,
    TORCH_DISCRETE_IQL_TRAINING_POLICY,
    TORCH_NEURAL_ARCHITECTURE,
    TORCH_NEURAL_BACKEND,
    TORCH_NEURAL_HIDDEN_UNITS,
    TORCH_NEURAL_TRAINING_POLICY,
)
from evolution_sim.mind.evolution import (
    MIND_V3_CONTROLLER_ARCHITECTURE,
    MIND_V3_CONTROLLER_SCHEMA_VERSION,
    MIND_V3_CONTEXT_FEATURE_FIELDS,
    MIND_V3_NEED_GATED_FEATURE_FIELDS,
)
from evolution_sim.mind.policy_inputs import ecological_policy_input_contract
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_ARCHITECTURE,
    MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
    MIND_V3_NEURAL_BACKEND,
    MIND_V3_NEURAL_MODEL_TYPE,
)

MIND_V1_DATA_CONTRACT_VERSION = "mind_v1_data_contract_v1"
MIND_MODEL_ARTIFACT_VERSION = "mind_model_artifact_v1"
MIND_ONLINE_LEARNING_CONTRACT_VERSION = "mind_online_learning_contract_v1"
MIND_V3_AUTONOMOUS_EVOLUTION_CONTRACT_VERSION = (
    "mind_v3_autonomous_evolution_contract_v1"
)
MIND_RUNTIME_ENABLED_DEFAULT = False


def mind_v1_data_contract(signal_config: Any | None = None) -> dict[str, object]:
    return {
        "contract_version": MIND_V1_DATA_CONTRACT_VERSION,
        "runtime_enabled_by_default": MIND_RUNTIME_ENABLED_DEFAULT,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "trajectory_record_fields": list(TRAJECTORY_RECORD_FIELDS),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_encoder_version": OBSERVATION_ENCODER_VERSION,
        "observation_input": {
            "shape": [OBSERVATION_INPUT_VECTOR_SIZE],
            "dtype": OBSERVATION_INPUT_DTYPE,
        },
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
        "action_contract": action_contract(signal_config),
        "observation_contract": observation_contract(signal_config),
        "reward_contract": reward_contract(),
        "model_artifact_version": MIND_MODEL_ARTIFACT_VERSION,
    }


def mind_online_learning_contract() -> dict[str, object]:
    return {
        "contract_version": MIND_ONLINE_LEARNING_CONTRACT_VERSION,
        "runtime_enabled_by_default": MIND_RUNTIME_ENABLED_DEFAULT,
        "online_weight_updates_enabled_by_default": False,
        "in_simulation_weight_updates_allowed": False,
        "current_executable_slice": "torch_discrete_iql_artifact_v1",
        "executable_slices": [
            "learned_policy_trajectory_collection_v1",
            "neural_actor_critic_bc_artifact_v1",
            "torch_actor_critic_bc_artifact_v1",
            "torch_advantage_actor_critic_bc_artifact_v1",
            "torch_discrete_iql_artifact_v1",
            "heuristic_free_autonomous_controller_v1",
            "in_run_contextual_bandit_adapter_v1",
        ],
        "trajectory_collection": {
            "heuristic_policy": "sim:trajectory",
            "learned_policy": (
                "sim:trajectory -- --mind-artifact <artifact> --enable-mind "
                "--mind-runtime-mode guarded"
            ),
            "runtime_modes": ["guarded", "autonomous", "autonomous-online"],
            "requires_explicit_mind_enable": True,
            "policy_decision_diagnostics_opt_in": (
                "--include-policy-diagnostics"
            ),
            "policy_update_trace_opt_in": "--include-policy-update-trace",
            "diagnostic_record_field": "policy_decision_diagnostics",
            "update_trace_record_field": "policy_update_trace",
            "diagnostic_payload_required_by_default": False,
            "summary_only": True,
            "full_replay_required": False,
        },
        "autonomous_controller": {
            "policy": "heuristic_free_autonomous_controller_v1",
            "enabled_by_default": False,
            "heuristic_guard": False,
            "heuristic_delegate": False,
            "action_mask_required": True,
            "promotion_status": "experiment_only",
        },
        "online_in_run_adapter": {
            "policy": "in_run_contextual_bandit_adapter_v1",
            "enabled_by_default": False,
            "updates": "contextual_action_score_offsets",
            "neural_weight_updates": False,
            "requires_trajectory_feedback": True,
            "update_trace_schema_version": "mind_policy_update_trace_v1",
            "update_trace_replay": "deterministic_context_action_offset_replay_v1",
            "promotion_status": "experiment_only",
        },
        "neural_actor_critic": {
            "trainer": NEURAL_ACTOR_CRITIC_TRAINER,
            "model_type": NEURAL_ACTOR_CRITIC_MODEL_TYPE,
            "backend": NEURAL_BACKEND,
            "architecture": NEURAL_ARCHITECTURE,
            "training_policy": NEURAL_TRAINING_POLICY,
            "hidden_units": NEURAL_HIDDEN_UNITS,
            "third_party_ml_dependency": None,
            "weights_mutable_during_run": False,
        },
        "torch_actor_critic": {
            "trainer": TORCH_NEURAL_ACTOR_CRITIC_TRAINER,
            "model_type": TORCH_NEURAL_ACTOR_CRITIC_MODEL_TYPE,
            "backend": TORCH_NEURAL_BACKEND,
            "architecture": TORCH_NEURAL_ARCHITECTURE,
            "training_policy": TORCH_NEURAL_TRAINING_POLICY,
            "hidden_units": TORCH_NEURAL_HIDDEN_UNITS,
            "third_party_ml_dependency": "requirements-mind-ml.txt",
            "weights_mutable_during_run": False,
            "inference_requires_third_party_ml_dependency": False,
        },
        "torch_advantage_actor_critic": {
            "trainer": TORCH_ADVANTAGE_ACTOR_CRITIC_TRAINER,
            "model_type": TORCH_ADVANTAGE_ACTOR_CRITIC_MODEL_TYPE,
            "backend": TORCH_NEURAL_BACKEND,
            "architecture": TORCH_NEURAL_ARCHITECTURE,
            "training_policy": TORCH_ADVANTAGE_TRAINING_POLICY,
            "hidden_units": TORCH_NEURAL_HIDDEN_UNITS,
            "third_party_ml_dependency": "requirements-mind-ml.txt",
            "weights_mutable_during_run": False,
            "inference_requires_third_party_ml_dependency": False,
        },
        "torch_discrete_iql": {
            "trainer": TORCH_DISCRETE_IQL_TRAINER,
            "model_type": TORCH_DISCRETE_IQL_MODEL_TYPE,
            "backend": TORCH_NEURAL_BACKEND,
            "architecture": TORCH_NEURAL_ARCHITECTURE,
            "training_policy": TORCH_DISCRETE_IQL_TRAINING_POLICY,
            "hidden_units": TORCH_NEURAL_HIDDEN_UNITS,
            "third_party_ml_dependency": "requirements-mind-ml.txt",
            "weights_mutable_during_run": False,
            "inference_requires_third_party_ml_dependency": False,
        },
        "algorithm_ladder": [
            {
                "stage": "neural_behavior_cloning_actor_critic",
                "status": "executable_schema",
                "purpose": (
                    "Train a deterministic neural policy and value head from "
                    "the existing trajectory bank before any online updates."
                ),
                "families": ["supervised_bc", "actor_critic_value_head"],
            },
            {
                "stage": "pytorch_behavior_cloning_actor_critic",
                "status": "optional_ml_executable",
                "purpose": (
                    "Train the same artifact-backed actor/value controller "
                    "with a real PyTorch optimization loop while keeping "
                    "runtime inference frozen and dependency-free."
                ),
                "families": ["pytorch", "supervised_bc", "actor_critic_value_head"],
            },
            {
                "stage": "conservative_offline_rl",
                "status": "first_advantage_weighted_slice",
                "purpose": (
                    "Improve beyond behavior cloning while limiting "
                    "out-of-distribution actions."
                ),
                "families": [
                    "advantage_weighted_bc",
                    "iql",
                    "cql",
                    "td3_bc",
                    "rebrac",
                ],
            },
            {
                "stage": "discrete_iql_actor_critic",
                "status": "transition_adapter_executable",
                "purpose": (
                    "Train Q and expectile V heads from trajectory transitions "
                    "and extract a masked advantage-weighted actor."
                ),
                "families": ["iql", "discrete_offline_rl", "awbc"],
            },
            {
                "stage": "offline_to_online_finetuning",
                "status": "planned",
                "purpose": (
                    "Collect learned-policy rollouts, retrain outside the "
                    "simulation tick loop, and promote only through held-out gates."
                ),
                "families": ["ppo", "sac", "offline_to_online_replay"],
            },
            {
                "stage": "open_ended_population_search",
                "status": "future",
                "purpose": (
                    "Maintain diverse controller lineages and environmental "
                    "curricula for observable emergent behavior."
                ),
                "families": ["quality_diversity", "map_elites", "poet"],
            },
            {
                "stage": "world_model_control",
                "status": "future",
                "purpose": (
                    "Learn predictive ecological dynamics before planning or "
                    "Dreamer-style imagination is allowed to affect runtime."
                ),
                "families": ["dreamer_v3_style_world_models"],
            },
        ],
        "promotion_gates": {
            "zero_per_seed_alive_regression": True,
            "zero_per_seed_birth_regression": True,
            "strict_hard_guard_cap": 0.1189,
            "strict_total_fallback_cap": 0.4779,
            "extended_validation_seeds": [5, 13, 19, 29, 37, 41],
            "extended_validation_ticks": [120, 180],
        },
    }


def mind_v3_autonomous_evolution_contract() -> dict[str, object]:
    return {
        "contract_version": MIND_V3_AUTONOMOUS_EVOLUTION_CONTRACT_VERSION,
        "enabled_by_default": False,
        "policy": "mind_v3_autonomous_evolution_policy_v1",
        "action_selection": "inherited_controller_masked_argmax_v1",
        "heuristic_guard": False,
        "heuristic_delegate": False,
        "heuristic_action_selection": False,
        "action_mask_required": True,
        "learning_mechanism": "bounded_parental_inheritance_with_mutation_v1",
        "mind_state_storage": "agent.mind_inheritance_metadata",
        "promotion_metric_family": "autonomous_survival_reproduction",
        "controller": {
            "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
            "runtime_backend": "pure_python_deterministic_v1",
            "feature_source": "mind_observation_v3_encoded_input",
            "architecture": MIND_V3_CONTROLLER_ARCHITECTURE,
            "feature_scope": "policy_visible_self_local_patch_navigation",
            "feature_fields": list(MIND_V3_CONTEXT_FEATURE_FIELDS),
            "derived_feature_fields": list(MIND_V3_NEED_GATED_FEATURE_FIELDS),
            "controller_private_fields_excluded_from_features": [
                "mind_inheritance_available"
            ],
            "inherited_parameters": "action_head_weights_and_bias",
        },
        "frozen_neural_artifact": {
            "schema_version": MIND_V3_NEURAL_ARTIFACT_SCHEMA_VERSION,
            "model_type": MIND_V3_NEURAL_MODEL_TYPE,
            "runtime_backend": MIND_V3_NEURAL_BACKEND,
            "architecture": MIND_V3_NEURAL_ARCHITECTURE,
            "input_contract": ecological_policy_input_contract(),
            "weights_mutable_during_run": False,
            "enabled_by_default": False,
            "promotion_status": "experiment_only",
        },
    }
