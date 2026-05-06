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

MIND_V1_DATA_CONTRACT_VERSION = "mind_v1_data_contract_v1"
MIND_MODEL_ARTIFACT_VERSION = "mind_model_artifact_v1"
MIND_ONLINE_LEARNING_CONTRACT_VERSION = "mind_online_learning_contract_v1"
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
        "current_executable_slice": "learned_policy_trajectory_collection_v1",
        "trajectory_collection": {
            "heuristic_policy": "sim:trajectory",
            "learned_policy": (
                "sim:trajectory -- --mind-artifact <artifact> --enable-mind"
            ),
            "requires_explicit_mind_enable": True,
            "summary_only": True,
            "full_replay_required": False,
        },
        "algorithm_ladder": [
            {
                "stage": "neural_behavior_cloning_actor_critic",
                "status": "planned",
                "purpose": (
                    "Train a deterministic neural policy and value head from "
                    "the existing trajectory bank before any online updates."
                ),
                "families": ["supervised_bc", "actor_critic_value_head"],
            },
            {
                "stage": "conservative_offline_rl",
                "status": "planned",
                "purpose": (
                    "Improve beyond behavior cloning while limiting "
                    "out-of-distribution actions."
                ),
                "families": ["iql", "cql", "td3_bc", "rebrac"],
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
