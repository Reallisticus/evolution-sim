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
