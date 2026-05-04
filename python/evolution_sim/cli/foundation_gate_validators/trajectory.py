from __future__ import annotations

from collections.abc import Mapping, Sequence

from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    validate_observation_input_payload,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
)
from evolution_sim.env.runtime.signals import SIGNAL_CONTRACT_VERSION
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_RECORD_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
)
from evolution_sim.genome.recombination import (
    GENOME_RECOMBINATION_CONTRACT_VERSION,
)
from evolution_sim.cli.foundation_gate_validators.common import (
    _as_optional_int,
    _flag,
)
from evolution_sim.cli.foundation_gate_validators.reproduction import (
    _reproductive_group_catalog_flags,
)


def _mind_contract_flags(
    *,
    scope: str,
    summary: dict[str, object],
    viewer: dict[str, object],
    events: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    contracts = summary.get("mind_contracts")
    if not isinstance(contracts, dict):
        flags.append(
            _flag(
                "error",
                scope,
                "summary.mind_contracts",
                "Full replay summary is missing Mind contract metadata.",
            )
        )
    else:
        expected_versions = {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "policy_interface_version": POLICY_INTERFACE_VERSION,
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
            "genome_recombination_contract_version": (
                GENOME_RECOMBINATION_CONTRACT_VERSION
            ),
        }
        for field, expected in expected_versions.items():
            if contracts.get(field) != expected:
                flags.append(
                    _flag(
                        "error",
                        scope,
                        f"summary.mind_contracts.{field}",
                        f"Expected {expected}, found {contracts.get(field)!r}.",
                    )
                )
        if int(contracts.get("record_count", 0)) <= 0:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "summary.mind_contracts.record_count",
                    "Full replay did not record any per-agent trajectory rows.",
                )
            )

    trajectory = viewer.get("trajectory")
    if not isinstance(trajectory, dict):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory",
                "Full replay viewer is missing trajectory payload.",
            )
        )
        return flags
    expected_trajectory_versions = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_interface_version": POLICY_INTERFACE_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "reproductive_group_contract_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "genome_recombination_contract_version": GENOME_RECOMBINATION_CONTRACT_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "action_outcome_schema_version": ACTION_OUTCOME_SCHEMA_VERSION,
    }
    for field, expected in expected_trajectory_versions.items():
        if trajectory.get(field) != expected:
            flags.append(
                _flag(
                    "error",
                    scope,
                    f"viewer.trajectory.{field}",
                    f"Expected {expected}, found {trajectory.get(field)!r}.",
                )
            )
    action_contract = trajectory.get("action_contract")
    if (
        not isinstance(action_contract, dict)
        or action_contract.get("schema_version") != ACTION_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.action_contract.schema_version",
                "Action contract metadata is missing or stale.",
            )
        )
    reproductive_group_contract = trajectory.get("reproductive_group_contract")
    if (
        not isinstance(reproductive_group_contract, dict)
        or reproductive_group_contract.get("schema_version")
        != REPRODUCTIVE_GROUP_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.reproductive_group_contract.schema_version",
                "Reproductive group contract metadata is missing or stale.",
            )
        )
    genome_recombination_contract = trajectory.get("genome_recombination_contract")
    if (
        not isinstance(genome_recombination_contract, dict)
        or genome_recombination_contract.get("schema_version")
        != GENOME_RECOMBINATION_CONTRACT_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.genome_recombination_contract.schema_version",
                "Genome recombination contract metadata is missing or stale.",
            )
        )
    reproductive_group_catalog = viewer.get("reproductive_group_catalog")
    if (
        not isinstance(reproductive_group_catalog, dict)
        or reproductive_group_catalog.get("schema_version")
        != REPRODUCTIVE_GROUP_CONTRACT_VERSION
        or not isinstance(reproductive_group_catalog.get("groups"), dict)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.reproductive_group_catalog",
                "Viewer reproductive group catalog is missing or stale.",
            )
        )
    else:
        flags.extend(
            _reproductive_group_catalog_flags(
                scope=scope,
                summary=summary,
                viewer=viewer,
                reproductive_group_catalog=reproductive_group_catalog,
                events=events,
            )
        )
    observation_contract = trajectory.get("observation_contract")
    if (
        not isinstance(observation_contract, dict)
        or observation_contract.get("schema_version") != OBSERVATION_SCHEMA_VERSION
        or observation_contract.get("privileged_world_state") is not False
        or observation_contract.get("metadata_policy_excluded") is not True
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract",
                "Observation contract metadata is missing or permits privileged world state.",
            )
        )
    else:
        policy_input = observation_contract.get("policy_input")
        if (
            not isinstance(policy_input, dict)
            or policy_input.get("encoder_version") != OBSERVATION_ENCODER_VERSION
            or policy_input.get("shape") != [OBSERVATION_INPUT_VECTOR_SIZE]
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.policy_input",
                    "Observation contract does not declare the canonical policy input encoder.",
                )
            )
        embedded_action_contract = observation_contract.get("action_contract")
        if (
            not isinstance(embedded_action_contract, dict)
            or embedded_action_contract.get("schema_version")
            != ACTION_CONTRACT_VERSION
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.action_contract.schema_version",
                    "Observation contract action metadata is missing or stale.",
                )
            )
        signal_contract = observation_contract.get("signal_contract")
        signal_metadata_fields = (
            signal_contract.get("emission_debug_metadata_fields")
            if isinstance(signal_contract, dict)
            else None
        )
        required_signal_metadata_fields = {
            "profile_id",
            "source_agent_id",
            "token_id",
            "profile_index",
            "radius",
            "decay_rate",
            "energy_cost",
        }
        if (
            not isinstance(signal_contract, dict)
            or signal_contract.get("schema_version") != SIGNAL_CONTRACT_VERSION
            or signal_contract.get("profile_metadata_policy_visible") is not False
            or not isinstance(signal_metadata_fields, list)
            or not required_signal_metadata_fields.issubset(signal_metadata_fields)
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.signal_contract",
                    "Signal contract does not declare debug-only profile/provenance metadata.",
                )
            )
        elif isinstance(observation_contract, dict):
            flags.extend(
                _signal_action_capacity_flags(
                    scope=scope,
                    trajectory=trajectory,
                    observation_contract=observation_contract,
                    signal_contract=signal_contract,
                )
            )
    frames = viewer.get("frames")
    last_frame = frames[-1] if isinstance(frames, list) and frames else None
    signal_emissions = (
        last_frame.get("signal_emissions")
        if isinstance(last_frame, dict)
        else None
    )
    if (
        not isinstance(signal_emissions, dict)
        or signal_emissions.get("schema_version") != SIGNAL_CONTRACT_VERSION
        or signal_emissions.get("policy_visible") is not False
        or not isinstance(signal_emissions.get("events"), list)
        or not isinstance(signal_emissions.get("active_counts"), dict)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.frames.signal_emissions",
                "Full replay frames are missing versioned signal emission debug metadata.",
            )
        )
    records = trajectory.get("records")
    reward_contract = trajectory.get("reward_contract")
    if (
        not isinstance(reward_contract, dict)
        or reward_contract.get("schema_version") != REWARD_SCHEMA_VERSION
        or not isinstance(reward_contract.get("component_bounds"), dict)
        or not isinstance(reward_contract.get("total_bounds"), list)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.reward_contract",
                "Trajectory payload does not declare versioned reward component bounds.",
            )
        )
    if not isinstance(records, list) or not records:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records",
                "Trajectory payload has no decision records.",
            )
        )
        return flags
    for index, record in enumerate(records):
        flags.extend(
            _trajectory_record_contract_flags(
                scope=scope,
                record=record,
                index=index,
            )
        )
    return flags


def _trajectory_record_contract_flags(
    *,
    scope: str,
    record: object,
    index: int,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    required_record_fields = set(TRAJECTORY_RECORD_FIELDS)
    if not isinstance(record, dict) or not required_record_fields.issubset(record):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records",
                (
                    f"Trajectory record {index} does not include the canonical "
                    "contract fields."
                ),
            )
        )
        return flags

    if record.get("observation_schema") != OBSERVATION_SCHEMA_VERSION:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.observation_schema",
                (
                    f"Trajectory record {index} does not declare the current "
                    "observation schema."
                ),
            )
        )
    observation_metadata = record.get("observation_metadata")
    if (
        not isinstance(observation_metadata, dict)
        or observation_metadata.get("agent_id") != record.get("agent_id")
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.observation_metadata",
                (
                    f"Trajectory record {index} does not include policy-excluded "
                    "observation metadata."
                ),
            )
        )
    observation_input = record.get("observation_input")
    if not isinstance(observation_input, dict):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.observation_input",
                f"Trajectory record {index} does not include encoded observation inputs.",
            )
        )
    else:
        validation_errors = validate_observation_input_payload(observation_input)
        if validation_errors:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.records.observation_input",
                    f"Trajectory record {index}: {validation_errors[0]}",
                )
            )
    reward = record.get("reward")
    if (
        not isinstance(reward, dict)
        or reward.get("schema_version") != REWARD_SCHEMA_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.reward",
                f"Trajectory record {index} does not include versioned reward components.",
            )
        )
    outcome = record.get("outcome")
    if (
        not isinstance(outcome, dict)
        or outcome.get("schema_version") != ACTION_OUTCOME_SCHEMA_VERSION
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.records.outcome",
                f"Trajectory record {index} does not include a versioned action outcome.",
            )
        )
    if isinstance(outcome, dict):
        signal_outcome = outcome.get("signal")
        required_signal_outcome_fields = {
            "emitted",
            "token_id",
            "profile_index",
            "intensity",
            "radius",
            "duration_ticks",
            "decay_rate",
            "energy_cost",
            "invalid_reason",
        }
        if (
            not isinstance(signal_outcome, dict)
            or not isinstance(signal_outcome.get("emitted"), bool)
            or not required_signal_outcome_fields.issubset(signal_outcome)
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.records.outcome.signal",
                    (
                        f"Trajectory record {index} action outcome must include "
                        "stable signal outcome metadata."
                    ),
                )
            )
    return flags

def _signal_action_capacity_flags(
    *,
    scope: str,
    trajectory: dict[str, object],
    observation_contract: dict[str, object],
    signal_contract: dict[str, object],
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    token_count = _as_optional_int(signal_contract.get("communication_token_count"))
    profiles_per_token = _as_optional_int(
        signal_contract.get("communication_profiles_per_token")
    )
    communication_enabled = signal_contract.get(
        "communication_signal_emission_enabled"
    )
    reserved_profiles = signal_contract.get("reserved_profiles")
    if (
        token_count is None
        or profiles_per_token is None
        or not isinstance(communication_enabled, bool)
    ):
        return [
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.signal_contract",
                "Signal contract does not declare communication capacity and enablement.",
            )
        ]

    expected_actions = [
        f"signal_{token_index}_profile_{profile_index}"
        for token_index in range(token_count)
        for profile_index in range(profiles_per_token)
    ]
    if (
        not isinstance(reserved_profiles, list)
        or len(reserved_profiles) != len(expected_actions)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.signal_contract.reserved_profiles",
                "Signal contract reserved profile count does not match communication capacity.",
            )
        )

    for path, action_contract in (
        ("viewer.trajectory.action_contract", trajectory.get("action_contract")),
        (
            "viewer.trajectory.observation_contract.action_contract",
            observation_contract.get("action_contract"),
        ),
    ):
        if not _action_contract_matches_signal_capacity(
            action_contract,
            token_count=token_count,
            profiles_per_token=profiles_per_token,
            expected_actions=expected_actions,
            communication_enabled=communication_enabled,
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    path,
                    "Action contract communication slots do not match signal contract capacity.",
                )
            )

    action_names = observation_contract.get("action_names")
    if not isinstance(action_names, list) or not set(expected_actions).issubset(
        set(action_names)
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.action_names",
                "Observation action names do not include every reserved communication slot.",
            )
        )
    return flags


def _action_contract_matches_signal_capacity(
    action_contract: object,
    *,
    token_count: int,
    profiles_per_token: int,
    expected_actions: list[str],
    communication_enabled: bool,
) -> bool:
    if not isinstance(action_contract, dict):
        return False
    communication = action_contract.get("communication")
    if not isinstance(communication, dict):
        return False
    if communication.get("emission_enabled") is not communication_enabled:
        return False
    if communication.get("token_count") != token_count:
        return False
    if communication.get("profiles_per_token") != profiles_per_token:
        return False
    if communication.get("action_keys") != expected_actions:
        return False
    reserved_actions = action_contract.get("reserved_action_keys")
    if not isinstance(reserved_actions, list) or not set(expected_actions).issubset(
        set(reserved_actions)
    ):
        return False
    active_actions = action_contract.get("active_action_keys")
    if not isinstance(active_actions, list):
        return False
    active_set = set(active_actions)
    expected_set = set(expected_actions)
    if communication_enabled:
        return expected_set.issubset(active_set)
    return active_set.isdisjoint(expected_set)
