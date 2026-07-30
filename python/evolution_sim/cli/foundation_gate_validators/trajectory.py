from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from evolution_sim.config import SignalConfig
from evolution_sim.env.runtime.action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract as canonical_action_contract,
    action_names as canonical_action_names,
)
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    PATCH_FIELDS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_FIELDS,
    SELF_INPUT_FIELDS,
    TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    validate_observation_input_payload,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
)
from evolution_sim.env.runtime.signals import (
    COMMUNICATION_AGGREGATE_PROJECTION,
    COMMUNICATION_GLOBAL_REPORTING_POLICY,
    COMMUNICATION_RECEIVER_OBSERVATION_POLICY,
    COMMUNICATION_RECEIVER_PROJECTION_SCHEMA_VERSION,
    COMMUNICATION_SIGNAL_FIELD,
    SIGNAL_CONTRACT_VERSION,
    TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
    TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
)
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_COMPONENT_BOUNDS,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_RECORD_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
    TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
    reward_contract as canonical_reward_contract,
)
from evolution_sim.env.contracts import (
    SUMMARY_SCHEMA_VERSION,
    TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
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
    trajectory_payload = viewer.get("trajectory")
    tokenized_communication = (
        isinstance(trajectory_payload, dict)
        and trajectory_payload.get("schema_version")
        == TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION
    )
    expected_trajectory_schema = (
        TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION
        if tokenized_communication
        else TRAJECTORY_SCHEMA_VERSION
    )
    expected_observation_schema = (
        TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
        if tokenized_communication
        else OBSERVATION_SCHEMA_VERSION
    )
    expected_observation_encoder = (
        TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION
        if tokenized_communication
        else OBSERVATION_ENCODER_VERSION
    )
    expected_signal_schema = (
        TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION
        if tokenized_communication
        else SIGNAL_CONTRACT_VERSION
    )
    expected_summary_schema = (
        TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION
        if tokenized_communication
        else SUMMARY_SCHEMA_VERSION
    )
    if summary.get("summary_schema_version") != expected_summary_schema:
        flags.append(
            _flag(
                "error",
                scope,
                "summary.summary_schema_version",
                (
                    f"Expected {expected_summary_schema}, found "
                    f"{summary.get('summary_schema_version')!r}."
                ),
            )
        )
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
            "observation_schema_version": expected_observation_schema,
            "policy_interface_version": POLICY_INTERFACE_VERSION,
            "schema_version": expected_trajectory_schema,
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

    trajectory = trajectory_payload
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
        "schema_version": expected_trajectory_schema,
        "observation_schema_version": expected_observation_schema,
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
    expected_frame_signal_fields = [
        "reproductive_signal",
        "communication_signal",
    ]
    expected_record_token_fields: tuple[str, ...] = ()
    expected_observation_size = OBSERVATION_INPUT_VECTOR_SIZE
    observation_contract = trajectory.get("observation_contract")
    if (
        not isinstance(observation_contract, dict)
        or observation_contract.get("schema_version") != expected_observation_schema
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
        token_channel_contract = observation_contract.get(
            "communication_token_channels"
        )
        token_fields = (
            token_channel_contract.get("field_order")
            if isinstance(token_channel_contract, dict)
            else None
        )
        token_count = (
            len(token_fields)
            if tokenized_communication and isinstance(token_fields, list)
            else 0
        )
        if tokenized_communication:
            expected_record_token_fields = tuple(
                f"communication_signal_token_{token_index}"
                for token_index in range(token_count)
            )
            expected_frame_signal_fields.extend(
                expected_record_token_fields
            )
        expected_observation_size = (
            OBSERVATION_INPUT_VECTOR_SIZE
            + token_count * (1 + PATCH_CELL_COUNT)
        )
        if (
            not isinstance(policy_input, dict)
            or policy_input.get("encoder_version") != expected_observation_encoder
            or policy_input.get("shape") != [expected_observation_size]
            or (tokenized_communication and token_count <= 0)
            or (
                not tokenized_communication
                and token_channel_contract is not None
            )
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
            or signal_contract.get("schema_version") != expected_signal_schema
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
    frame_contract_valid = isinstance(frames, list) and bool(frames)
    if isinstance(frames, list):
        for frame in frames:
            signal_emissions = (
                frame.get("signal_emissions")
                if isinstance(frame, dict)
                else None
            )
            signal_fields = (
                frame.get("signal_fields")
                if isinstance(frame, dict)
                else None
            )
            active_counts = (
                signal_emissions.get("active_counts")
                if isinstance(signal_emissions, dict)
                else None
            )
            if (
                not isinstance(signal_emissions, dict)
                or signal_emissions.get("schema_version")
                != expected_signal_schema
                or signal_emissions.get("policy_visible") is not False
                or not isinstance(signal_emissions.get("events"), list)
                or not isinstance(active_counts, dict)
                or not isinstance(signal_fields, dict)
                or list(signal_fields) != expected_frame_signal_fields
                or list(active_counts) != expected_frame_signal_fields
                or any(
                    not isinstance(signal_fields[field], list)
                    for field in expected_frame_signal_fields
                )
            ):
                frame_contract_valid = False
                break
    if not frame_contract_valid:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.frames.signal_emissions",
                (
                    "Full replay frames are missing versioned signal emission "
                    "metadata or canonical signal-field matrices."
                ),
            )
        )
    records = trajectory.get("records")
    declared_reward_contract = trajectory.get("reward_contract")
    if declared_reward_contract != canonical_reward_contract():
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.reward_contract",
                (
                    "Trajectory payload reward contract does not exactly match "
                    "the canonical component set and bounds."
                ),
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
                expected_observation_schema=expected_observation_schema,
                expected_observation_encoder=expected_observation_encoder,
                expected_observation_size=expected_observation_size,
                expected_token_fields=expected_record_token_fields,
            )
        )
    return flags


def _trajectory_record_contract_flags(
    *,
    scope: str,
    record: object,
    index: int,
    expected_observation_schema: str = OBSERVATION_SCHEMA_VERSION,
    expected_observation_encoder: str = OBSERVATION_ENCODER_VERSION,
    expected_observation_size: int = OBSERVATION_INPUT_VECTOR_SIZE,
    expected_token_fields: tuple[str, ...] = (),
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

    if record.get("observation_schema") != expected_observation_schema:
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
        if (
            observation_input.get("schema_version")
            != expected_observation_schema
            or observation_input.get("encoder_version")
            != expected_observation_encoder
            or observation_input.get("shape") != [expected_observation_size]
            or (
                bool(expected_token_fields)
                and (
                    observation_input.get("communication_token_count")
                    != len(expected_token_fields)
                    or observation_input.get("communication_token_field_order")
                    != list(expected_token_fields)
                )
            )
            or (
                not expected_token_fields
                and (
                    "communication_token_count" in observation_input
                    or "communication_token_field_order" in observation_input
                )
            )
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.records.observation_input",
                    (
                        f"Trajectory record {index} observation input is not "
                        "bound to the enclosing observation contract."
                    ),
                )
            )
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
    reward_components = (
        reward.get("components") if isinstance(reward, dict) else None
    )
    if (
        not isinstance(reward, dict)
        or set(reward) != {"schema_version", "components", "total"}
        or reward.get("schema_version") != REWARD_SCHEMA_VERSION
        or not isinstance(reward_components, dict)
        or set(reward_components) != set(REWARD_COMPONENT_BOUNDS)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in (
                *reward_components.values(),
                reward.get("total"),
            )
        )
        or any(
            not REWARD_COMPONENT_BOUNDS[name][0]
            <= float(reward_components[name])
            <= REWARD_COMPONENT_BOUNDS[name][1]
            for name in REWARD_COMPONENT_BOUNDS
        )
        or round(
            sum(float(reward_components[name]) for name in REWARD_COMPONENT_BOUNDS),
            4,
        )
        != float(reward.get("total"))
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
    try:
        canonical_signal_config = SignalConfig(
            communication_token_count=token_count,
            communication_profiles_per_token=profiles_per_token,
            communication_signal_emission_enabled=communication_enabled,
        )
    except ValueError:
        return [
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.signal_contract",
                "Signal contract communication capacity is outside canonical bounds.",
            )
        ]

    expected_actions = [
        f"signal_{token_index}_profile_{profile_index}"
        for token_index in range(token_count)
        for profile_index in range(profiles_per_token)
    ]
    expected_action_contract = canonical_action_contract(canonical_signal_config)
    expected_action_names = list(canonical_action_names(canonical_signal_config))
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
    tokenized_communication = (
        signal_contract.get("schema_version")
        == TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION
    )
    expected_token_fields = [
        f"communication_signal_token_{token_index}"
        for token_index in range(token_count)
    ]
    signal_token_contract = signal_contract.get(
        "communication_token_observation"
    )
    observation_token_contract = observation_contract.get(
        "communication_token_channels"
    )
    policy_input = observation_contract.get("policy_input")
    expected_token_order = list(range(token_count))
    token_fields_are_ordered = (
        isinstance(policy_input, dict)
        and policy_input.get("patch_cell_count") == PATCH_CELL_COUNT
        and observation_contract.get("self_fields")
        == [*SELF_FIELDS, *expected_token_fields]
        and observation_contract.get("patch_fields")
        == [*PATCH_FIELDS, *expected_token_fields]
        and policy_input.get("self_input_fields")
        == [*SELF_INPUT_FIELDS, *expected_token_fields]
        and policy_input.get("patch_input_fields")
        == [*PATCH_INPUT_FIELDS, *expected_token_fields]
    )
    if tokenized_communication:
        if (
            communication_enabled is not True
            or signal_contract.get("policy_semantics") != "opaque"
            or signal_contract.get(
                "communication_tokens_have_simulator_assigned_meaning"
            )
            is not False
            or not isinstance(signal_token_contract, dict)
            or set(signal_token_contract)
            != {
                "schema_version",
                "policy_visible",
                "spatial",
                "field_order",
                "token_order",
                "aggregate_field",
                "aggregate_field_retained_for_compatibility",
                "aggregation",
                "receiver_projection_schema_version",
                "receiver_observation_policy",
                "global_reporting_policy",
                "simulator_assigned_meanings",
                "profile_provenance_policy_visible",
            }
            or signal_token_contract.get("schema_version")
            != TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION
            or signal_token_contract.get("policy_visible") is not True
            or signal_token_contract.get("spatial") is not True
            or signal_token_contract.get("field_order") != expected_token_fields
            or signal_token_contract.get("token_order") != expected_token_order
            or signal_token_contract.get("aggregate_field")
            != COMMUNICATION_SIGNAL_FIELD
            or signal_token_contract.get(
                "aggregate_field_retained_for_compatibility"
            )
            is not True
            or signal_token_contract.get("aggregation")
            != COMMUNICATION_AGGREGATE_PROJECTION
            or signal_token_contract.get("receiver_projection_schema_version")
            != COMMUNICATION_RECEIVER_PROJECTION_SCHEMA_VERSION
            or signal_token_contract.get("receiver_observation_policy")
            != COMMUNICATION_RECEIVER_OBSERVATION_POLICY
            or signal_token_contract.get("global_reporting_policy")
            != COMMUNICATION_GLOBAL_REPORTING_POLICY
            or signal_token_contract.get("simulator_assigned_meanings") is not False
            or signal_token_contract.get("profile_provenance_policy_visible")
            is not False
            or not isinstance(observation_token_contract, dict)
            or set(observation_token_contract)
            != {
                "policy_visible",
                "spatial",
                "field_order",
                "token_order",
                "simulator_assigned_meanings",
                "profile_provenance_policy_visible",
                "aggregate_communication_field_retained",
                "aggregate_projection",
            }
            or observation_token_contract.get("policy_visible") is not True
            or observation_token_contract.get("spatial") is not True
            or observation_token_contract.get("field_order")
            != expected_token_fields
            or observation_token_contract.get("token_order")
            != expected_token_order
            or observation_token_contract.get("simulator_assigned_meanings")
            is not False
            or observation_token_contract.get(
                "profile_provenance_policy_visible"
            )
            is not False
            or observation_token_contract.get(
                "aggregate_communication_field_retained"
            )
            is not True
            or observation_token_contract.get("aggregate_projection")
            != COMMUNICATION_AGGREGATE_PROJECTION
            or not token_fields_are_ordered
            or signal_contract.get("fields")
            != [
                "reproductive_signal",
                "communication_signal",
                *expected_token_fields,
            ]
        ):
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.communication_token_channels",
                    (
                        "Token-aware signal and observation contracts do not "
                        "declare the same ordered opaque public channels."
                    ),
                )
            )
        token_profile_fields_match = isinstance(reserved_profiles, list)
        if isinstance(reserved_profiles, list):
            for profile_index, profile in enumerate(reserved_profiles):
                if not isinstance(profile, dict):
                    token_profile_fields_match = False
                    break
                token_id = _as_optional_int(profile.get("token_id"))
                expected_token_id = profile_index // profiles_per_token
                expected_profile_index = profile_index % profiles_per_token
                if (
                    token_id is None
                    or token_id < 0
                    or token_id >= len(expected_token_fields)
                    or token_id != expected_token_id
                    or profile.get("profile_index") != expected_profile_index
                    or profile.get("field_name")
                    != expected_token_fields[token_id]
                    or profile.get("policy_visible") is not False
                ):
                    token_profile_fields_match = False
                    break
        if not token_profile_fields_match:
            flags.append(
                _flag(
                    "error",
                    scope,
                    "viewer.trajectory.observation_contract.signal_contract.reserved_profiles",
                    "Token-aware communication profiles do not target their opaque token channels.",
                )
            )
    elif (
        signal_token_contract is not None
        or observation_token_contract is not None
    ):
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.communication_token_channels",
                "Legacy observation contracts must not declare token channels.",
            )
        )

    for path, action_contract in (
        ("viewer.trajectory.action_contract", trajectory.get("action_contract")),
        (
            "viewer.trajectory.observation_contract.action_contract",
            observation_contract.get("action_contract"),
        ),
    ):
        if action_contract != expected_action_contract:
            flags.append(
                _flag(
                    "error",
                    scope,
                    path,
                    "Action contract communication slots do not match signal contract capacity.",
                )
            )

    action_names = observation_contract.get("action_names")
    if action_names != expected_action_names:
        flags.append(
            _flag(
                "error",
                scope,
                "viewer.trajectory.observation_contract.action_names",
                "Observation action names do not include every reserved communication slot.",
            )
        )
    return flags
