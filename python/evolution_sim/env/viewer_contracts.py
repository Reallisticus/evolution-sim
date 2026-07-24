from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolution_sim.env.contracts import (
    SUMMARY_SCHEMA_VERSION,
    TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
    VIEWER_AGENT_ENCODING,
)
from evolution_sim.env.runtime.action_contract import (
    ACTION_CONTRACT_VERSION,
    action_contract,
)
from evolution_sim.env.runtime.mating import (
    ASEXUAL_REPRODUCTION_MODE,
    REPRODUCTIVE_STAGE_ORDER,
    SEXUAL_REPRODUCTION_MODE,
)
from evolution_sim.env.runtime.observations import (
    OBSERVATION_ENCODER_VERSION,
    OBSERVATION_INPUT_DTYPE,
    OBSERVATION_INPUT_VALUE_RANGE,
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    PATCH_CELL_COUNT,
    PATCH_FIELDS,
    PATCH_INPUT_FIELDS,
    SELF_FIELDS,
    SELF_INPUT_FIELDS,
    OBSERVATION_STORAGE_DTYPE,
    OBSERVATION_STORAGE_ENCODING,
    REPRODUCTIVE_EXPRESSION_VOCAB,
    TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
)
from evolution_sim.env.runtime.reproduction import (
    REPRODUCTION_EVENT_SCHEMA_VERSION,
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
)
from evolution_sim.env.runtime.signals import (
    COMMUNICATION_AGGREGATE_PROJECTION,
    SIGNAL_CONTRACT_VERSION,
    TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
    TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
    signal_contract,
)
from evolution_sim.env.runtime.trajectory import (
    ACTION_OUTCOME_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    TRAJECTORY_RECORD_FIELDS,
    TRAJECTORY_SCHEMA_VERSION,
    TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
    reward_contract,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reporting_contracts import BIOTIC_FIELD_NAMES
from evolution_sim.genome.recombination import GENOME_RECOMBINATION_CONTRACT_VERSION


REQUIRED_MAP_FIELDS: tuple[str, ...] = ("fertility", "moisture", "heat")
REQUIRED_AGENT_CATALOG_FIELDS: tuple[str, ...] = (
    "agent_id",
    "parent_id",
    "secondary_parent_id",
    "parent_ids",
    "lineage_id",
    "reproductive_group_id",
    "reproductive_stage",
    "reproductive_expression",
    "birth_tick",
    "death_tick",
    "genome",
    "mind_inheritance",
)
REQUIRED_REPRODUCTIVE_GROUP_FIELDS: tuple[str, ...] = (
    "group_id",
    "founder_lineage_id",
    "founder_agent_id",
    "created_tick",
    "last_seen_tick",
    "stage",
    "parent_group_ids",
    "member_count",
    "alive_member_count",
    "alive_stage_counts",
    "alive_expression_counts",
    "asexual_births",
    "sexual_births",
    "hybrid_births",
)
REQUIRED_FRAME_MATRIX_FIELDS: tuple[str, ...] = (
    "fresh_kill_energy_codes",
    "carcass_energy_codes",
    "carcass_freshness_codes",
    "habitat_state_codes",
    "hydrology_primary_codes",
    "hydrology_support_codes",
    "refuge_codes",
    "refuge_score_codes",
    "hazard_type_codes",
    "hazard_level_codes",
    "trophic_role_codes",
    "meat_mode_codes",
    "ecology_state_codes",
)
REQUIRED_SIGNAL_OUTCOME_FIELDS: tuple[str, ...] = (
    "emitted",
    "token_id",
    "profile_index",
    "intensity",
    "radius",
    "duration_ticks",
    "decay_rate",
    "energy_cost",
    "invalid_reason",
)
REWARD_RECORD_FIELDS: tuple[str, ...] = (
    "schema_version",
    "components",
    "total",
)
VIEWER_EMPTY_LABELS: dict[str, str] = {
    "missing": "-",
    "unknown": "Unknown",
    "not_applicable": "Not applicable",
    "still_alive": "Still alive",
    "no_recorded_death": "No recorded death",
}
VIEWER_DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "trophic_role": {
        "none": "No trophic role",
        "herbivore": "Herbivore",
        "omnivore": "Omnivore",
        "carnivore": "Carnivore",
    },
    "meat_mode": {
        "none": "Plant-focused",
        "scavenger": "Scavenger",
        "hunter": "Hunter",
        "mixed": "Mixed feeder",
    },
    "water_access_reason": {
        "none": "No water access",
        "adjacent_water": "Adjacent water",
        "wetland": "Wetland substrate",
        "flooded": "Flooded support",
    },
    "soft_refuge_reason": {
        "none": "No soft refuge",
        "canopy_refuge": "Canopy refuge",
    },
    "hazard_type": {
        "none": "No active hazard",
        "exposure": "Exposure hazard",
        "instability": "Instability hazard",
    },
    "habitat_state": {
        "non_land": "Non-land",
        "stable": "Stable habitat",
        "bloom": "Bloom habitat",
        "flooded": "Flooded habitat",
        "parched": "Parched habitat",
    },
    "ecology_state": {
        "stable": "Stable ecology",
        "lush": "Lush ecology",
        "recovering": "Recovering ecology",
        "depleted": "Depleted ecology",
    },
    "damage_source": {
        "none": "No recent damage",
        "attack": "Attack wound",
        "hazard_exposure": "Exposure hazard",
        "hazard_instability": "Instability hazard",
        "health_depletion": "Health depleted",
    },
    "food_source": {
        "plant": "Vegetation",
        "carcass": "Carcass",
        "fresh_kill": "Fresh kill",
    },
    "reproduction_mode": {
        ASEXUAL_REPRODUCTION_MODE: "Asexual birth",
        SEXUAL_REPRODUCTION_MODE: "Paired birth",
    },
    "death_cause": {
        "unknown": "Unknown death cause",
        "attack": "Killed by attack",
        "hazard_exposure": "Exposure hazard",
        "hazard_instability": "Instability hazard",
        "health_depletion": "Health depleted",
        "energy_depletion": "Energy depleted",
        "hydration_depletion": "Hydration depleted",
        "old_age": "Old age",
    },
}


def viewer_contract_payload() -> dict[str, Any]:
    signal_payload = signal_contract()
    action_payload = action_contract()
    communication_action_keys = set(action_payload["communication"]["action_keys"])
    action_specs = action_payload["actions"]
    communication_action_spec = next(
        spec for spec in action_specs if spec["key"] in communication_action_keys
    )
    return {
        "REQUIRED_AGENT_FIELDS": list(VIEWER_AGENT_ENCODING),
        "SUMMARY_SCHEMA_VERSION": SUMMARY_SCHEMA_VERSION,
        "TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION": (
            TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION
        ),
        "TRAJECTORY_SCHEMA_VERSION": TRAJECTORY_SCHEMA_VERSION,
        "TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION": (
            TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION
        ),
        "OBSERVATION_SCHEMA_VERSION": OBSERVATION_SCHEMA_VERSION,
        "TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION": (
            TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
        ),
        "OBSERVATION_ENCODER_VERSION": OBSERVATION_ENCODER_VERSION,
        "TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION": (
            TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION
        ),
        "OBSERVATION_INPUT_DECODED_DTYPE": OBSERVATION_INPUT_DTYPE,
        "OBSERVATION_INPUT_STORAGE_DTYPE": OBSERVATION_STORAGE_DTYPE,
        "OBSERVATION_INPUT_STORAGE_ENCODING": OBSERVATION_STORAGE_ENCODING,
        "OBSERVATION_INPUT_VECTOR_SIZE": OBSERVATION_INPUT_VECTOR_SIZE,
        "OBSERVATION_INPUT_VALUE_RANGE": list(OBSERVATION_INPUT_VALUE_RANGE),
        "PATCH_CELL_COUNT": PATCH_CELL_COUNT,
        "OBSERVATION_SELF_FIELDS": list(SELF_FIELDS),
        "OBSERVATION_PATCH_FIELDS": list(PATCH_FIELDS),
        "OBSERVATION_SELF_INPUT_FIELDS": list(SELF_INPUT_FIELDS),
        "OBSERVATION_PATCH_INPUT_FIELDS": list(PATCH_INPUT_FIELDS),
        "POLICY_INTERFACE_VERSION": POLICY_INTERFACE_VERSION,
        "ACTION_CONTRACT_VERSION": ACTION_CONTRACT_VERSION,
        "ACTION_CONTRACT_FIELDS": list(action_payload),
        "ACTION_COMMUNICATION_FIELDS": list(action_payload["communication"]),
        "ACTION_SPEC_FIELDS": list(communication_action_spec),
        "ACTION_NON_COMMUNICATION_SPECS": [
            spec
            for spec in action_specs
            if spec["key"] not in communication_action_keys
        ],
        "ACTION_COMMUNICATION_SPEC_FIXED_FIELDS": {
            field: value
            for field, value in communication_action_spec.items()
            if field not in {"action_id", "key", "active", "debug_label"}
        },
        "ACTION_POLICY_ID_ENCODING": action_payload["policy_id_encoding"],
        "ACTION_DEBUG_KEY_ENCODING": action_payload["debug_key_encoding"],
        "ACTION_MATE_ACTION_KEY": action_payload["mate_action_key"],
        "ACTION_COMMUNICATION_MEANING": action_payload["communication"]["meaning"],
        "SIGNAL_CONTRACT_VERSION": SIGNAL_CONTRACT_VERSION,
        "TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION": (
            TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION
        ),
        "TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION": (
            TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION
        ),
        "COMMUNICATION_AGGREGATE_PROJECTION": COMMUNICATION_AGGREGATE_PROJECTION,
        "REPRODUCTIVE_GROUP_CONTRACT_VERSION": (
            REPRODUCTIVE_GROUP_CONTRACT_VERSION
        ),
        "GENOME_RECOMBINATION_CONTRACT_VERSION": (
            GENOME_RECOMBINATION_CONTRACT_VERSION
        ),
        "REWARD_SCHEMA_VERSION": REWARD_SCHEMA_VERSION,
        "REWARD_CONTRACT": reward_contract(),
        "REWARD_RECORD_FIELDS": list(REWARD_RECORD_FIELDS),
        "ACTION_OUTCOME_SCHEMA_VERSION": ACTION_OUTCOME_SCHEMA_VERSION,
        "REPRODUCTION_EVENT_SCHEMA_VERSION": REPRODUCTION_EVENT_SCHEMA_VERSION,
        "ASEXUAL_REPRODUCTION_MODE": ASEXUAL_REPRODUCTION_MODE,
        "SEXUAL_REPRODUCTION_MODE": SEXUAL_REPRODUCTION_MODE,
        "REQUIRED_MAP_FIELDS": list(REQUIRED_MAP_FIELDS),
        "REQUIRED_AGENT_CATALOG_FIELDS": list(REQUIRED_AGENT_CATALOG_FIELDS),
        "REQUIRED_REPRODUCTIVE_GROUP_FIELDS": list(
            REQUIRED_REPRODUCTIVE_GROUP_FIELDS
        ),
        "REQUIRED_FRAME_MATRIX_FIELDS": list(REQUIRED_FRAME_MATRIX_FIELDS),
        "REQUIRED_BIOTIC_FIELDS": list(BIOTIC_FIELD_NAMES),
        "REQUIRED_SIGNAL_FIELDS": signal_payload["fields"],
        "REQUIRED_SIGNAL_EMISSION_FIELDS": (
            signal_payload["emission_debug_metadata_fields"]
        ),
        "REQUIRED_SIGNAL_OUTCOME_FIELDS": list(REQUIRED_SIGNAL_OUTCOME_FIELDS),
        "REQUIRED_TRAJECTORY_RECORD_FIELDS": list(TRAJECTORY_RECORD_FIELDS),
        "REPRODUCTIVE_STAGE_ORDER": list(REPRODUCTIVE_STAGE_ORDER),
        "REPRODUCTIVE_EXPRESSIONS": list(REPRODUCTIVE_EXPRESSION_VOCAB),
        "VIEWER_DISPLAY_LABELS": VIEWER_DISPLAY_LABELS,
        "VIEWER_EMPTY_LABELS": VIEWER_EMPTY_LABELS,
    }


def render_viewer_contract_module() -> str:
    lines = [
        "// Generated by python -m evolution_sim.cli.viewer_contracts.",
        "// Do not edit by hand.",
        "",
    ]
    for name, value in viewer_contract_payload().items():
        rendered_value = json.dumps(value, indent=2, sort_keys=False)
        lines.append(f"export const {name} = {rendered_value};")
        lines.append("")
    return "\n".join(lines)


def write_viewer_contract_module(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_viewer_contract_module())
