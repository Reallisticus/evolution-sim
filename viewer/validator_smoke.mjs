import { validateReplayPayload } from "./replay_validator.mjs";
import { REQUIRED_AGENT_FIELDS } from "./contracts.generated.mjs";

const emissionFields = [
  "field_name",
  "profile_id",
  "source_kind",
  "source_agent_id",
  "token_id",
  "profile_index",
  "x",
  "y",
  "intensity",
  "radius",
  "duration_ticks",
  "remaining_ticks",
  "decay_rate",
  "energy_cost",
  "emitted_tick",
];
const agentEncoding = [...REQUIRED_AGENT_FIELDS];
const coreActionKeys = ["stay", "eat", "drink"];
const movementActionKeys = ["move_north", "move_south", "move_east", "move_west"];
const attackActionKeys = movementActionKeys.map((action) =>
  action.replace("move_", "attack_"),
);
const communicationTokenCount = 4;
const communicationProfilesPerToken = 2;

function matrix(value) {
  return [
    [value, value],
    [value, value],
  ];
}

function frame(tick) {
  return {
    tick,
    agents: [agentRow(tick)],
    species_counts: [[1, 1]],
    species_metrics: {},
    field_state: {},
    signal_flow: {
      reproductive_emissions: 0,
      communication_emissions: 0,
      energy_spent: 0,
    },
    fresh_kill_energy_codes: matrix(0),
    carcass_energy_codes: matrix(0),
    carcass_freshness_codes: matrix(0),
    habitat_state_codes: matrix(0),
    hydrology_primary_codes: matrix(0),
    hydrology_support_codes: matrix(0),
    refuge_codes: matrix(0),
    refuge_score_codes: matrix(0),
    hazard_type_codes: matrix(0),
    hazard_level_codes: matrix(0),
    trophic_role_codes: matrix(1),
    meat_mode_codes: matrix(0),
    ecology_state_codes: matrix(0),
    biotic_fields: {
      prey_biomass: matrix(0),
      carrion: matrix(0),
      predator_risk: matrix(0),
    },
    signal_fields: {
      reproductive_signal: matrix(0),
      communication_signal: matrix(0),
    },
    signal_emissions: {
      schema_version: "foundation_signal_contract_v2",
      policy_visible: false,
      events: [],
      active_counts: {
        reproductive_signal: 0,
        communication_signal: 0,
      },
    },
  };
}

function agentRow(tick) {
  const values = {
    agent_id: 1,
    x: 0,
    y: 0,
    energy: 1,
    energy_ratio: 1,
    hydration: 1,
    hydration_ratio: 1,
    health: 1,
    health_ratio: 1,
    injury_load: 0,
    age: tick,
    energy_modifier: 1,
    hydration_modifier: 1,
    tile_vegetation: 0.5,
    tile_recovery_debt: 0,
    reproduction_ready: false,
    trophic_role: "herbivore",
    meat_mode: "none",
    last_damage_source: "none",
    water_access_reason: "none",
    soft_refuge_reason: "none",
    hydrology_support_code: 0,
    refuge_score: 0,
    matched_diet_ratio: 1,
    ecotype_id: null,
    species_id: 1,
  };
  return agentEncoding.map((field) => values[field] ?? null);
}

function validPayload() {
  return {
    run_id: "validator-smoke",
    config: {},
    summary: {
      run_id: "validator-smoke",
      ticks_executed: 2,
    },
    events: [],
    viewer: {
      map: {
        width: 2,
        height: 2,
        terrain_codes: matrix(0),
        base_tile_fields: {
          fertility: matrix(0.5),
          moisture: matrix(0.5),
          heat: matrix(0.5),
        },
      },
      frames: [frame(0), frame(1)],
      agent_encoding: agentEncoding,
      agent_catalog: agentCatalog(),
      reproductive_group_catalog: reproductiveGroupCatalog(),
      species_catalog: {},
      ecotype_catalog: {},
      trajectory: {
        schema_version: "mind_trajectory_v1",
        observation_schema_version: "mind_observation_v3",
        policy_interface_version: "mind_policy_interface_v1",
        action_contract_version: "mind_action_contract_v1",
        reproductive_group_contract_version: "reproductive_group_contract_v1",
        genome_recombination_contract_version: "genome_recombination_contract_v1",
        reward_schema_version: "mind_reward_v1",
        action_outcome_schema_version: "mind_action_outcome_v2",
        action_contract: actionContract(),
        reproductive_group_contract: {
          schema_version: "reproductive_group_contract_v1",
        },
        genome_recombination_contract: {
          schema_version: "genome_recombination_contract_v1",
        },
        reward_contract: {
          schema_version: "mind_reward_v1",
        },
        observation_contract: {
          schema_version: "mind_observation_v3",
          policy_input: observationPolicyInput(),
          action_names: actionNames(),
          action_contract: actionContract(),
          signal_contract: signalContract(),
        },
        records: [
          {
            tick: 0,
            agent_id: 1,
            lineage_id: 1,
            runtime_species_id: 1,
            runtime_ecotype_id: null,
            observation_schema: "mind_observation_v3",
            observation_metadata: { agent_id: 1 },
            observation_input: observationInput(),
            observation_digest: "0".repeat(64),
            action_mask: {},
            resolution_action_mask: {},
            requested_action: "stay",
            action_source: "smoke",
            policy_id: "smoke",
            policy_version: "v0",
            action_valid: true,
            resolution_action_valid: true,
            resolved_action: "stay",
            moved: false,
            before: {},
            after: {},
            outcome: {
              schema_version: "mind_action_outcome_v2",
              signal: signalOutcome(),
            },
            reward: { schema_version: "mind_reward_v1" },
          },
        ],
      },
    },
  };
}

function clone(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function expectError(name, mutate, expectedMessage) {
  const payload = clone(validPayload());
  mutate(payload);
  try {
    validateReplayPayload(payload);
  } catch (error) {
    if (!String(error.message).includes(expectedMessage)) {
      throw new Error(`${name}: expected ${expectedMessage}, got ${error.message}`);
    }
    return;
  }
  throw new Error(`${name}: expected validation failure`);
}

function expectValid(name, mutate) {
  const payload = clone(validPayload());
  mutate(payload);
  try {
    validateReplayPayload(payload);
  } catch (error) {
    throw new Error(`${name}: expected validation success, got ${error.message}`);
  }
}

validateReplayPayload(validPayload());

expectValid("valid multi-offspring sibling group", (payload) => {
  payload.viewer.agent_catalog["4"] = {
    agent_id: 4,
    parent_id: null,
    secondary_parent_id: null,
    parent_ids: [],
    lineage_id: 1,
    reproductive_group_id: 1,
    reproductive_stage: "stage0_asexual",
    reproductive_expression: "asexual",
    birth_tick: 0,
    death_tick: 1,
    genome: {},
    mind_inheritance: { schema_version: "mind_inheritance_placeholder_v1" },
  };
  for (const childId of [2, 3]) {
    payload.viewer.agent_catalog[String(childId)] = {
      agent_id: childId,
      parent_id: 1,
      secondary_parent_id: 4,
      parent_ids: [1, 4],
      lineage_id: 1,
      reproductive_group_id: 1,
      reproductive_stage: "stage0_asexual",
      reproductive_expression: "asexual",
      birth_tick: 1,
      death_tick: 1,
      genome: {},
      mind_inheritance: { schema_version: "mind_inheritance_placeholder_v1" },
    };
  }
  payload.viewer.reproductive_group_catalog.groups["1"].member_count = 4;
  payload.viewer.reproductive_group_catalog.groups["1"].sexual_births = 2;
  for (const [offspringIndex, childId] of [2, 3].entries()) {
    payload.events.push({
      tick: 1,
      type: "agent_reproduced",
      agent_id: 1,
      data: {
        schema_version: "reproduction_event_v1",
        child_id: childId,
        child_reproductive_group_id: 1,
        reproduction_mode: "same_group_sexual",
        offspring_count: 2,
        multi_offspring_desired_count: 2,
        multi_offspring_actual_count: 2,
        multi_offspring_limit_reasons: [],
        offspring_index: offspringIndex + 1,
        sibling_child_ids: [2, 3],
        multi_offspring: true,
        parent_ids: [1, 4],
        parent_energy_costs_total: [
          { agent_id: 1, energy_cost: 0.2 },
          { agent_id: 4, energy_cost: 0.2 },
        ],
        hybrid: false,
      },
    });
  }
});

expectError(
  "ragged terrain",
  (payload) => {
    payload.viewer.map.terrain_codes[1] = [0];
  },
  "must have 2 columns",
);
expectError(
  "missing signal emission metadata",
  (payload) => {
    delete payload.viewer.frames[0].signal_emissions;
  },
  "signal_emissions",
);
expectError(
  "stale signal schema",
  (payload) => {
    payload.viewer.frames[0].signal_emissions.schema_version = "stale";
  },
  "schema version does not match",
);
expectError(
  "bad agent row length",
  (payload) => {
    payload.viewer.frames[0].agents[0].pop();
  },
  "length must match agent_encoding",
);
expectError(
  "nonmonotonic ticks",
  (payload) => {
    payload.viewer.frames[1].tick = 0;
  },
  "strictly increasing",
);
expectError(
  "missing trajectory observation input",
  (payload) => {
    delete payload.viewer.trajectory.records[0].observation_input;
  },
  "observation_input",
);
expectError(
  "stale trajectory observation input encoder",
  (payload) => {
    payload.viewer.trajectory.records[0].observation_input.encoder_version = "stale";
  },
  "mind_observation_encoder_v2",
);
expectError(
  "stale trajectory observation input shape",
  (payload) => {
    payload.viewer.trajectory.records[0].observation_input.shape = [541];
  },
  "shape",
);
expectError(
  "mismatched action signal capacity",
  (payload) => {
    payload.viewer.trajectory.action_contract.communication.token_count = 3;
  },
  "must match signal contract capacity",
);
expectError(
  "stale action contract version",
  (payload) => {
    payload.viewer.trajectory.action_contract.schema_version = "stale";
  },
  "mind_action_contract_v1",
);
expectError(
  "mismatched communication enablement",
  (payload) => {
    payload.viewer.trajectory.action_contract.communication.emission_enabled = true;
  },
  "communication.emission_enabled",
);
expectError(
  "active communication action while disabled",
  (payload) => {
    payload.viewer.trajectory.action_contract.active_action_keys.push(
      "signal_0_profile_0",
    );
  },
  "active_action_keys",
);
expectError(
  "stale trajectory observation version",
  (payload) => {
    payload.viewer.trajectory.observation_schema_version = "stale";
  },
  "mind_observation_v3",
);
expectError(
  "stale observation policy input encoder",
  (payload) => {
    payload.viewer.trajectory.observation_contract.policy_input.encoder_version = "stale";
  },
  "mind_observation_encoder_v2",
);
expectError(
  "stale observation policy input shape",
  (payload) => {
    payload.viewer.trajectory.observation_contract.policy_input.shape = [541];
  },
  "policy_input.shape",
);
expectError(
  "stale record outcome version",
  (payload) => {
    payload.viewer.trajectory.records[0].outcome.schema_version = "stale";
  },
  "mind_action_outcome_v2",
);
expectError(
  "stale later record outcome version",
  (payload) => {
    const secondRecord = JSON.parse(
      JSON.stringify(payload.viewer.trajectory.records[0]),
    );
    secondRecord.tick = 1;
    secondRecord.outcome.schema_version = "stale";
    payload.viewer.trajectory.records.push(secondRecord);
  },
  "mind_action_outcome_v2",
);
expectError(
  "stale record reward version",
  (payload) => {
    payload.viewer.trajectory.records[0].reward.schema_version = "stale";
  },
  "mind_reward_v1",
);
expectError(
  "missing reproductive group catalog",
  (payload) => {
    delete payload.viewer.reproductive_group_catalog;
  },
  "reproductive_group_catalog",
);
expectError(
  "mismatched reproductive group member count",
  (payload) => {
    payload.viewer.reproductive_group_catalog.groups["1"].member_count = 2;
  },
  "member_count does not match agent catalog",
);
expectError(
  "mismatched reproductive group birth event count",
  (payload) => {
    payload.events.push({
      tick: 0,
      type: "agent_reproduced",
      agent_id: 1,
      data: {
        schema_version: "reproduction_event_v1",
        child_id: 2,
        child_reproductive_group_id: 1,
        reproduction_mode: "asexual",
        offspring_count: 1,
        hybrid: false,
      },
    });
  },
  "asexual_births does not match agent_reproduced events",
);
expectError(
  "stale reproductive birth event schema",
  (payload) => {
    payload.events.push({
      tick: 0,
      type: "agent_reproduced",
      agent_id: 1,
      data: {
        schema_version: "stale",
        child_id: 2,
        child_reproductive_group_id: 1,
        reproduction_mode: "asexual",
        offspring_count: 1,
        hybrid: false,
      },
    });
  },
  "reproduction_event_v1",
);
expectError(
  "incomplete multi-offspring sibling group",
  (payload) => {
    payload.events.push({
      tick: 0,
      type: "agent_reproduced",
      agent_id: 1,
      data: {
        schema_version: "reproduction_event_v1",
        child_id: 2,
        child_reproductive_group_id: 1,
        reproduction_mode: "same_group_sexual",
        offspring_count: 2,
        offspring_index: 1,
        sibling_child_ids: [2, 3],
        multi_offspring: true,
        parent_ids: [1, 1],
        parent_energy_costs_total: [
          { agent_id: 1, energy_cost: 0.2 },
          { agent_id: 1, energy_cost: 0.2 },
        ],
        hybrid: false,
      },
    });
  },
  "multi-offspring sibling group",
);
expectError(
  "clamped multi-offspring missing limit reason",
  (payload) => {
    payload.events.push({
      tick: 0,
      type: "agent_reproduced",
      agent_id: 1,
      data: {
        schema_version: "reproduction_event_v1",
        child_id: 2,
        child_reproductive_group_id: 1,
        reproduction_mode: "same_group_sexual",
        offspring_count: 1,
        multi_offspring_desired_count: 2,
        multi_offspring_actual_count: 1,
        multi_offspring_limit_reasons: [],
        hybrid: false,
      },
    });
  },
  "must include a limit reason",
);
expectError(
  "unknown reproductive expression",
  (payload) => {
    payload.viewer.agent_catalog["1"].reproductive_expression = "mystery";
  },
  "unknown reproductive expression",
);
expectError(
  "missing agent reproductive group",
  (payload) => {
    payload.viewer.agent_catalog["1"].reproductive_group_id = null;
  },
  "missing reproductive group",
);
expectError(
  "invalid agent reproductive group type",
  (payload) => {
    payload.viewer.agent_catalog["1"].reproductive_group_id = "1";
  },
  "reproductive_group_id must be an integer",
);
expectError(
  "invalid agent death tick",
  (payload) => {
    payload.viewer.agent_catalog["1"].death_tick = "later";
  },
  "death_tick must be an integer",
);
expectError(
  "mismatched reproductive group id",
  (payload) => {
    payload.viewer.reproductive_group_catalog.groups["1"].group_id = 2;
  },
  "mismatched id",
);
expectError(
  "missing signal outcome",
  (payload) => {
    delete payload.viewer.trajectory.records[0].outcome.signal;
  },
  "outcome.signal",
);

console.log("viewer_validator_smoke_ok malformed_cases=30");

function observationInput() {
  return {
    schema_version: "mind_observation_v3",
    encoder_version: "mind_observation_encoder_v2",
    decoded_dtype: "float32",
    storage_dtype: "int16",
    storage_encoding: "zlib_base64_little_endian_int16",
    shape: [542],
    value_range: [-1, 1],
    data: "smoke",
  };
}

function observationPolicyInput() {
  return {
    encoder_version: "mind_observation_encoder_v2",
    decoded_dtype: "float32",
    storage_dtype: "int16",
    storage_encoding: "zlib_base64_little_endian_int16",
    shape: [542],
    value_range: [-1, 1],
  };
}

function signalContract() {
  return {
    schema_version: "foundation_signal_contract_v2",
    signal_substrate_enabled: true,
    reproductive_signal_emission_enabled: true,
    communication_signal_emission_enabled: false,
    profile_metadata_policy_visible: false,
    emission_debug_metadata_fields: emissionFields,
    communication_token_count: communicationTokenCount,
    communication_profiles_per_token: communicationProfilesPerToken,
    reserved_profiles: reservedProfiles(),
  };
}

function signalOutcome() {
  return {
    emitted: false,
    token_id: null,
    profile_index: null,
    intensity: 0,
    radius: 0,
    duration_ticks: 0,
    decay_rate: 0,
    energy_cost: 0,
    invalid_reason: null,
  };
}

function actionContract() {
  const communicationActions = communicationActionKeys();
  return {
    schema_version: "mind_action_contract_v1",
    active_action_keys: [
      ...coreActionKeys,
      ...movementActionKeys,
      ...attackActionKeys,
    ],
    communication: {
      token_count: communicationTokenCount,
      profiles_per_token: communicationProfilesPerToken,
      action_keys: communicationActions,
      emission_enabled: false,
    },
    reserved_action_keys: ["mate", ...communicationActions],
  };
}

function actionNames() {
  return [
    ...coreActionKeys,
    ...movementActionKeys,
    ...attackActionKeys,
    "mate",
    ...communicationActionKeys(),
  ];
}

function communicationActionKeys() {
  const keys = [];
  for (let tokenIndex = 0; tokenIndex < communicationTokenCount; tokenIndex += 1) {
    for (
      let profileIndex = 0;
      profileIndex < communicationProfilesPerToken;
      profileIndex += 1
    ) {
      keys.push(`signal_${tokenIndex}_profile_${profileIndex}`);
    }
  }
  return keys;
}

function agentCatalog() {
  return {
    "1": {
      agent_id: 1,
      parent_id: null,
      secondary_parent_id: null,
      parent_ids: [],
      lineage_id: 1,
      reproductive_group_id: 1,
      reproductive_stage: "stage0_asexual",
      reproductive_expression: "asexual",
      birth_tick: 0,
      death_tick: null,
      genome: {},
      mind_inheritance: { schema_version: "mind_inheritance_placeholder_v1" },
    },
  };
}

function reproductiveGroupCatalog() {
  return {
    schema_version: "reproductive_group_contract_v1",
    groups: {
      "1": {
        group_id: 1,
        founder_lineage_id: 1,
        founder_agent_id: 1,
        created_tick: 0,
        last_seen_tick: 0,
        stage: "stage0_asexual",
        parent_group_ids: [],
        member_count: 1,
        alive_member_count: 1,
        alive_stage_counts: { stage0_asexual: 1 },
        alive_expression_counts: { asexual: 1 },
        asexual_births: 0,
        sexual_births: 0,
        hybrid_births: 0,
      },
    },
  };
}

function reservedProfiles() {
  const profiles = [];
  for (let tokenIndex = 0; tokenIndex < communicationTokenCount; tokenIndex += 1) {
    for (
      let profileIndex = 0;
      profileIndex < communicationProfilesPerToken;
      profileIndex += 1
    ) {
      profiles.push({
        field_name: "communication_signal",
        token_id: tokenIndex,
        profile_index: profileIndex,
        policy_visible: false,
      });
    }
  }
  return profiles;
}
