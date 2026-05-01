import { validateReplayPayload } from "./replay_validator.mjs";

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
const agentEncoding = [
  "agent_id",
  "x",
  "y",
  "energy",
  "hydration",
  "health",
  "health_ratio",
  "injury_load",
  "age",
  "energy_modifier",
  "hydration_modifier",
  "trophic_role",
  "last_damage_source",
  "water_access_reason",
  "species_id",
];
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
    agents: [[1, 0, 0, 1, 1, 1, 1, 0, tick, 1, 1, "herbivore", "none", "none", 1]],
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
      agent_catalog: {},
      species_catalog: {},
      ecotype_catalog: {},
      trajectory: {
        schema_version: "mind_trajectory_v1",
        action_contract: actionContract(),
        observation_contract: {
          schema_version: "mind_observation_v2",
          action_names: actionNames(),
          action_contract: actionContract(),
          signal_contract: signalContract(),
        },
        records: [
          {
            agent_id: 1,
            observation_metadata: { agent_id: 1 },
            observation_input: {},
            observation_digest: "0".repeat(64),
            action_mask: {},
            resolution_action_mask: {},
            requested_action: "stay",
            policy_id: "smoke",
            policy_version: "v0",
            action_valid: true,
            resolution_action_valid: true,
            resolved_action: "stay",
            outcome: {},
            reward: {},
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

validateReplayPayload(validPayload());

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
  "mismatched action signal capacity",
  (payload) => {
    payload.viewer.trajectory.action_contract.communication.token_count = 3;
  },
  "must match signal contract capacity",
);

console.log("viewer_validator_smoke_ok malformed_cases=7");

function signalContract() {
  return {
    schema_version: "foundation_signal_contract_v2",
    profile_metadata_policy_visible: false,
    emission_debug_metadata_fields: emissionFields,
    communication_token_count: communicationTokenCount,
    communication_profiles_per_token: communicationProfilesPerToken,
    reserved_profiles: reservedProfiles(),
  };
}

function actionContract() {
  const communicationActions = communicationActionKeys();
  return {
    communication: {
      token_count: communicationTokenCount,
      profiles_per_token: communicationProfilesPerToken,
      action_keys: communicationActions,
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
