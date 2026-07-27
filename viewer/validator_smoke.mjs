import { validateReplayPayload } from "./replay_validator.mjs";
import { zlibSync } from "./vendor/fflate-0.8.3.mjs";
import {
  ACTION_COMMUNICATION_MEANING,
  ACTION_COMMUNICATION_SPEC_FIXED_FIELDS,
  ACTION_DEBUG_KEY_ENCODING,
  ACTION_MATE_ACTION_KEY,
  ACTION_NON_COMMUNICATION_SPECS,
  ACTION_POLICY_ID_ENCODING,
  COMMUNICATION_AGGREGATE_PROJECTION,
  OBSERVATION_INPUT_VECTOR_SIZE,
  OBSERVATION_PATCH_FIELDS,
  OBSERVATION_PATCH_INPUT_FIELDS,
  OBSERVATION_SELF_FIELDS,
  OBSERVATION_SELF_INPUT_FIELDS,
  PATCH_CELL_COUNT,
  REQUIRED_AGENT_FIELDS,
  REWARD_CONTRACT,
  SUMMARY_SCHEMA_VERSION,
  TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
  TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
  TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
  TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
  TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
  TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
} from "./contracts.generated.mjs";

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
let malformedCaseCount = 0;

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
      summary_schema_version: SUMMARY_SCHEMA_VERSION,
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
        reward_contract: clone(REWARD_CONTRACT),
        observation_contract: {
          schema_version: "mind_observation_v3",
          self_fields: [...OBSERVATION_SELF_FIELDS],
          patch_fields: [...OBSERVATION_PATCH_FIELDS],
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
            reward: reward(),
          },
        ],
      },
    },
  };
}

function validTokenizedPayload() {
  const payload = validPayload();
  const tokenCount = 2;
  const profilesPerToken = 1;
  const tokenFields = communicationTokenFields(tokenCount);
  const actionOptions = {
    tokenCount,
    profilesPerToken,
    emissionEnabled: true,
  };
  const trajectory = payload.viewer.trajectory;
  const observationContract = trajectory.observation_contract;
  const record = trajectory.records[0];

  payload.summary.summary_schema_version =
    TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION;
  trajectory.schema_version =
    TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION;
  trajectory.observation_schema_version =
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION;
  trajectory.action_contract = actionContract(actionOptions);

  observationContract.schema_version =
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION;
  observationContract.self_fields = [
    ...OBSERVATION_SELF_FIELDS,
    ...tokenFields,
  ];
  observationContract.patch_fields = [
    ...OBSERVATION_PATCH_FIELDS,
    ...tokenFields,
  ];
  observationContract.policy_input = observationPolicyInput({
    encoderVersion: TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
    tokenFields,
  });
  observationContract.action_names = actionNames(actionOptions);
  observationContract.action_contract = actionContract(actionOptions);
  observationContract.signal_contract = signalContract(actionOptions);
  observationContract.communication_token_channels = {
    policy_visible: true,
    spatial: true,
    field_order: [...tokenFields],
    token_order: tokenFields.map((_, tokenId) => tokenId),
    simulator_assigned_meanings: false,
    profile_provenance_policy_visible: false,
    aggregate_communication_field_retained: true,
    aggregate_projection: COMMUNICATION_AGGREGATE_PROJECTION,
  };

  record.observation_schema =
    TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION;
  record.observation_input = observationInput({
    schemaVersion: TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    encoderVersion: TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
    tokenFields,
  });
  for (const replayFrame of payload.viewer.frames) {
    replayFrame.signal_emissions.schema_version =
      TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION;
    for (const field of tokenFields) {
      replayFrame.signal_fields[field] = matrix(0);
      replayFrame.signal_emissions.active_counts[field] = 0;
    }
  }
  return payload;
}

function clone(payload) {
  return JSON.parse(JSON.stringify(payload));
}

function expectError(
  name,
  mutate,
  expectedMessage,
  payloadFactory = validPayload,
) {
  const payload = clone(payloadFactory());
  mutate(payload);
  try {
    validateReplayPayload(payload);
  } catch (error) {
    if (!String(error.message).includes(expectedMessage)) {
      throw new Error(`${name}: expected ${expectedMessage}, got ${error.message}`);
    }
    malformedCaseCount += 1;
    return;
  }
  throw new Error(`${name}: expected validation failure`);
}

function expectValid(name, mutate, payloadFactory = validPayload) {
  const payload = clone(payloadFactory());
  mutate(payload);
  try {
    validateReplayPayload(payload);
  } catch (error) {
    throw new Error(`${name}: expected validation success, got ${error.message}`);
  }
}

validateReplayPayload(validPayload());
validateReplayPayload(validTokenizedPayload());

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

expectError(
  "legacy summary schema must match legacy trajectory",
  (payload) => {
    payload.summary.summary_schema_version =
      TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION;
  },
  "foundation_summary_v1",
);
expectError(
  "token summary schema must match token trajectory",
  (payload) => {
    payload.summary.summary_schema_version = SUMMARY_SCHEMA_VERSION;
  },
  TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
  validTokenizedPayload,
);
expectError(
  "policy patch cell count must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.policy_input.patch_cell_count =
      PATCH_CELL_COUNT + 1;
  },
  `patch_cell_count must be ${PATCH_CELL_COUNT}`,
  validTokenizedPayload,
);
expectError(
  "observation bytes must be base64",
  (payload) => {
    payload.viewer.trajectory.records[0].observation_input.data = "%%%=";
  },
  "valid padded base64",
);
expectError(
  "observation bytes must be zlib",
  (payload) => {
    payload.viewer.trajectory.records[0].observation_input.data =
      Buffer.from([1, 2, 3, 4]).toString("base64");
  },
  "valid zlib stream",
);
expectError(
  "observation bytes must match vector size",
  (payload) => {
    payload.viewer.trajectory.records[0].observation_input.data =
      compressedZeroInt16(OBSERVATION_INPUT_VECTOR_SIZE - 1);
  },
  "decode to exactly",
);
expectError(
  "token order must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.communication_token_channels.token_order.reverse();
  },
  "token_order must be [0, 1]",
  validTokenizedPayload,
);
expectError(
  "policy patch token order must be canonical",
  (payload) => {
    const fields =
      payload.viewer.trajectory.observation_contract.policy_input
        .patch_input_fields;
    fields.splice(
      fields.length - 2,
      2,
      "communication_signal_token_1",
      "communication_signal_token_0",
    );
  },
  "communication_signal_token_0",
  validTokenizedPayload,
);
expectError(
  "token frame matrix is required",
  (payload) => {
    delete payload.viewer.frames[0].signal_fields.communication_signal_token_0;
  },
  "signal_fields.communication_signal_token_0",
  validTokenizedPayload,
);
expectError(
  "observation token channels must be policy visible",
  (payload) => {
    payload.viewer.trajectory.observation_contract.communication_token_channels.policy_visible =
      false;
  },
  "canonical opaque spatial token projection",
  validTokenizedPayload,
);
expectError(
  "observation token profile provenance must stay hidden",
  (payload) => {
    payload.viewer.trajectory.observation_contract.communication_token_channels.profile_provenance_policy_visible =
      true;
  },
  "canonical opaque spatial token projection",
  validTokenizedPayload,
);
expectError(
  "observation aggregate field retention is required",
  (payload) => {
    payload.viewer.trajectory.observation_contract.communication_token_channels.aggregate_communication_field_retained =
      false;
  },
  "canonical opaque spatial token projection",
  validTokenizedPayload,
);
expectError(
  "observation aggregate projection must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.communication_token_channels.aggregate_projection =
      "cellwise_token_sum";
  },
  "canonical opaque spatial token projection",
  validTokenizedPayload,
);
expectError(
  "observation token contract rejects invented meaning fields",
  (payload) => {
    payload.viewer.trajectory.observation_contract.communication_token_channels.token_0_means_food =
      true;
  },
  "keys must be exactly",
  validTokenizedPayload,
);
expectError(
  "signal token reporting schema must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.schema_version =
      "stale";
  },
  TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
  validTokenizedPayload,
);
expectError(
  "signal token order must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.token_order.reverse();
  },
  "token_order must be [0, 1]",
  validTokenizedPayload,
);
expectError(
  "signal token channels must be policy visible",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.policy_visible =
      false;
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "signal token profile provenance must stay hidden",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.profile_provenance_policy_visible =
      true;
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "signal aggregate field retention is required",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.aggregate_field_retained_for_compatibility =
      false;
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "signal aggregate projection must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.aggregation =
      "cellwise_token_sum";
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "signal receiver projection schema must be canonical",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.receiver_projection_schema_version =
      "stale";
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "signal receiver observation policy must exclude self emissions",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.receiver_observation_policy =
      "include_receiver_self_emissions";
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "signal global reporting policy must include all emitters",
  (payload) => {
    payload.viewer.trajectory.observation_contract.signal_contract.communication_token_observation.global_reporting_policy =
      "exclude_all_receiver_emissions";
  },
  "opaque public spatial token channels",
  validTokenizedPayload,
);
expectError(
  "action contract rejects invented meaning fields",
  (payload) => {
    payload.viewer.trajectory.action_contract.token_0_means_food = true;
  },
  "keys must be exactly",
);
expectError(
  "action communication rejects invented meaning fields",
  (payload) => {
    payload.viewer.trajectory.action_contract.communication.token_0_means_food =
      true;
  },
  "keys must be exactly",
);
expectError(
  "action entries reject invented meaning fields",
  (payload) => {
    payload.viewer.trajectory.action_contract.actions[12].token_0_means_food =
      true;
  },
  "keys must be exactly",
);
expectError(
  "action communication meaning must remain opaque",
  (payload) => {
    payload.viewer.trajectory.action_contract.communication.meaning =
      "token_0_means_food";
  },
  ACTION_COMMUNICATION_MEANING,
);
expectError(
  "reward contract rejects unknown fields",
  (payload) => {
    payload.viewer.trajectory.reward_contract.communication_token_0_bonus = 1;
  },
  "keys must be exactly",
);
expectError(
  "reward contract rejects unknown components",
  (payload) => {
    payload.viewer.trajectory.reward_contract.component_bounds.communication_token_0_bonus =
      [0, 1];
  },
  "keys must be exactly",
);
expectError(
  "record reward rejects unknown fields",
  (payload) => {
    payload.viewer.trajectory.records[0].reward.communication_token_0_bonus = 1;
  },
  "keys must be exactly",
);
expectError(
  "record reward rejects unknown components",
  (payload) => {
    payload.viewer.trajectory.records[0].reward.components.communication_token_0_bonus =
      0;
  },
  "keys must be exactly",
);
expectError(
  "record reward requires every component",
  (payload) => {
    delete payload.viewer.trajectory.records[0].reward.components.movement_cost;
  },
  "keys must be exactly",
);
expectError(
  "record reward component must stay bounded",
  (payload) => {
    payload.viewer.trajectory.records[0].reward.components.resource_acquisition =
      2;
  },
  "resource_acquisition must be within",
);
expectError(
  "record reward total must equal component sum",
  (payload) => {
    payload.viewer.trajectory.records[0].reward.total = 0.03;
  },
  "must equal the four-decimal component sum",
);

console.log(
  `viewer_validator_smoke_ok malformed_cases=${malformedCaseCount}`,
);

function observationInput({
  schemaVersion = "mind_observation_v3",
  encoderVersion = "mind_observation_encoder_v2",
  tokenFields = [],
} = {}) {
  const size =
    OBSERVATION_INPUT_VECTOR_SIZE +
    tokenFields.length * (1 + PATCH_CELL_COUNT);
  return {
    schema_version: schemaVersion,
    encoder_version: encoderVersion,
    decoded_dtype: "float32",
    storage_dtype: "int16",
    storage_encoding: "zlib_base64_little_endian_int16",
    shape: [size],
    value_range: [-1, 1],
    data: compressedZeroInt16(size),
    ...(tokenFields.length > 0
      ? {
          communication_token_count: tokenFields.length,
          communication_token_field_order: [...tokenFields],
        }
      : {}),
  };
}

function observationPolicyInput({
  encoderVersion = "mind_observation_encoder_v2",
  tokenFields = [],
} = {}) {
  const size =
    OBSERVATION_INPUT_VECTOR_SIZE +
    tokenFields.length * (1 + PATCH_CELL_COUNT);
  return {
    encoder_version: encoderVersion,
    decoded_dtype: "float32",
    storage_dtype: "int16",
    storage_encoding: "zlib_base64_little_endian_int16",
    shape: [size],
    value_range: [-1, 1],
    self_input_fields: [...OBSERVATION_SELF_INPUT_FIELDS, ...tokenFields],
    patch_input_fields: [...OBSERVATION_PATCH_INPUT_FIELDS, ...tokenFields],
    patch_cell_count: PATCH_CELL_COUNT,
  };
}

function signalContract({
  tokenCount = communicationTokenCount,
  profilesPerToken = communicationProfilesPerToken,
  emissionEnabled = false,
} = {}) {
  const tokenFields = communicationTokenFields(tokenCount);
  const contract = {
    schema_version: emissionEnabled
      ? TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION
      : "foundation_signal_contract_v2",
    fields: [
      "reproductive_signal",
      "communication_signal",
      ...(emissionEnabled ? tokenFields : []),
    ],
    signal_substrate_enabled: true,
    reproductive_signal_emission_enabled: true,
    communication_signal_emission_enabled: emissionEnabled,
    communication_tokens_have_simulator_assigned_meaning: false,
    profile_metadata_policy_visible: false,
    emission_debug_metadata_fields: emissionFields,
    communication_token_count: tokenCount,
    communication_profiles_per_token: profilesPerToken,
    reserved_profiles: reservedProfiles({
      tokenCount,
      profilesPerToken,
      emissionEnabled,
    }),
  };
  if (emissionEnabled) {
    contract.communication_token_observation = {
      schema_version: TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
      policy_visible: true,
      spatial: true,
      field_order: [...tokenFields],
      token_order: tokenFields.map((_, tokenId) => tokenId),
      aggregate_field: "communication_signal",
      aggregate_field_retained_for_compatibility: true,
      aggregation: COMMUNICATION_AGGREGATE_PROJECTION,
      receiver_projection_schema_version:
        "foundation_communication_receiver_projection_v1",
      receiver_observation_policy:
        "exclude_receiver_own_communication_emissions_across_local_patch_v1",
      global_reporting_policy: "include_all_emitters_v1",
      simulator_assigned_meanings: false,
      profile_provenance_policy_visible: false,
    };
  }
  return contract;
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

function actionContract({
  tokenCount = communicationTokenCount,
  profilesPerToken = communicationProfilesPerToken,
  emissionEnabled = false,
} = {}) {
  const communicationActions = communicationActionKeys(
    tokenCount,
    profilesPerToken,
  );
  const communicationSpecs = communicationActions.map((key, index) => ({
    action_id: ACTION_NON_COMMUNICATION_SPECS.length + index,
    key,
    ...ACTION_COMMUNICATION_SPEC_FIXED_FIELDS,
    active: emissionEnabled,
    debug_label: key,
  }));
  const actions = [
    ...ACTION_NON_COMMUNICATION_SPECS.map((spec) => ({ ...spec })),
    ...communicationSpecs,
  ];
  return {
    schema_version: "mind_action_contract_v1",
    policy_id_encoding: ACTION_POLICY_ID_ENCODING,
    debug_key_encoding: ACTION_DEBUG_KEY_ENCODING,
    active_action_keys: actions
      .filter((spec) => spec.active)
      .map((spec) => spec.key),
    reserved_action_keys: actions
      .filter((spec) => spec.reserved)
      .map((spec) => spec.key),
    mate_action_key: ACTION_MATE_ACTION_KEY,
    communication: {
      token_count: tokenCount,
      profiles_per_token: profilesPerToken,
      action_keys: communicationActions,
      emission_enabled: emissionEnabled,
      meaning: ACTION_COMMUNICATION_MEANING,
    },
    actions,
  };
}

function actionNames(options = {}) {
  return actionContract(options).actions.map((spec) => spec.key);
}

function communicationActionKeys(
  tokenCount = communicationTokenCount,
  profilesPerToken = communicationProfilesPerToken,
) {
  const keys = [];
  for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
    for (
      let profileIndex = 0;
      profileIndex < profilesPerToken;
      profileIndex += 1
    ) {
      keys.push(`signal_${tokenIndex}_profile_${profileIndex}`);
    }
  }
  return keys;
}

function communicationTokenFields(tokenCount) {
  return Array.from(
    { length: tokenCount },
    (_, tokenId) => `communication_signal_token_${tokenId}`,
  );
}

function compressedZeroInt16(size) {
  const bytes = new Uint8Array(size * Int16Array.BYTES_PER_ELEMENT);
  return Buffer.from(zlibSync(bytes)).toString("base64");
}

function reward() {
  const components = Object.fromEntries(
    Object.keys(REWARD_CONTRACT.component_bounds).map((component) => [
      component,
      component === "survival_continuation" ? 0.02 : 0,
    ]),
  );
  return {
    schema_version: "mind_reward_v1",
    components,
    total: 0.02,
  };
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

function reservedProfiles({
  tokenCount = communicationTokenCount,
  profilesPerToken = communicationProfilesPerToken,
  emissionEnabled = false,
} = {}) {
  const profiles = [];
  for (let tokenIndex = 0; tokenIndex < tokenCount; tokenIndex += 1) {
    for (
      let profileIndex = 0;
      profileIndex < profilesPerToken;
      profileIndex += 1
    ) {
      profiles.push({
        profile_id: `communication_token_${tokenIndex}_profile_${profileIndex}`,
        field_name: emissionEnabled
          ? `communication_signal_token_${tokenIndex}`
          : "communication_signal",
        source_kind: "reserved_opaque_communication",
        token_id: tokenIndex,
        profile_index: profileIndex,
        policy_visible: false,
      });
    }
  }
  return profiles;
}
