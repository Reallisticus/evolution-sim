import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";
import { zlibSync } from "./vendor/fflate-0.8.3.mjs";
import {
  ACTION_COMMUNICATION_MEANING,
  ACTION_COMMUNICATION_SPEC_FIXED_FIELDS,
  ACTION_DEBUG_KEY_ENCODING,
  ACTION_MATE_ACTION_KEY,
  ACTION_NON_COMMUNICATION_SPECS,
  ACTION_POLICY_ID_ENCODING,
  OBSERVATION_INPUT_VECTOR_SIZE,
  OBSERVATION_PATCH_FIELDS,
  OBSERVATION_PATCH_INPUT_FIELDS,
  OBSERVATION_SELF_FIELDS,
  OBSERVATION_SELF_INPUT_FIELDS,
  PATCH_CELL_COUNT,
  REQUIRED_AGENT_FIELDS,
  REWARD_CONTRACT,
  SUMMARY_SCHEMA_VERSION,
} from "./contracts.generated.mjs";
import { ensureViewerServer, stopViewerServer } from "./smoke_server.mjs";

const outputDir = "output/playwright/malformed-replays";
mkdirSync(outputDir, { recursive: true });

const cases = [
  {
    name: "empty_frames",
    fileName: "empty-frames.json",
    payload: malformedValidBase((payload) => {
      payload.summary.ticks_executed = 0;
      payload.viewer.frames = [];
    }),
    expected: "viewer.frames must be a non-empty array",
  },
  {
    name: "missing_signal_emissions",
    fileName: "missing-signal-emissions.json",
    payload: malformedValidBase((payload) => {
      delete payload.viewer.frames[0].signal_emissions;
    }),
    expected: "signal_emissions",
  },
  {
    name: "ragged_terrain",
    fileName: "ragged-terrain.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.map.terrain_codes[1] = [0];
    }),
    expected: "must have 2 columns",
  },
  {
    name: "mismatched_action_signal_capacity",
    fileName: "mismatched-action-signal-capacity.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.action_contract.communication.token_count = 3;
    }),
    expected: "must match signal contract capacity",
  },
  {
    name: "stale_action_contract_version",
    fileName: "stale-action-contract-version.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.action_contract.schema_version = "stale";
    }),
    expected: "mind_action_contract_v1",
  },
  {
    name: "mismatched_communication_enablement",
    fileName: "mismatched-communication-enablement.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.action_contract.communication.emission_enabled = true;
    }),
    expected: "communication.emission_enabled",
  },
  {
    name: "active_communication_action_while_disabled",
    fileName: "active-communication-action-while-disabled.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.action_contract.active_action_keys.push(
        "signal_0_profile_0",
      );
    }),
    expected: "active_action_keys",
  },
  {
    name: "stale_record_outcome_version",
    fileName: "stale-record-outcome-version.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.records[0].outcome.schema_version = "stale";
    }),
    expected: "mind_action_outcome_v2",
  },
  {
    name: "stale_observation_input_encoder",
    fileName: "stale-observation-input-encoder.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.records[0].observation_input.encoder_version = "stale";
    }),
    expected: "mind_observation_encoder_v2",
  },
  {
    name: "stale_observation_input_shape",
    fileName: "stale-observation-input-shape.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.records[0].observation_input.shape = [541];
    }),
    expected: "observation_input.shape",
  },
  {
    name: "stale_later_record_outcome_version",
    fileName: "stale-later-record-outcome-version.json",
    payload: malformedValidBase((payload) => {
      const secondRecord = JSON.parse(
        JSON.stringify(payload.viewer.trajectory.records[0]),
      );
      secondRecord.tick = 1;
      secondRecord.outcome.schema_version = "stale";
      payload.viewer.trajectory.records.push(secondRecord);
    }),
    expected: "mind_action_outcome_v2",
  },
  {
    name: "stale_observation_policy_input_encoder",
    fileName: "stale-observation-policy-input-encoder.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.observation_contract.policy_input.encoder_version =
        "stale";
    }),
    expected: "mind_observation_encoder_v2",
  },
  {
    name: "stale_observation_policy_input_shape",
    fileName: "stale-observation-policy-input-shape.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.observation_contract.policy_input.shape = [541];
    }),
    expected: "policy_input.shape",
  },
  {
    name: "stale_record_reward_version",
    fileName: "stale-record-reward-version.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.records[0].reward.schema_version = "stale";
    }),
    expected: "mind_reward_v1",
  },
  {
    name: "missing_reproductive_group_catalog",
    fileName: "missing-reproductive-group-catalog.json",
    payload: malformedValidBase((payload) => {
      delete payload.viewer.reproductive_group_catalog;
    }),
    expected: "reproductive_group_catalog",
  },
  {
    name: "mismatched_reproductive_group_member_count",
    fileName: "mismatched-reproductive-group-member-count.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.reproductive_group_catalog.groups["1"].member_count = 2;
    }),
    expected: "member_count does not match agent catalog",
  },
  {
    name: "mismatched_reproductive_group_birth_event_count",
    fileName: "mismatched-reproductive-group-birth-event-count.json",
    payload: malformedValidBase((payload) => {
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
    }),
    expected: "asexual_births does not match agent_reproduced events",
  },
  {
    name: "stale_reproductive_birth_event_schema",
    fileName: "stale-reproductive-birth-event-schema.json",
    payload: malformedValidBase((payload) => {
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
    }),
    expected: "reproduction_event_v1",
  },
  {
    name: "incomplete_multi_offspring_sibling_group",
    fileName: "incomplete-multi-offspring-sibling-group.json",
    payload: malformedValidBase((payload) => {
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
    }),
    expected: "multi-offspring sibling group",
  },
  {
    name: "clamped_multi_offspring_missing_limit_reason",
    fileName: "clamped-multi-offspring-missing-limit-reason.json",
    payload: malformedValidBase((payload) => {
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
    }),
    expected: "must include a limit reason",
  },
  {
    name: "unknown_reproductive_expression",
    fileName: "unknown-reproductive-expression.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.agent_catalog["1"].reproductive_expression = "mystery";
    }),
    expected: "unknown reproductive expression",
  },
  {
    name: "missing_agent_reproductive_group",
    fileName: "missing-agent-reproductive-group.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.agent_catalog["1"].reproductive_group_id = null;
    }),
    expected: "missing reproductive group",
  },
  {
    name: "invalid_agent_reproductive_group_type",
    fileName: "invalid-agent-reproductive-group-type.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.agent_catalog["1"].reproductive_group_id = "1";
    }),
    expected: "reproductive_group_id must be an integer",
  },
  {
    name: "invalid_agent_death_tick",
    fileName: "invalid-agent-death-tick.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.agent_catalog["1"].death_tick = "later";
    }),
    expected: "death_tick must be an integer",
  },
  {
    name: "mismatched_reproductive_group_id",
    fileName: "mismatched-reproductive-group-id.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.reproductive_group_catalog.groups["1"].group_id = 2;
    }),
    expected: "mismatched id",
  },
  {
    name: "missing_signal_outcome",
    fileName: "missing-signal-outcome.json",
    payload: malformedValidBase((payload) => {
      delete payload.viewer.trajectory.records[0].outcome.signal;
    }),
    expected: "outcome.signal",
  },
];

const viewerServer = await ensureViewerServer();
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 760 } });
  for (const testCase of cases) {
    const path = `${outputDir}/${testCase.fileName}`;
    writeFileSync(path, JSON.stringify(testCase.payload), "utf8");
    const replayPath = `../${path}`;
    const viewerUrl = `http://127.0.0.1:4173/viewer/index.html?replay=${encodeURIComponent(replayPath)}`;

    await page.goto(viewerUrl, { waitUntil: "networkidle" });
    await page.waitForFunction(
      (expected) => {
        const status = document.querySelector("#load-status")?.textContent ?? "";
        return status.includes("Failed to load replay") && status.includes(expected);
      },
      testCase.expected,
    );
    const debugState = await page.evaluate(() => window.__viewerDebug);
    if (debugState?.loaded !== false) {
      throw new Error(`${testCase.name}: malformed replay was marked loaded`);
    }
  }

  console.log(`viewer_malformed_smoke_ok cases=${cases.length}`);
} finally {
  if (browser) {
    await browser.close();
  }
  await stopViewerServer(viewerServer);
}

function malformedValidBase(mutator) {
  const payload = validPayload();
  mutator(payload);
  return payload;
}

function validPayload() {
  return {
    events: [],
    summary: {
      run_id: "malformed-smoke",
      summary_schema_version: SUMMARY_SCHEMA_VERSION,
      ticks_executed: 1,
    },
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
      frames: [frame()],
      agent_encoding: agentEncoding(),
      agent_catalog: agentCatalog(),
      reproductive_group_catalog: reproductiveGroupCatalog(),
      species_catalog: {},
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
        reward_contract: JSON.parse(JSON.stringify(REWARD_CONTRACT)),
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

function observationInput() {
  return {
    schema_version: "mind_observation_v3",
    encoder_version: "mind_observation_encoder_v2",
    decoded_dtype: "float32",
    storage_dtype: "int16",
    storage_encoding: "zlib_base64_little_endian_int16",
    shape: [OBSERVATION_INPUT_VECTOR_SIZE],
    value_range: [-1, 1],
    data: compressedZeroInt16(OBSERVATION_INPUT_VECTOR_SIZE),
  };
}

function observationPolicyInput() {
  return {
    encoder_version: "mind_observation_encoder_v2",
    decoded_dtype: "float32",
    storage_dtype: "int16",
    storage_encoding: "zlib_base64_little_endian_int16",
    shape: [OBSERVATION_INPUT_VECTOR_SIZE],
    value_range: [-1, 1],
    self_input_fields: [...OBSERVATION_SELF_INPUT_FIELDS],
    patch_input_fields: [...OBSERVATION_PATCH_INPUT_FIELDS],
    patch_cell_count: PATCH_CELL_COUNT,
  };
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

function frame() {
  return {
    tick: 0,
    agents: [agentRow()],
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

function agentRow() {
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
    age: 0,
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
  return agentEncoding().map((field) => values[field] ?? null);
}

function matrix(value) {
  return [
    [value, value],
    [value, value],
  ];
}

function signalEmissionFields() {
  return [
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
}

function signalContract() {
  return {
    schema_version: "foundation_signal_contract_v2",
    fields: ["reproductive_signal", "communication_signal"],
    signal_substrate_enabled: true,
    reproductive_signal_emission_enabled: true,
    communication_signal_emission_enabled: false,
    communication_tokens_have_simulator_assigned_meaning: false,
    profile_metadata_policy_visible: false,
    emission_debug_metadata_fields: signalEmissionFields(),
    communication_token_count: 4,
    communication_profiles_per_token: 2,
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
  const communicationSpecs = communicationActions.map((key, index) => ({
    action_id: ACTION_NON_COMMUNICATION_SPECS.length + index,
    key,
    ...ACTION_COMMUNICATION_SPEC_FIXED_FIELDS,
    active: false,
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
      token_count: 4,
      profiles_per_token: 2,
      action_keys: communicationActions,
      emission_enabled: false,
      meaning: ACTION_COMMUNICATION_MEANING,
    },
    actions,
  };
}

function actionNames() {
  return actionContract().actions.map((spec) => spec.key);
}

function communicationActionKeys() {
  const keys = [];
  for (let tokenIndex = 0; tokenIndex < 4; tokenIndex += 1) {
    for (let profileIndex = 0; profileIndex < 2; profileIndex += 1) {
      keys.push(`signal_${tokenIndex}_profile_${profileIndex}`);
    }
  }
  return keys;
}

function reservedProfiles() {
  const profiles = [];
  for (let tokenIndex = 0; tokenIndex < 4; tokenIndex += 1) {
    for (let profileIndex = 0; profileIndex < 2; profileIndex += 1) {
      profiles.push({
        profile_id: `communication_token_${tokenIndex}_profile_${profileIndex}`,
        field_name: "communication_signal",
        source_kind: "reserved_opaque_communication",
        token_id: tokenIndex,
        profile_index: profileIndex,
        policy_visible: false,
      });
    }
  }
  return profiles;
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

function agentEncoding() {
  return [...REQUIRED_AGENT_FIELDS];
}
