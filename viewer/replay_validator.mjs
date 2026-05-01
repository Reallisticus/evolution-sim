export const REQUIRED_AGENT_FIELDS = [
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

const REQUIRED_MAP_FIELDS = ["fertility", "moisture", "heat"];
const REQUIRED_FRAME_MATRIX_FIELDS = [
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
];
const REQUIRED_BIOTIC_FIELDS = ["prey_biomass", "carrion", "predator_risk"];
const REQUIRED_SIGNAL_FIELDS = ["reproductive_signal", "communication_signal"];
const REQUIRED_SIGNAL_EMISSION_FIELDS = [
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
const REQUIRED_TRAJECTORY_RECORD_FIELDS = [
  "observation_metadata",
  "observation_input",
  "observation_digest",
  "action_mask",
  "resolution_action_mask",
  "requested_action",
  "policy_id",
  "policy_version",
  "action_valid",
  "resolution_action_valid",
  "resolved_action",
  "outcome",
  "reward",
];

export function validateReplayPayload(payload) {
  assertObject(payload, "Replay payload");
  assertObject(payload.summary, "Replay summary");
  assertObject(payload.viewer, "Replay viewer");

  const { summary, viewer } = payload;
  const map = validateMap(viewer.map);
  validateAgentEncoding(viewer.agent_encoding);
  validateCatalogs(viewer);
  validateTrajectory(viewer.trajectory);
  validateFrames(viewer.frames, map, summary, viewer.agent_encoding, viewer.trajectory);

  return payload;
}

function validateMap(map) {
  assertObject(map, "Replay viewer.map");
  assertPositiveInteger(map.width, "Replay viewer.map.width");
  assertPositiveInteger(map.height, "Replay viewer.map.height");
  assertMatrix(map.terrain_codes, map.width, map.height, "Replay viewer.map.terrain_codes");

  const baseFields = map.base_tile_fields ?? map.environment_fields;
  assertObject(baseFields, "Replay viewer.map.base_tile_fields");
  for (const fieldName of REQUIRED_MAP_FIELDS) {
    assertMatrix(
      baseFields[fieldName],
      map.width,
      map.height,
      `Replay viewer.map.base_tile_fields.${fieldName}`,
    );
  }

  return { width: map.width, height: map.height };
}

function validateAgentEncoding(agentEncoding) {
  assertArray(agentEncoding, "Replay viewer.agent_encoding");
  for (const field of REQUIRED_AGENT_FIELDS) {
    if (!agentEncoding.includes(field)) {
      throw new Error(`Replay viewer.agent_encoding is missing required field ${field}.`);
    }
  }
}

function validateCatalogs(viewer) {
  assertObject(viewer.agent_catalog, "Replay viewer.agent_catalog");
  assertObject(viewer.species_catalog, "Replay viewer.species_catalog");
  if (viewer.ecotype_catalog !== undefined) {
    assertObject(viewer.ecotype_catalog, "Replay viewer.ecotype_catalog");
  }
}

function validateTrajectory(trajectory) {
  assertObject(trajectory, "Replay viewer.trajectory");
  assertArray(trajectory.records, "Replay viewer.trajectory.records");
  assertObject(
    trajectory.observation_contract,
    "Replay viewer.trajectory.observation_contract",
  );
  const observationContract = trajectory.observation_contract;
  const signalContract = trajectory.observation_contract.signal_contract;
  assertObject(
    signalContract,
    "Replay viewer.trajectory.observation_contract.signal_contract",
  );
  validateSignalCommunicationContract(signalContract);
  if (signalContract.profile_metadata_policy_visible !== false) {
    throw new Error(
      "Replay signal contract must mark profile metadata as policy-hidden.",
    );
  }
  assertArray(
    signalContract.emission_debug_metadata_fields,
    "Replay viewer.trajectory.observation_contract.signal_contract.emission_debug_metadata_fields",
  );
  for (const field of REQUIRED_SIGNAL_EMISSION_FIELDS) {
    if (!signalContract.emission_debug_metadata_fields.includes(field)) {
      throw new Error(
        `Replay signal contract is missing debug metadata field ${field}.`,
      );
    }
  }
  validateActionSignalContract(
    trajectory.action_contract,
    signalContract,
    "Replay viewer.trajectory.action_contract",
  );
  validateActionSignalContract(
    observationContract.action_contract,
    signalContract,
    "Replay viewer.trajectory.observation_contract.action_contract",
    observationContract.action_names,
  );

  if (trajectory.records.length > 0) {
    const firstRecord = trajectory.records[0];
    assertObject(firstRecord, "Replay viewer.trajectory.records[0]");
    for (const field of REQUIRED_TRAJECTORY_RECORD_FIELDS) {
      if (!(field in firstRecord)) {
        throw new Error(`Replay trajectory record is missing field ${field}.`);
      }
    }
    assertObject(
      firstRecord.observation_input,
      "Replay viewer.trajectory.records[0].observation_input",
    );
    assertObject(
      firstRecord.observation_metadata,
      "Replay viewer.trajectory.records[0].observation_metadata",
    );
  }
}

function validateSignalCommunicationContract(signalContract) {
  assertPositiveInteger(
    signalContract.communication_token_count,
    "Replay signal contract communication_token_count",
  );
  assertPositiveInteger(
    signalContract.communication_profiles_per_token,
    "Replay signal contract communication_profiles_per_token",
  );
  assertArray(
    signalContract.reserved_profiles,
    "Replay signal contract reserved_profiles",
  );
  const tokenCount = signalContract.communication_token_count;
  const profilesPerToken = signalContract.communication_profiles_per_token;
  const expectedCount = tokenCount * profilesPerToken;
  if (signalContract.reserved_profiles.length !== expectedCount) {
    throw new Error(
      "Replay signal contract reserved profile count must match communication capacity.",
    );
  }
  for (const [index, profile] of signalContract.reserved_profiles.entries()) {
    assertObject(profile, `Replay signal contract reserved_profiles[${index}]`);
    const expectedTokenId = Math.floor(index / profilesPerToken);
    const expectedProfileIndex = index % profilesPerToken;
    if (
      profile.field_name !== "communication_signal" ||
      profile.token_id !== expectedTokenId ||
      profile.profile_index !== expectedProfileIndex ||
      profile.policy_visible !== false
    ) {
      throw new Error(
        "Replay signal contract reserved communication profiles must be ordered, opaque, and policy-hidden.",
      );
    }
  }
}

function validateActionSignalContract(
  actionContract,
  signalContract,
  path,
  observationActionNames = undefined,
) {
  assertObject(actionContract, path);
  assertObject(actionContract.communication, `${path}.communication`);
  assertPositiveInteger(
    actionContract.communication.token_count,
    `${path}.communication.token_count`,
  );
  assertPositiveInteger(
    actionContract.communication.profiles_per_token,
    `${path}.communication.profiles_per_token`,
  );
  if (
    actionContract.communication.token_count !==
      signalContract.communication_token_count ||
    actionContract.communication.profiles_per_token !==
      signalContract.communication_profiles_per_token
  ) {
    throw new Error(`${path}.communication must match signal contract capacity.`);
  }

  const expectedActionKeys = communicationActionKeys(
    signalContract.communication_token_count,
    signalContract.communication_profiles_per_token,
  );
  assertArray(actionContract.communication.action_keys, `${path}.communication.action_keys`);
  assertStringArray(actionContract.communication.action_keys, `${path}.communication.action_keys`);
  assertStringArray(actionContract.reserved_action_keys, `${path}.reserved_action_keys`);
  assertExactStringArray(
    actionContract.communication.action_keys,
    expectedActionKeys,
    `${path}.communication.action_keys`,
  );

  const reserved = new Set(actionContract.reserved_action_keys);
  for (const actionKey of expectedActionKeys) {
    if (!reserved.has(actionKey)) {
      throw new Error(`${path}.reserved_action_keys is missing ${actionKey}.`);
    }
  }

  if (observationActionNames !== undefined) {
    assertStringArray(observationActionNames, "Replay observation action_names");
    const actionNameSet = new Set(observationActionNames);
    for (const actionKey of expectedActionKeys) {
      if (!actionNameSet.has(actionKey)) {
        throw new Error(
          `Replay observation action_names is missing ${actionKey}.`,
        );
      }
    }
  }
}

function communicationActionKeys(tokenCount, profilesPerToken) {
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

function validateFrames(frames, map, summary, agentEncoding, trajectory) {
  assertArray(frames, "Replay viewer.frames");
  if (frames.length === 0) {
    throw new Error("Replay viewer.frames must be a non-empty array.");
  }
  if (
    Number.isInteger(summary.ticks_executed) &&
    summary.ticks_executed !== frames.length
  ) {
    throw new Error(
      "Replay viewer.frames length must match summary.ticks_executed.",
    );
  }

  let previousTick = -1;
  const signalContractVersion =
    trajectory.observation_contract.signal_contract.schema_version;
  const agentFieldMap = Object.fromEntries(
    agentEncoding.map((field, index) => [field, index]),
  );
  for (const [index, frame] of frames.entries()) {
    assertObject(frame, `Replay frame ${index}`);
    assertInteger(frame.tick, `Replay frame ${index}.tick`);
    if (frame.tick <= previousTick) {
      throw new Error("Replay frame ticks must be strictly increasing.");
    }
    previousTick = frame.tick;

    assertArray(frame.agents, `Replay frame ${index}.agents`);
    validateAgents(frame.agents, agentEncoding.length, agentFieldMap, map, index);
    assertArray(frame.species_counts, `Replay frame ${index}.species_counts`);
    assertObject(frame.field_state, `Replay frame ${index}.field_state`);
    assertObject(frame.signal_flow, `Replay frame ${index}.signal_flow`);
    assertObject(frame.species_metrics, `Replay frame ${index}.species_metrics`);

    for (const fieldName of REQUIRED_FRAME_MATRIX_FIELDS) {
      assertMatrix(
        frame[fieldName],
        map.width,
        map.height,
        `Replay frame ${index}.${fieldName}`,
      );
    }
    validateNestedFieldMatrices(
      frame.biotic_fields,
      REQUIRED_BIOTIC_FIELDS,
      map,
      `Replay frame ${index}.biotic_fields`,
    );
    validateNestedFieldMatrices(
      frame.signal_fields,
      REQUIRED_SIGNAL_FIELDS,
      map,
      `Replay frame ${index}.signal_fields`,
    );
    validateSignalEmissions(frame.signal_emissions, signalContractVersion, index);
  }
}

function validateAgents(agents, encodedLength, agentFieldMap, map, frameIndex) {
  for (const [agentIndex, encoded] of agents.entries()) {
    assertArray(encoded, `Replay frame ${frameIndex}.agents[${agentIndex}]`);
    if (encoded.length !== encodedLength) {
      throw new Error(
        `Replay frame ${frameIndex}.agents[${agentIndex}] length must match agent_encoding.`,
      );
    }
    const agentId = encoded[agentFieldMap.agent_id];
    const x = encoded[agentFieldMap.x];
    const y = encoded[agentFieldMap.y];
    assertInteger(agentId, `Replay frame ${frameIndex}.agents[${agentIndex}].agent_id`);
    assertInteger(x, `Replay frame ${frameIndex}.agents[${agentIndex}].x`);
    assertInteger(y, `Replay frame ${frameIndex}.agents[${agentIndex}].y`);
    if (x < 0 || x >= map.width || y < 0 || y >= map.height) {
      throw new Error(
        `Replay frame ${frameIndex}.agents[${agentIndex}] position is outside the map.`,
      );
    }
  }
}

function validateNestedFieldMatrices(container, fields, map, path) {
  assertObject(container, path);
  for (const fieldName of fields) {
    assertMatrix(
      container[fieldName],
      map.width,
      map.height,
      `${path}.${fieldName}`,
    );
  }
}

function validateSignalEmissions(signalEmissions, signalContractVersion, frameIndex) {
  assertObject(signalEmissions, `Replay frame ${frameIndex}.signal_emissions`);
  if (signalEmissions.schema_version !== signalContractVersion) {
    throw new Error(
      `Replay frame ${frameIndex}.signal_emissions schema version does not match signal contract.`,
    );
  }
  if (signalEmissions.policy_visible !== false) {
    throw new Error(
      `Replay frame ${frameIndex}.signal_emissions must be debug-only.`,
    );
  }
  assertArray(signalEmissions.events, `Replay frame ${frameIndex}.signal_emissions.events`);
  assertObject(
    signalEmissions.active_counts,
    `Replay frame ${frameIndex}.signal_emissions.active_counts`,
  );
  for (const [eventIndex, event] of signalEmissions.events.entries()) {
    assertObject(
      event,
      `Replay frame ${frameIndex}.signal_emissions.events[${eventIndex}]`,
    );
    for (const field of REQUIRED_SIGNAL_EMISSION_FIELDS) {
      if (!(field in event)) {
        throw new Error(
          `Replay signal emission event is missing debug metadata field ${field}.`,
        );
      }
    }
    if (event.policy_visible !== false) {
      throw new Error("Replay signal emission event must be policy-hidden.");
    }
  }
}

function assertMatrix(matrix, width, height, path) {
  assertArray(matrix, path);
  if (matrix.length !== height) {
    throw new Error(`${path} must have ${height} rows.`);
  }
  for (const [rowIndex, row] of matrix.entries()) {
    assertArray(row, `${path}[${rowIndex}]`);
    if (row.length !== width) {
      throw new Error(`${path}[${rowIndex}] must have ${width} columns.`);
    }
    for (const [columnIndex, value] of row.entries()) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(
          `${path}[${rowIndex}][${columnIndex}] must be a finite number.`,
        );
      }
    }
  }
}

function assertObject(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${path} must be a JSON object.`);
  }
}

function assertArray(value, path) {
  if (!Array.isArray(value)) {
    throw new Error(`${path} must be an array.`);
  }
}

function assertStringArray(value, path) {
  assertArray(value, path);
  for (const [index, item] of value.entries()) {
    if (typeof item !== "string") {
      throw new Error(`${path}[${index}] must be a string.`);
    }
  }
}

function assertExactStringArray(actual, expected, path) {
  if (actual.length !== expected.length) {
    throw new Error(`${path} must contain ${expected.length} entries.`);
  }
  for (const [index, expectedValue] of expected.entries()) {
    if (actual[index] !== expectedValue) {
      throw new Error(`${path}[${index}] must be ${expectedValue}.`);
    }
  }
}

function assertPositiveInteger(value, path) {
  assertInteger(value, path);
  if (value <= 0) {
    throw new Error(`${path} must be greater than zero.`);
  }
}

function assertInteger(value, path) {
  if (!Number.isInteger(value)) {
    throw new Error(`${path} must be an integer.`);
  }
}
