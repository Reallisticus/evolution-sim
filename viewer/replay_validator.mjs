import { unzlibSync } from "./vendor/fflate-0.8.3.mjs";

import {
  ACTION_COMMUNICATION_FIELDS,
  ACTION_COMMUNICATION_MEANING,
  ACTION_COMMUNICATION_SPEC_FIXED_FIELDS,
  ACTION_CONTRACT_VERSION,
  ACTION_CONTRACT_FIELDS,
  ACTION_DEBUG_KEY_ENCODING,
  ACTION_MATE_ACTION_KEY,
  ACTION_NON_COMMUNICATION_SPECS,
  ACTION_OUTCOME_SCHEMA_VERSION,
  ACTION_POLICY_ID_ENCODING,
  ACTION_SPEC_FIELDS,
  ASEXUAL_REPRODUCTION_MODE,
  COMMUNICATION_AGGREGATE_PROJECTION,
  GENOME_RECOMBINATION_CONTRACT_VERSION,
  OBSERVATION_ENCODER_VERSION,
  OBSERVATION_INPUT_DECODED_DTYPE,
  OBSERVATION_INPUT_STORAGE_DTYPE,
  OBSERVATION_INPUT_STORAGE_ENCODING,
  OBSERVATION_INPUT_VALUE_RANGE,
  OBSERVATION_INPUT_VECTOR_SIZE,
  OBSERVATION_PATCH_FIELDS,
  OBSERVATION_PATCH_INPUT_FIELDS,
  OBSERVATION_SCHEMA_VERSION,
  OBSERVATION_SELF_FIELDS,
  OBSERVATION_SELF_INPUT_FIELDS,
  PATCH_CELL_COUNT,
  POLICY_INTERFACE_VERSION,
  REPRODUCTION_EVENT_SCHEMA_VERSION,
  REPRODUCTIVE_EXPRESSIONS,
  REPRODUCTIVE_GROUP_CONTRACT_VERSION,
  REPRODUCTIVE_STAGE_ORDER,
  REQUIRED_AGENT_CATALOG_FIELDS,
  REQUIRED_AGENT_FIELDS,
  REQUIRED_BIOTIC_FIELDS,
  REQUIRED_FRAME_MATRIX_FIELDS,
  REQUIRED_MAP_FIELDS,
  REQUIRED_REPRODUCTIVE_GROUP_FIELDS,
  REQUIRED_SIGNAL_EMISSION_FIELDS,
  REQUIRED_SIGNAL_FIELDS,
  REQUIRED_SIGNAL_OUTCOME_FIELDS,
  REQUIRED_TRAJECTORY_RECORD_FIELDS,
  REWARD_CONTRACT,
  REWARD_RECORD_FIELDS,
  REWARD_SCHEMA_VERSION,
  SEXUAL_REPRODUCTION_MODE,
  SIGNAL_CONTRACT_VERSION,
  SUMMARY_SCHEMA_VERSION,
  TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION,
  TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
  TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION,
  TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
  TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION,
  TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION,
  TRAJECTORY_SCHEMA_VERSION,
} from "./contracts.generated.mjs";

export { REQUIRED_AGENT_FIELDS } from "./contracts.generated.mjs";

export function validateTrajectoryContractForEvidence(trajectory) {
  const validation = validateTrajectory(trajectory);
  return {
    actionOrdering: trajectory.action_contract.actions.map((spec) => spec.key),
    communicationTokenCount:
      trajectory.action_contract.communication.token_count,
    communicationProfilesPerToken:
      trajectory.action_contract.communication.profiles_per_token,
    tokenizedCommunication: validation.tokenizedCommunication,
  };
}

export function validateReplayPayload(payload) {
  assertObject(payload, "Replay payload");
  assertObject(payload.summary, "Replay summary");
  assertObject(payload.viewer, "Replay viewer");

  const { summary, viewer } = payload;
  const map = validateMap(viewer.map);
  validateAgentEncoding(viewer.agent_encoding);
  const trajectoryContract = validateTrajectory(viewer.trajectory);
  validateSummaryContract(summary, trajectoryContract.tokenizedCommunication);
  validateCatalogs(viewer, viewer.trajectory, payload.events);
  validateFrames(
    viewer.frames,
    map,
    summary,
    viewer.agent_encoding,
    viewer.trajectory,
    viewer.agent_catalog,
  );

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

function validateCatalogs(viewer, trajectory, events = undefined) {
  assertObject(viewer.agent_catalog, "Replay viewer.agent_catalog");
  assertObject(viewer.species_catalog, "Replay viewer.species_catalog");
  if (viewer.ecotype_catalog !== undefined) {
    assertObject(viewer.ecotype_catalog, "Replay viewer.ecotype_catalog");
  }
  assertObject(
    viewer.reproductive_group_catalog,
    "Replay viewer.reproductive_group_catalog",
  );
  if (
    viewer.reproductive_group_catalog.schema_version !==
    trajectory.reproductive_group_contract_version
  ) {
    throw new Error(
      "Replay reproductive group catalog schema version must match trajectory contract.",
    );
  }
  assertObject(
    viewer.reproductive_group_catalog.groups,
    "Replay viewer.reproductive_group_catalog.groups",
  );
  const groupCounts = new Map();
  const aliveGroupCounts = new Map();
  const aliveStageCounts = new Map();
  const aliveExpressionCounts = new Map();
  const reproductiveEventBirthCounts = reproductiveBirthCountsFromEvents(
    events,
    viewer.reproductive_group_catalog.groups,
  );

  for (const [agentId, agent] of Object.entries(viewer.agent_catalog)) {
    assertObject(agent, `Replay viewer.agent_catalog.${agentId}`);
    for (const field of REQUIRED_AGENT_CATALOG_FIELDS) {
      if (!(field in agent)) {
        throw new Error(`Replay agent catalog entry is missing field ${field}.`);
      }
    }
    if (String(agent.agent_id) !== agentId) {
      throw new Error(`Replay agent catalog entry ${agentId} has mismatched id.`);
    }
    if (!REPRODUCTIVE_STAGE_ORDER.includes(agent.reproductive_stage)) {
      throw new Error(
        `Replay agent catalog entry ${agentId} has unknown reproductive stage.`,
      );
    }
    if (!REPRODUCTIVE_EXPRESSIONS.includes(agent.reproductive_expression)) {
      throw new Error(
        `Replay agent catalog entry ${agentId} has unknown reproductive expression.`,
      );
    }
    assertArray(agent.parent_ids, `Replay viewer.agent_catalog.${agentId}.parent_ids`);
    assertObject(agent.genome, `Replay viewer.agent_catalog.${agentId}.genome`);
    assertObject(
      agent.mind_inheritance,
      `Replay viewer.agent_catalog.${agentId}.mind_inheritance`,
    );
    if (agent.reproductive_group_id == null) {
      throw new Error(
        `Replay agent catalog entry ${agentId} is missing reproductive group.`,
      );
    }
    assertNonnegativeInteger(
      agent.reproductive_group_id,
      `Replay viewer.agent_catalog.${agentId}.reproductive_group_id`,
    );
    if (agent.death_tick !== null) {
      assertNonnegativeInteger(
        agent.death_tick,
        `Replay viewer.agent_catalog.${agentId}.death_tick`,
      );
    }
    if (
      !(String(agent.reproductive_group_id) in
        viewer.reproductive_group_catalog.groups)
    ) {
      throw new Error(
        `Replay agent catalog entry ${agentId} references missing reproductive group.`,
      );
    }
    const groupId = String(agent.reproductive_group_id);
    incrementCount(groupCounts, groupId);
    if (agent.death_tick == null) {
      incrementCount(aliveGroupCounts, groupId);
      incrementNestedCount(aliveStageCounts, groupId, agent.reproductive_stage);
      incrementNestedCount(
        aliveExpressionCounts,
        groupId,
        agent.reproductive_expression,
      );
    }
  }

  for (const [groupId, group] of Object.entries(
    viewer.reproductive_group_catalog.groups,
  )) {
    assertObject(group, `Replay viewer.reproductive_group_catalog.groups.${groupId}`);
    for (const field of REQUIRED_REPRODUCTIVE_GROUP_FIELDS) {
      if (!(field in group)) {
        throw new Error(
          `Replay reproductive group catalog entry is missing field ${field}.`,
        );
      }
    }
    if (String(group.group_id) !== groupId) {
      throw new Error(
        `Replay reproductive group catalog entry ${groupId} has mismatched id.`,
      );
    }
    assertNonnegativeInteger(
      group.group_id,
      `Replay viewer.reproductive_group_catalog.groups.${groupId}.group_id`,
    );
    if (!REPRODUCTIVE_STAGE_ORDER.includes(group.stage)) {
      throw new Error(
        `Replay reproductive group catalog entry ${groupId} has unknown stage.`,
      );
    }
    assertArray(
      group.parent_group_ids,
      `Replay viewer.reproductive_group_catalog.groups.${groupId}.parent_group_ids`,
    );
    assertObject(
      group.alive_stage_counts,
      `Replay viewer.reproductive_group_catalog.groups.${groupId}.alive_stage_counts`,
    );
    assertObject(
      group.alive_expression_counts,
      `Replay viewer.reproductive_group_catalog.groups.${groupId}.alive_expression_counts`,
    );
    assertNonnegativeInteger(
      group.member_count,
      `Replay viewer.reproductive_group_catalog.groups.${groupId}.member_count`,
    );
    assertNonnegativeInteger(
      group.alive_member_count,
      `Replay viewer.reproductive_group_catalog.groups.${groupId}.alive_member_count`,
    );
    for (const birthField of ["asexual_births", "sexual_births", "hybrid_births"]) {
      assertNonnegativeInteger(
        group[birthField],
        `Replay viewer.reproductive_group_catalog.groups.${groupId}.${birthField}`,
      );
    }
    if (group.hybrid_births > group.sexual_births) {
      throw new Error(
        `Replay reproductive group ${groupId} has more hybrid births than sexual births.`,
      );
    }
    if (reproductiveEventBirthCounts !== null) {
      const eventCounts = reproductiveEventBirthCounts.get(groupId) ?? emptyBirthCounts();
      for (const birthField of ["asexual_births", "sexual_births", "hybrid_births"]) {
        if (group[birthField] !== eventCounts[birthField]) {
          throw new Error(
            `Replay reproductive group ${groupId} ${birthField} does not match agent_reproduced events.`,
          );
        }
      }
    }
    if (group.member_count !== (groupCounts.get(groupId) ?? 0)) {
      throw new Error(
        `Replay reproductive group ${groupId} member_count does not match agent catalog.`,
      );
    }
    if (group.alive_member_count !== (aliveGroupCounts.get(groupId) ?? 0)) {
      throw new Error(
        `Replay reproductive group ${groupId} alive_member_count does not match agent catalog.`,
      );
    }
    assertExactCountObject(
      group.alive_stage_counts,
      aliveStageCounts.get(groupId) ?? new Map(),
      `Replay reproductive group ${groupId} alive_stage_counts`,
    );
    assertExactCountObject(
      group.alive_expression_counts,
      aliveExpressionCounts.get(groupId) ?? new Map(),
      `Replay reproductive group ${groupId} alive_expression_counts`,
    );
    const highestAliveStage = highestStage(aliveStageCounts.get(groupId) ?? new Map());
    if (highestAliveStage && stageRank(group.stage) < stageRank(highestAliveStage)) {
      throw new Error(
        `Replay reproductive group ${groupId} stage is below an alive member stage.`,
      );
    }
  }
}

function reproductiveBirthCountsFromEvents(events, groups) {
  if (events === undefined) {
    return null;
  }
  assertArray(events, "Replay events");
  const groupIds = new Set(Object.keys(groups));
  const counts = new Map();
  const multiOffspringGroups = new Map();
  for (const groupId of groupIds) {
    counts.set(groupId, emptyBirthCounts());
  }
  for (const [eventIndex, event] of events.entries()) {
    assertObject(event, `Replay events[${eventIndex}]`);
    if (event.type !== "agent_reproduced") {
      continue;
    }
    assertObject(event.data, `Replay events[${eventIndex}].data`);
    validateVersion(
      event.data.schema_version,
      REPRODUCTION_EVENT_SCHEMA_VERSION,
      `Replay events[${eventIndex}].data.schema_version`,
    );
    assertNonnegativeInteger(
      event.data.child_reproductive_group_id,
      `Replay events[${eventIndex}].data.child_reproductive_group_id`,
    );
    const groupId = String(event.data.child_reproductive_group_id);
    if (!groupIds.has(groupId)) {
      throw new Error(
        `Replay reproduction event ${eventIndex} references missing reproductive group.`,
      );
    }
    if (typeof event.data.hybrid !== "boolean") {
      throw new Error(`Replay reproduction event ${eventIndex} hybrid must be boolean.`);
    }
    validateReproductionEventOffspringMetadata(
      event.data,
      eventIndex,
      multiOffspringGroups,
    );
    const groupCounts = counts.get(groupId);
    if (event.data.reproduction_mode === ASEXUAL_REPRODUCTION_MODE) {
      groupCounts.asexual_births += 1;
      if (event.data.hybrid) {
        throw new Error(
          `Replay reproduction event ${eventIndex} cannot mark an asexual birth as hybrid.`,
        );
      }
    } else if (event.data.reproduction_mode === SEXUAL_REPRODUCTION_MODE) {
      groupCounts.sexual_births += 1;
      if (event.data.hybrid) {
        groupCounts.hybrid_births += 1;
      }
    } else {
      throw new Error(
        `Replay reproduction event ${eventIndex} has unsupported reproduction_mode.`,
      );
    }
  }
  validateMultiOffspringGroups(multiOffspringGroups);
  return counts;
}

function validateReproductionEventOffspringMetadata(
  data,
  eventIndex,
  multiOffspringGroups,
) {
  const basePath = `Replay events[${eventIndex}].data`;
  assertNonnegativeInteger(data.child_id, `${basePath}.child_id`);
  assertPositiveInteger(data.offspring_count, `${basePath}.offspring_count`);
  validateMultiOffspringAttemptMetadata(data, eventIndex);
  if (data.offspring_count === 1) {
    if (data.multi_offspring === true) {
      throw new Error(
        `Replay reproduction event ${eventIndex} single-child birth cannot be marked multi_offspring.`,
      );
    }
    for (const field of [
      "offspring_index",
      "sibling_child_ids",
      "parent_energy_costs_total",
    ]) {
      if (field in data) {
        throw new Error(
          `Replay reproduction event ${eventIndex} single-child birth cannot carry ${field}.`,
        );
      }
    }
    return;
  }
  if (data.reproduction_mode !== SEXUAL_REPRODUCTION_MODE) {
    throw new Error(
      `Replay reproduction event ${eventIndex} multi-offspring birth must be sexual.`,
    );
  }
  if (data.multi_offspring !== true) {
    throw new Error(
      `Replay reproduction event ${eventIndex} multi-offspring birth must set multi_offspring true.`,
    );
  }
  assertPositiveInteger(data.offspring_index, `${basePath}.offspring_index`);
  if (data.offspring_index > data.offspring_count) {
    throw new Error(
      `Replay reproduction event ${eventIndex} offspring_index exceeds offspring_count.`,
    );
  }
  assertArray(data.sibling_child_ids, `${basePath}.sibling_child_ids`);
  if (data.sibling_child_ids.length !== data.offspring_count) {
    throw new Error(
      `Replay reproduction event ${eventIndex} sibling_child_ids must contain offspring_count entries.`,
    );
  }
  const siblingIds = new Set();
  for (const [index, childId] of data.sibling_child_ids.entries()) {
    assertNonnegativeInteger(childId, `${basePath}.sibling_child_ids[${index}]`);
    if (siblingIds.has(childId)) {
      throw new Error(
        `Replay reproduction event ${eventIndex} sibling_child_ids cannot contain duplicates.`,
      );
    }
    siblingIds.add(childId);
  }
  if (!siblingIds.has(data.child_id)) {
    throw new Error(
      `Replay reproduction event ${eventIndex} sibling_child_ids must include child_id.`,
    );
  }
  if (data.sibling_child_ids[data.offspring_index - 1] !== data.child_id) {
    throw new Error(
      `Replay reproduction event ${eventIndex} offspring_index must identify child_id within sibling_child_ids.`,
    );
  }
  assertArray(data.parent_energy_costs_total, `${basePath}.parent_energy_costs_total`);
  if (Array.isArray(data.parent_ids) && data.parent_energy_costs_total.length !== data.parent_ids.length) {
    throw new Error(
      `Replay reproduction event ${eventIndex} parent_energy_costs_total must match parent_ids length.`,
    );
  }
  for (const [index, entry] of data.parent_energy_costs_total.entries()) {
    assertObject(entry, `${basePath}.parent_energy_costs_total[${index}]`);
    assertNonnegativeInteger(
      entry.agent_id,
      `${basePath}.parent_energy_costs_total[${index}].agent_id`,
    );
    if (
      typeof entry.energy_cost !== "number" ||
      !Number.isFinite(entry.energy_cost) ||
      entry.energy_cost < 0
    ) {
      throw new Error(
        `${basePath}.parent_energy_costs_total[${index}].energy_cost must be a nonnegative finite number.`,
      );
    }
  }
  const siblingKey = data.sibling_child_ids.join(",");
  const group = multiOffspringGroups.get(siblingKey) ?? {
    offspringCount: data.offspring_count,
    seenChildIds: new Set(),
    seenIndexes: new Set(),
  };
  if (group.offspringCount !== data.offspring_count) {
    throw new Error(
      `Replay multi-offspring sibling group ${siblingKey} has inconsistent offspring_count values.`,
    );
  }
  if (group.seenChildIds.has(data.child_id)) {
    throw new Error(
      `Replay multi-offspring sibling group ${siblingKey} repeats child_id ${data.child_id}.`,
    );
  }
  if (group.seenIndexes.has(data.offspring_index)) {
    throw new Error(
      `Replay multi-offspring sibling group ${siblingKey} repeats offspring_index ${data.offspring_index}.`,
    );
  }
  group.seenChildIds.add(data.child_id);
  group.seenIndexes.add(data.offspring_index);
  multiOffspringGroups.set(siblingKey, group);
}

function validateMultiOffspringAttemptMetadata(data, eventIndex) {
  const hasAttemptMetadata =
    "multi_offspring_desired_count" in data ||
    "multi_offspring_actual_count" in data ||
    "multi_offspring_limit_reasons" in data;
  if (!hasAttemptMetadata) {
    return;
  }
  const basePath = `Replay events[${eventIndex}].data`;
  if (data.reproduction_mode !== SEXUAL_REPRODUCTION_MODE) {
    throw new Error(
      `Replay reproduction event ${eventIndex} multi-offspring attempt metadata must be sexual.`,
    );
  }
  assertPositiveInteger(
    data.multi_offspring_desired_count,
    `${basePath}.multi_offspring_desired_count`,
  );
  assertPositiveInteger(
    data.multi_offspring_actual_count,
    `${basePath}.multi_offspring_actual_count`,
  );
  if (data.multi_offspring_desired_count <= 1) {
    throw new Error(
      `Replay reproduction event ${eventIndex} multi-offspring attempt metadata requires desired_count above one.`,
    );
  }
  if (data.multi_offspring_actual_count !== data.offspring_count) {
    throw new Error(
      `Replay reproduction event ${eventIndex} multi_offspring_actual_count must match offspring_count.`,
    );
  }
  if (data.multi_offspring_desired_count < data.multi_offspring_actual_count) {
    throw new Error(
      `Replay reproduction event ${eventIndex} multi_offspring_desired_count cannot be below actual count.`,
    );
  }
  assertStringArray(
    data.multi_offspring_limit_reasons,
    `${basePath}.multi_offspring_limit_reasons`,
  );
  if (
    data.multi_offspring_desired_count > data.multi_offspring_actual_count &&
    data.multi_offspring_limit_reasons.length === 0
  ) {
    throw new Error(
      `Replay reproduction event ${eventIndex} clamped multi-offspring attempt must include a limit reason.`,
    );
  }
  if (
    data.multi_offspring_desired_count === data.multi_offspring_actual_count &&
    data.multi_offspring_limit_reasons.length > 0
  ) {
    throw new Error(
      `Replay reproduction event ${eventIndex} unclamped multi-offspring attempt cannot include limit reasons.`,
    );
  }
}

function validateMultiOffspringGroups(groups) {
  for (const [siblingKey, group] of groups.entries()) {
    if (
      group.seenChildIds.size !== group.offspringCount ||
      group.seenIndexes.size !== group.offspringCount
    ) {
      throw new Error(
        `Replay multi-offspring sibling group ${siblingKey} has incomplete child events.`,
      );
    }
  }
}

function emptyBirthCounts() {
  return {
    asexual_births: 0,
    sexual_births: 0,
    hybrid_births: 0,
  };
}

function incrementCount(counts, key) {
  counts.set(key, (counts.get(key) ?? 0) + 1);
}

function incrementNestedCount(countsByKey, key, nestedKey) {
  const nestedCounts = countsByKey.get(key) ?? new Map();
  countsByKey.set(key, nestedCounts);
  incrementCount(nestedCounts, String(nestedKey));
}

function assertExactCountObject(actual, expected, path) {
  const expectedKeys = [...expected.keys()].sort();
  const actualKeys = Object.keys(actual).sort();
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new Error(`${path} keys do not match agent catalog counts.`);
  }
  for (const key of expectedKeys) {
    if (actual[key] !== expected.get(key)) {
      throw new Error(`${path}.${key} does not match agent catalog counts.`);
    }
  }
}

function highestStage(stageCounts) {
  let highest = null;
  for (const stage of stageCounts.keys()) {
    if (highest === null || stageRank(stage) > stageRank(highest)) {
      highest = stage;
    }
  }
  return highest;
}

function stageRank(stage) {
  const index = REPRODUCTIVE_STAGE_ORDER.indexOf(stage);
  return index >= 0 ? index : 0;
}

function validateSummaryContract(summary, tokenizedCommunication) {
  validateVersion(
    summary.summary_schema_version,
    tokenizedCommunication
      ? TOKENIZED_COMMUNICATION_SUMMARY_SCHEMA_VERSION
      : SUMMARY_SCHEMA_VERSION,
    "Replay summary.summary_schema_version",
  );
}

function validateTrajectory(trajectory) {
  assertObject(trajectory, "Replay viewer.trajectory");
  const tokenizedCommunication =
    trajectory.schema_version ===
    TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION;
  const expectedTrajectorySchema = tokenizedCommunication
    ? TOKENIZED_COMMUNICATION_TRAJECTORY_SCHEMA_VERSION
    : TRAJECTORY_SCHEMA_VERSION;
  const expectedObservationSchema = tokenizedCommunication
    ? TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
    : OBSERVATION_SCHEMA_VERSION;
  const expectedObservationEncoder = tokenizedCommunication
    ? TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION
    : OBSERVATION_ENCODER_VERSION;
  const expectedSignalSchema = tokenizedCommunication
    ? TOKENIZED_COMMUNICATION_SIGNAL_CONTRACT_VERSION
    : SIGNAL_CONTRACT_VERSION;
  validateVersion(
    trajectory.schema_version,
    expectedTrajectorySchema,
    "Replay viewer.trajectory.schema_version",
  );
  validateVersion(
    trajectory.observation_schema_version,
    expectedObservationSchema,
    "Replay viewer.trajectory.observation_schema_version",
  );
  validateVersion(
    trajectory.policy_interface_version,
    POLICY_INTERFACE_VERSION,
    "Replay viewer.trajectory.policy_interface_version",
  );
  validateVersion(
    trajectory.action_contract_version,
    ACTION_CONTRACT_VERSION,
    "Replay viewer.trajectory.action_contract_version",
  );
  validateVersion(
    trajectory.reproductive_group_contract_version,
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
    "Replay viewer.trajectory.reproductive_group_contract_version",
  );
  validateVersion(
    trajectory.genome_recombination_contract_version,
    GENOME_RECOMBINATION_CONTRACT_VERSION,
    "Replay viewer.trajectory.genome_recombination_contract_version",
  );
  validateVersion(
    trajectory.reward_schema_version,
    REWARD_SCHEMA_VERSION,
    "Replay viewer.trajectory.reward_schema_version",
  );
  validateVersion(
    trajectory.action_outcome_schema_version,
    ACTION_OUTCOME_SCHEMA_VERSION,
    "Replay viewer.trajectory.action_outcome_schema_version",
  );
  assertArray(trajectory.records, "Replay viewer.trajectory.records");
  assertObject(
    trajectory.observation_contract,
    "Replay viewer.trajectory.observation_contract",
  );
  const observationContract = trajectory.observation_contract;
  validateVersion(
    observationContract.schema_version,
    expectedObservationSchema,
    "Replay viewer.trajectory.observation_contract.schema_version",
  );
  const tokenFields = validateCommunicationTokenObservationContracts(
    observationContract,
    tokenizedCommunication,
  );
  const expectedObservationSize =
    OBSERVATION_INPUT_VECTOR_SIZE +
    tokenFields.length * (1 + PATCH_CELL_COUNT);
  validateObservationPolicyInput(
    observationContract.policy_input,
    expectedObservationEncoder,
    expectedObservationSize,
    tokenFields,
  );
  validateSchemaContract(
    trajectory.reproductive_group_contract,
    REPRODUCTIVE_GROUP_CONTRACT_VERSION,
    "Replay viewer.trajectory.reproductive_group_contract",
  );
  validateSchemaContract(
    trajectory.genome_recombination_contract,
    GENOME_RECOMBINATION_CONTRACT_VERSION,
    "Replay viewer.trajectory.genome_recombination_contract",
  );
  validateRewardContract(
    trajectory.reward_contract,
    "Replay viewer.trajectory.reward_contract",
  );
  const signalContract = trajectory.observation_contract.signal_contract;
  assertObject(
    signalContract,
    "Replay viewer.trajectory.observation_contract.signal_contract",
  );
  validateVersion(
    signalContract.schema_version,
    expectedSignalSchema,
    "Replay viewer.trajectory.observation_contract.signal_contract.schema_version",
  );
  validateSignalEnablementContract(signalContract);
  validateSignalCommunicationContract(
    signalContract,
    tokenizedCommunication,
    tokenFields,
  );
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

  for (const [index, record] of trajectory.records.entries()) {
    validateTrajectoryRecord(
      record,
      index,
      expectedObservationSchema,
      expectedObservationEncoder,
      expectedObservationSize,
      tokenFields,
    );
  }
  return { tokenizedCommunication };
}

function validateTrajectoryRecord(
  record,
  index,
  expectedObservationSchema,
  expectedObservationEncoder,
  expectedObservationSize,
  tokenFields,
) {
  const path = `Replay viewer.trajectory.records[${index}]`;
  assertObject(record, path);
  for (const field of REQUIRED_TRAJECTORY_RECORD_FIELDS) {
    if (!(field in record)) {
      throw new Error(`Replay trajectory record is missing field ${field}.`);
    }
  }
  validateVersion(
    record.observation_schema,
    expectedObservationSchema,
    `${path}.observation_schema`,
  );
  assertInteger(record.tick, `${path}.tick`);
  assertInteger(record.agent_id, `${path}.agent_id`);
  validateObservationInput(
    record.observation_input,
    `${path}.observation_input`,
    expectedObservationSchema,
    expectedObservationEncoder,
    expectedObservationSize,
    tokenFields,
  );
  assertObject(record.observation_metadata, `${path}.observation_metadata`);
  if (record.observation_metadata.agent_id !== record.agent_id) {
    throw new Error(`${path}.observation_metadata.agent_id must match record agent_id.`);
  }
  assertObject(record.before, `${path}.before`);
  assertObject(record.after, `${path}.after`);
  assertObject(record.outcome, `${path}.outcome`);
  validateVersion(
    record.outcome.schema_version,
    ACTION_OUTCOME_SCHEMA_VERSION,
    `${path}.outcome.schema_version`,
  );
  assertObject(record.outcome.signal, `${path}.outcome.signal`);
  for (const field of REQUIRED_SIGNAL_OUTCOME_FIELDS) {
    if (!(field in record.outcome.signal)) {
      throw new Error(`Replay trajectory signal outcome is missing field ${field}.`);
    }
  }
  if (typeof record.outcome.signal.emitted !== "boolean") {
    throw new Error("Replay trajectory signal outcome must declare emitted.");
  }
  validateReward(record.reward, `${path}.reward`);
}

function validateObservationInput(
  observationInput,
  path,
  expectedObservationSchema,
  expectedObservationEncoder,
  expectedObservationSize,
  tokenFields,
) {
  assertObject(observationInput, path);
  validateVersion(
    observationInput.schema_version,
    expectedObservationSchema,
    `${path}.schema_version`,
  );
  validateVersion(
    observationInput.encoder_version,
    expectedObservationEncoder,
    `${path}.encoder_version`,
  );
  validateVersion(
    observationInput.decoded_dtype,
    OBSERVATION_INPUT_DECODED_DTYPE,
    `${path}.decoded_dtype`,
  );
  validateVersion(
    observationInput.storage_dtype,
    OBSERVATION_INPUT_STORAGE_DTYPE,
    `${path}.storage_dtype`,
  );
  validateVersion(
    observationInput.storage_encoding,
    OBSERVATION_INPUT_STORAGE_ENCODING,
    `${path}.storage_encoding`,
  );
  assertExactNumberArray(
    observationInput.shape,
    [expectedObservationSize],
    `${path}.shape`,
  );
  assertExactNumberArray(
    observationInput.value_range,
    OBSERVATION_INPUT_VALUE_RANGE,
    `${path}.value_range`,
  );
  if (typeof observationInput.data !== "string") {
    throw new Error(`${path}.data must be a string.`);
  }
  validateObservationData(
    observationInput.data,
    expectedObservationSize * Int16Array.BYTES_PER_ELEMENT,
    `${path}.data`,
  );
  if (tokenFields.length > 0) {
    if (observationInput.communication_token_count !== tokenFields.length) {
      throw new Error(`${path}.communication_token_count must match the contract.`);
    }
    assertExactStringArray(
      observationInput.communication_token_field_order,
      tokenFields,
      `${path}.communication_token_field_order`,
    );
  } else if (
    "communication_token_count" in observationInput ||
    "communication_token_field_order" in observationInput
  ) {
    throw new Error(`${path} legacy schema must not declare token channels.`);
  }
}

function validateObservationPolicyInput(
  policyInput,
  expectedObservationEncoder,
  expectedObservationSize,
  tokenFields,
) {
  const path = "Replay viewer.trajectory.observation_contract.policy_input";
  assertObject(policyInput, path);
  validateVersion(
    policyInput.encoder_version,
    expectedObservationEncoder,
    `${path}.encoder_version`,
  );
  validateVersion(
    policyInput.decoded_dtype,
    OBSERVATION_INPUT_DECODED_DTYPE,
    `${path}.decoded_dtype`,
  );
  validateVersion(
    policyInput.storage_dtype,
    OBSERVATION_INPUT_STORAGE_DTYPE,
    `${path}.storage_dtype`,
  );
  validateVersion(
    policyInput.storage_encoding,
    OBSERVATION_INPUT_STORAGE_ENCODING,
    `${path}.storage_encoding`,
  );
  assertExactNumberArray(
    policyInput.shape,
    [expectedObservationSize],
    `${path}.shape`,
  );
  assertExactNumberArray(
    policyInput.value_range,
    OBSERVATION_INPUT_VALUE_RANGE,
    `${path}.value_range`,
  );
  if (policyInput.patch_cell_count !== PATCH_CELL_COUNT) {
    throw new Error(`${path}.patch_cell_count must be ${PATCH_CELL_COUNT}.`);
  }
  assertExactStringArray(
    policyInput.self_input_fields,
    [...OBSERVATION_SELF_INPUT_FIELDS, ...tokenFields],
    `${path}.self_input_fields`,
  );
  assertExactStringArray(
    policyInput.patch_input_fields,
    [...OBSERVATION_PATCH_INPUT_FIELDS, ...tokenFields],
    `${path}.patch_input_fields`,
  );
}

function validateObservationData(data, expectedByteLength, path) {
  let compressed;
  try {
    compressed = decodeBase64(data);
  } catch {
    throw new Error(`${path} must be valid padded base64.`);
  }
  let decoded;
  try {
    decoded = unzlibSync(compressed);
  } catch {
    throw new Error(`${path} must contain a valid zlib stream.`);
  }
  if (decoded.byteLength !== expectedByteLength) {
    throw new Error(
      `${path} must decode to exactly ${expectedByteLength} little-endian int16 bytes.`,
    );
  }
  const view = new DataView(
    decoded.buffer,
    decoded.byteOffset,
    decoded.byteLength,
  );
  for (let offset = 0; offset < decoded.byteLength; offset += 2) {
    view.getInt16(offset, true);
  }
}

function decodeBase64(data) {
  if (
    data.length % 4 !== 0 ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(
      data,
    )
  ) {
    throw new Error("invalid base64");
  }
  if (typeof globalThis.atob === "function") {
    const binary = globalThis.atob(data);
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  }
  if (
    typeof globalThis.Buffer !== "undefined" &&
    typeof globalThis.Buffer.from === "function"
  ) {
    return Uint8Array.from(globalThis.Buffer.from(data, "base64"));
  }
  throw new Error("base64 decoder unavailable");
}

function validateCommunicationTokenObservationContracts(
  observationContract,
  tokenizedCommunication,
) {
  const path =
    "Replay viewer.trajectory.observation_contract.communication_token_channels";
  const tokenContract = observationContract.communication_token_channels;
  let tokenFields = [];
  if (!tokenizedCommunication) {
    if (tokenContract !== undefined) {
      throw new Error(`${path} must be absent for the legacy observation schema.`);
    }
  } else {
    assertObject(tokenContract, path);
    assertExactObjectKeys(
      tokenContract,
      [
        "policy_visible",
        "spatial",
        "field_order",
        "token_order",
        "simulator_assigned_meanings",
        "profile_provenance_policy_visible",
        "aggregate_communication_field_retained",
        "aggregate_projection",
      ],
      path,
    );
    assertStringArray(tokenContract.field_order, `${path}.field_order`);
    if (tokenContract.field_order.length <= 0) {
      throw new Error(`${path}.field_order must not be empty.`);
    }
    tokenFields = tokenContract.field_order.map(
      (_, tokenId) => `communication_signal_token_${tokenId}`,
    );
    assertExactStringArray(
      tokenContract.field_order,
      tokenFields,
      `${path}.field_order`,
    );
    assertExactNumberArray(
      tokenContract.token_order,
      tokenFields.map((_, tokenId) => tokenId),
      `${path}.token_order`,
    );
    if (
      tokenContract.policy_visible !== true ||
      tokenContract.spatial !== true ||
      tokenContract.simulator_assigned_meanings !== false ||
      tokenContract.profile_provenance_policy_visible !== false ||
      tokenContract.aggregate_communication_field_retained !== true ||
      tokenContract.aggregate_projection !== COMMUNICATION_AGGREGATE_PROJECTION
    ) {
      throw new Error(
        `${path} must declare the canonical opaque spatial token projection.`,
      );
    }
  }
  assertExactStringArray(
    observationContract.self_fields,
    [...OBSERVATION_SELF_FIELDS, ...tokenFields],
    "Replay viewer.trajectory.observation_contract.self_fields",
  );
  assertExactStringArray(
    observationContract.patch_fields,
    [...OBSERVATION_PATCH_FIELDS, ...tokenFields],
    "Replay viewer.trajectory.observation_contract.patch_fields",
  );
  return tokenFields;
}

function validateRewardContract(contract, path) {
  assertExactJson(contract, REWARD_CONTRACT, path);
}

function validateReward(reward, path) {
  assertObject(reward, path);
  assertExactObjectKeys(reward, REWARD_RECORD_FIELDS, path);
  validateVersion(
    reward.schema_version,
    REWARD_SCHEMA_VERSION,
    `${path}.schema_version`,
  );
  assertObject(reward.components, `${path}.components`);
  const componentBounds = REWARD_CONTRACT.component_bounds;
  assertExactObjectKeys(
    reward.components,
    Object.keys(componentBounds),
    `${path}.components`,
  );
  let componentSum = 0;
  for (const [component, bounds] of Object.entries(componentBounds)) {
    const value = reward.components[component];
    assertFiniteNumber(value, `${path}.components.${component}`);
    if (value < bounds[0] || value > bounds[1]) {
      throw new Error(
        `${path}.components.${component} must be within [${bounds.join(", ")}].`,
      );
    }
    componentSum += value;
  }
  assertFiniteNumber(reward.total, `${path}.total`);
  const totalBounds = REWARD_CONTRACT.total_bounds;
  if (reward.total < totalBounds[0] || reward.total > totalBounds[1]) {
    throw new Error(
      `${path}.total must be within [${totalBounds.join(", ")}].`,
    );
  }
  const expectedTotal = Number(componentSum.toFixed(4));
  if (reward.total !== expectedTotal) {
    throw new Error(
      `${path}.total must equal the four-decimal component sum ${expectedTotal}.`,
    );
  }
}

function validateSignalCommunicationContract(
  signalContract,
  tokenizedCommunication,
  tokenFields,
) {
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
  if (
    tokenizedCommunication &&
    signalContract.communication_token_count !== tokenFields.length
  ) {
    throw new Error(
      "Replay signal contract communication_token_count must match token field order.",
    );
  }
  if (
    signalContract.communication_signal_emission_enabled !==
    tokenizedCommunication
  ) {
    throw new Error(
      "Replay signal contract communication enablement must match trajectory schema.",
    );
  }
  if (
    signalContract.communication_tokens_have_simulator_assigned_meaning !==
    false
  ) {
    throw new Error(
      "Replay signal contract communication tokens must remain simulator-opaque.",
    );
  }
  assertExactStringArray(
    signalContract.fields,
    [...REQUIRED_SIGNAL_FIELDS, ...tokenFields],
    "Replay signal contract fields",
  );
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
      profile.field_name !==
        (tokenizedCommunication
          ? tokenFields[expectedTokenId]
          : "communication_signal") ||
      profile.profile_id !==
        `communication_token_${expectedTokenId}_profile_${expectedProfileIndex}` ||
      profile.source_kind !== "reserved_opaque_communication" ||
      profile.token_id !== expectedTokenId ||
      profile.profile_index !== expectedProfileIndex ||
      profile.policy_visible !== false
    ) {
      throw new Error(
        "Replay signal contract reserved communication profiles must be ordered, opaque, and policy-hidden.",
      );
    }
  }
  const tokenContract = signalContract.communication_token_observation;
  if (!tokenizedCommunication) {
    if (tokenContract !== undefined) {
      throw new Error(
        "Legacy signal contracts must not declare token observation channels.",
      );
    }
    return;
  }
  assertObject(tokenContract, "Replay signal contract communication_token_observation");
  assertExactObjectKeys(
    tokenContract,
    [
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
    ],
    "Replay signal contract communication_token_observation",
  );
  validateVersion(
    tokenContract.schema_version,
    TOKENIZED_COMMUNICATION_SIGNAL_REPORTING_VERSION,
    "Replay signal contract communication_token_observation.schema_version",
  );
  assertExactStringArray(
    tokenContract.field_order,
    tokenFields,
    "Replay signal contract communication_token_observation.field_order",
  );
  assertExactNumberArray(
    tokenContract.token_order,
    tokenFields.map((_, tokenId) => tokenId),
    "Replay signal contract communication_token_observation.token_order",
  );
  if (
    signalContract.communication_signal_emission_enabled !== true ||
    tokenContract.policy_visible !== true ||
    tokenContract.spatial !== true ||
    tokenContract.simulator_assigned_meanings !== false ||
    tokenContract.profile_provenance_policy_visible !== false ||
    tokenContract.aggregate_field !== "communication_signal" ||
    tokenContract.aggregate_field_retained_for_compatibility !== true ||
    tokenContract.aggregation !== COMMUNICATION_AGGREGATE_PROJECTION ||
    tokenContract.receiver_projection_schema_version !==
      "foundation_communication_receiver_projection_v1" ||
    tokenContract.receiver_observation_policy !==
      "exclude_receiver_own_communication_emissions_across_local_patch_v1" ||
    tokenContract.global_reporting_policy !== "include_all_emitters_v1"
  ) {
    throw new Error(
      "Token-aware signal contract must declare opaque public spatial token channels.",
    );
  }
}

function validateSignalEnablementContract(signalContract) {
  for (const field of [
    "signal_substrate_enabled",
    "reproductive_signal_emission_enabled",
    "communication_signal_emission_enabled",
  ]) {
    if (typeof signalContract[field] !== "boolean") {
      throw new Error(`Replay signal contract ${field} must be boolean.`);
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
  assertExactObjectKeys(actionContract, ACTION_CONTRACT_FIELDS, path);
  validateVersion(
    actionContract.schema_version,
    ACTION_CONTRACT_VERSION,
    `${path}.schema_version`,
  );
  if (actionContract.policy_id_encoding !== ACTION_POLICY_ID_ENCODING) {
    throw new Error(
      `${path}.policy_id_encoding must be ${ACTION_POLICY_ID_ENCODING}.`,
    );
  }
  if (actionContract.debug_key_encoding !== ACTION_DEBUG_KEY_ENCODING) {
    throw new Error(
      `${path}.debug_key_encoding must be ${ACTION_DEBUG_KEY_ENCODING}.`,
    );
  }
  if (actionContract.mate_action_key !== ACTION_MATE_ACTION_KEY) {
    throw new Error(
      `${path}.mate_action_key must be ${ACTION_MATE_ACTION_KEY}.`,
    );
  }
  assertObject(actionContract.communication, `${path}.communication`);
  assertExactObjectKeys(
    actionContract.communication,
    ACTION_COMMUNICATION_FIELDS,
    `${path}.communication`,
  );
  if (
    actionContract.communication.emission_enabled !==
    signalContract.communication_signal_emission_enabled
  ) {
    throw new Error(
      `${path}.communication.emission_enabled must match signal contract communication enablement.`,
    );
  }
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
  assertStringArray(actionContract.active_action_keys, `${path}.active_action_keys`);
  assertStringArray(actionContract.reserved_action_keys, `${path}.reserved_action_keys`);
  assertExactStringArray(
    actionContract.communication.action_keys,
    expectedActionKeys,
    `${path}.communication.action_keys`,
  );
  if (
    actionContract.communication.meaning !== ACTION_COMMUNICATION_MEANING
  ) {
    throw new Error(
      `${path}.communication.meaning must be ${ACTION_COMMUNICATION_MEANING}.`,
    );
  }

  const communicationSpecs = expectedActionKeys.map((key, index) => ({
    action_id: ACTION_NON_COMMUNICATION_SPECS.length + index,
    key,
    ...ACTION_COMMUNICATION_SPEC_FIXED_FIELDS,
    active: signalContract.communication_signal_emission_enabled,
    debug_label: key,
  }));
  const expectedSpecs = [
    ...ACTION_NON_COMMUNICATION_SPECS,
    ...communicationSpecs,
  ];
  const expectedActiveKeys = expectedSpecs
    .filter((spec) => spec.active)
    .map((spec) => spec.key);
  const expectedReservedKeys = expectedSpecs
    .filter((spec) => spec.reserved)
    .map((spec) => spec.key);
  assertExactStringArray(
    actionContract.active_action_keys,
    expectedActiveKeys,
    `${path}.active_action_keys`,
  );
  assertExactStringArray(
    actionContract.reserved_action_keys,
    expectedReservedKeys,
    `${path}.reserved_action_keys`,
  );
  assertArray(actionContract.actions, `${path}.actions`);
  for (const [index, spec] of actionContract.actions.entries()) {
    assertObject(spec, `${path}.actions[${index}]`);
    assertExactObjectKeys(spec, ACTION_SPEC_FIELDS, `${path}.actions[${index}]`);
  }
  assertExactJson(actionContract.actions, expectedSpecs, `${path}.actions`);

  if (observationActionNames !== undefined) {
    assertExactStringArray(
      observationActionNames,
      expectedSpecs.map((spec) => spec.key),
      "Replay observation action_names",
    );
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

function validateFrames(frames, map, summary, agentEncoding, trajectory, agentCatalog) {
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
  const signalContract = trajectory.observation_contract.signal_contract;
  const signalContractVersion = signalContract.schema_version;
  const requiredSignalFields = Array.isArray(signalContract.fields)
    ? signalContract.fields
    : REQUIRED_SIGNAL_FIELDS;
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
    validateAgents(
      frame.agents,
      agentEncoding.length,
      agentFieldMap,
      agentCatalog,
      map,
      index,
    );
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
      requiredSignalFields,
      map,
      `Replay frame ${index}.signal_fields`,
    );
    validateSignalEmissions(frame.signal_emissions, signalContractVersion, index);
  }
}

function validateAgents(
  agents,
  encodedLength,
  agentFieldMap,
  agentCatalog,
  map,
  frameIndex,
) {
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
    if (!(String(agentId) in agentCatalog)) {
      throw new Error(
        `Replay frame ${frameIndex}.agents[${agentIndex}] references missing agent catalog entry.`,
      );
    }
    assertInteger(x, `Replay frame ${frameIndex}.agents[${agentIndex}].x`);
    assertInteger(y, `Replay frame ${frameIndex}.agents[${agentIndex}].y`);
    if (x < 0 || x >= map.width || y < 0 || y >= map.height) {
      throw new Error(
        `Replay frame ${frameIndex}.agents[${agentIndex}] position is outside the map.`,
      );
    }
  }
}

function validateSchemaContract(contract, expected, path) {
  assertObject(contract, path);
  validateVersion(contract.schema_version, expected, `${path}.schema_version`);
}

function validateVersion(actual, expected, path) {
  if (actual !== expected) {
    throw new Error(`${path} must be ${expected}.`);
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
  assertStringArray(actual, path);
  if (actual.length !== expected.length) {
    throw new Error(`${path} must contain ${expected.length} entries.`);
  }
  for (const [index, expectedValue] of expected.entries()) {
    if (actual[index] !== expectedValue) {
      throw new Error(`${path}[${index}] must be ${expectedValue}.`);
    }
  }
}

function assertExactObjectKeys(actual, expectedKeys, path) {
  assertObject(actual, path);
  const actualKeys = Object.keys(actual);
  const expected = new Set(expectedKeys);
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key) => !expected.has(key))
  ) {
    throw new Error(
      `${path} keys must be exactly [${expectedKeys.join(", ")}].`,
    );
  }
}

function assertExactJson(actual, expected, path) {
  if (Array.isArray(expected)) {
    assertArray(actual, path);
    if (actual.length !== expected.length) {
      throw new Error(`${path} must contain ${expected.length} entries.`);
    }
    for (const [index, expectedValue] of expected.entries()) {
      assertExactJson(actual[index], expectedValue, `${path}[${index}]`);
    }
    return;
  }
  if (expected && typeof expected === "object") {
    assertObject(actual, path);
    const expectedKeys = Object.keys(expected);
    assertExactObjectKeys(actual, expectedKeys, path);
    for (const key of expectedKeys) {
      assertExactJson(actual[key], expected[key], `${path}.${key}`);
    }
    return;
  }
  if (actual !== expected) {
    throw new Error(`${path} must match the canonical contract value.`);
  }
}

function assertExactNumberArray(actual, expected, path) {
  assertArray(actual, path);
  if (
    actual.length !== expected.length ||
    actual.some((value, index) => value !== expected[index])
  ) {
    throw new Error(`${path} must be [${expected.join(", ")}].`);
  }
}

function assertFiniteNumber(value, path) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path} must be a finite number.`);
  }
}

function assertPositiveInteger(value, path) {
  assertInteger(value, path);
  if (value <= 0) {
    throw new Error(`${path} must be greater than zero.`);
  }
}

function assertNonnegativeInteger(value, path) {
  assertInteger(value, path);
  if (value < 0) {
    throw new Error(`${path} must be nonnegative.`);
  }
}

function assertInteger(value, path) {
  if (!Number.isInteger(value)) {
    throw new Error(`${path} must be an integer.`);
  }
}
