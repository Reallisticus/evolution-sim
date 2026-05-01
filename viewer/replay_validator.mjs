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

const TRAJECTORY_SCHEMA_VERSION = "mind_trajectory_v1";
const OBSERVATION_SCHEMA_VERSION = "mind_observation_v3";
const OBSERVATION_ENCODER_VERSION = "mind_observation_encoder_v2";
const OBSERVATION_INPUT_DECODED_DTYPE = "float32";
const OBSERVATION_INPUT_STORAGE_DTYPE = "int16";
const OBSERVATION_INPUT_STORAGE_ENCODING = "zlib_base64_little_endian_int16";
const OBSERVATION_INPUT_VECTOR_SIZE = 542;
const OBSERVATION_INPUT_VALUE_RANGE = [-1, 1];
const POLICY_INTERFACE_VERSION = "mind_policy_interface_v1";
const ACTION_CONTRACT_VERSION = "mind_action_contract_v1";
const SIGNAL_CONTRACT_VERSION = "foundation_signal_contract_v2";
const REPRODUCTIVE_GROUP_CONTRACT_VERSION = "reproductive_group_contract_v1";
const GENOME_RECOMBINATION_CONTRACT_VERSION = "genome_recombination_contract_v1";
const REWARD_SCHEMA_VERSION = "mind_reward_v1";
const ACTION_OUTCOME_SCHEMA_VERSION = "mind_action_outcome_v2";
const REPRODUCTION_EVENT_SCHEMA_VERSION = "reproduction_event_v1";
const ASEXUAL_REPRODUCTION_MODE = "asexual";
const SEXUAL_REPRODUCTION_MODE = "same_group_sexual";
const REQUIRED_MAP_FIELDS = ["fertility", "moisture", "heat"];
const REQUIRED_AGENT_CATALOG_FIELDS = [
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
];
const REQUIRED_REPRODUCTIVE_GROUP_FIELDS = [
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
];
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
const REQUIRED_SIGNAL_OUTCOME_FIELDS = [
  "emitted",
  "token_id",
  "profile_index",
  "intensity",
  "radius",
  "duration_ticks",
  "decay_rate",
  "energy_cost",
  "invalid_reason",
];
const REQUIRED_TRAJECTORY_RECORD_FIELDS = [
  "tick",
  "agent_id",
  "lineage_id",
  "runtime_species_id",
  "runtime_ecotype_id",
  "observation_schema",
  "observation_metadata",
  "observation_input",
  "observation_digest",
  "action_mask",
  "resolution_action_mask",
  "requested_action",
  "action_source",
  "policy_id",
  "policy_version",
  "action_valid",
  "resolution_action_valid",
  "resolved_action",
  "moved",
  "before",
  "after",
  "outcome",
  "reward",
];
const REPRODUCTIVE_STAGE_ORDER = [
  "stage0_asexual",
  "stage1_facultative_sex",
  "stage2_proto_roles",
  "stage3_x_y_z",
  "stage4_hybridization",
];
const REPRODUCTIVE_EXPRESSIONS = [
  "asexual",
  "same_group_compatible",
  "proto_x_like",
  "proto_y_like",
  "proto_z_plastic",
  "x",
  "y",
  "z_plastic",
];

export function validateReplayPayload(payload) {
  assertObject(payload, "Replay payload");
  assertObject(payload.summary, "Replay summary");
  assertObject(payload.viewer, "Replay viewer");

  const { summary, viewer } = payload;
  const map = validateMap(viewer.map);
  validateAgentEncoding(viewer.agent_encoding);
  validateTrajectory(viewer.trajectory);
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

function validateTrajectory(trajectory) {
  assertObject(trajectory, "Replay viewer.trajectory");
  validateVersion(
    trajectory.schema_version,
    TRAJECTORY_SCHEMA_VERSION,
    "Replay viewer.trajectory.schema_version",
  );
  validateVersion(
    trajectory.observation_schema_version,
    OBSERVATION_SCHEMA_VERSION,
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
    OBSERVATION_SCHEMA_VERSION,
    "Replay viewer.trajectory.observation_contract.schema_version",
  );
  validateObservationPolicyInput(observationContract.policy_input);
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
  validateSchemaContract(
    trajectory.reward_contract,
    REWARD_SCHEMA_VERSION,
    "Replay viewer.trajectory.reward_contract",
  );
  const signalContract = trajectory.observation_contract.signal_contract;
  assertObject(
    signalContract,
    "Replay viewer.trajectory.observation_contract.signal_contract",
  );
  validateVersion(
    signalContract.schema_version,
    SIGNAL_CONTRACT_VERSION,
    "Replay viewer.trajectory.observation_contract.signal_contract.schema_version",
  );
  validateSignalEnablementContract(signalContract);
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

  for (const [index, record] of trajectory.records.entries()) {
    validateTrajectoryRecord(record, index);
  }
}

function validateTrajectoryRecord(record, index) {
  const path = `Replay viewer.trajectory.records[${index}]`;
  assertObject(record, path);
  for (const field of REQUIRED_TRAJECTORY_RECORD_FIELDS) {
    if (!(field in record)) {
      throw new Error(`Replay trajectory record is missing field ${field}.`);
    }
  }
  validateVersion(
    record.observation_schema,
    OBSERVATION_SCHEMA_VERSION,
    `${path}.observation_schema`,
  );
  assertInteger(record.tick, `${path}.tick`);
  assertInteger(record.agent_id, `${path}.agent_id`);
  validateObservationInput(record.observation_input, `${path}.observation_input`);
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
  assertObject(record.reward, `${path}.reward`);
  validateVersion(
    record.reward.schema_version,
    REWARD_SCHEMA_VERSION,
    `${path}.reward.schema_version`,
  );
}

function validateObservationInput(observationInput, path) {
  assertObject(observationInput, path);
  validateVersion(
    observationInput.schema_version,
    OBSERVATION_SCHEMA_VERSION,
    `${path}.schema_version`,
  );
  validateVersion(
    observationInput.encoder_version,
    OBSERVATION_ENCODER_VERSION,
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
    [OBSERVATION_INPUT_VECTOR_SIZE],
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
}

function validateObservationPolicyInput(policyInput) {
  const path = "Replay viewer.trajectory.observation_contract.policy_input";
  assertObject(policyInput, path);
  validateVersion(
    policyInput.encoder_version,
    OBSERVATION_ENCODER_VERSION,
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
    [OBSERVATION_INPUT_VECTOR_SIZE],
    `${path}.shape`,
  );
  assertExactNumberArray(
    policyInput.value_range,
    OBSERVATION_INPUT_VALUE_RANGE,
    `${path}.value_range`,
  );
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
  validateVersion(
    actionContract.schema_version,
    ACTION_CONTRACT_VERSION,
    `${path}.schema_version`,
  );
  assertObject(actionContract.communication, `${path}.communication`);
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

  const reserved = new Set(actionContract.reserved_action_keys);
  const active = new Set(actionContract.active_action_keys);
  for (const actionKey of expectedActionKeys) {
    if (!reserved.has(actionKey)) {
      throw new Error(`${path}.reserved_action_keys is missing ${actionKey}.`);
    }
    if (signalContract.communication_signal_emission_enabled) {
      if (!active.has(actionKey)) {
        throw new Error(`${path}.active_action_keys is missing active ${actionKey}.`);
      }
    } else if (active.has(actionKey)) {
      throw new Error(
        `${path}.active_action_keys must not include inactive ${actionKey}.`,
      );
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
      REQUIRED_SIGNAL_FIELDS,
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
  if (actual.length !== expected.length) {
    throw new Error(`${path} must contain ${expected.length} entries.`);
  }
  for (const [index, expectedValue] of expected.entries()) {
    if (actual[index] !== expectedValue) {
      throw new Error(`${path}[${index}] must be ${expectedValue}.`);
    }
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
