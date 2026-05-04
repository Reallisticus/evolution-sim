const SIGNIFICANT_EVENT_TYPES = new Set([
  "agent_reproduced",
  "agent_died",
  "agent_attacked",
  "carcass_deposited",
]);

const AGENT_EVENT_DATA_KEYS = [
  "target_id",
  "child_id",
  "source_agent_id",
  "killer_id",
];
const AGENT_EVENT_ARRAY_KEYS = ["parent_ids"];
const AGENT_EVENT_AGENT_OBJECT_KEYS = ["parent_energy_costs"];
const AGENT_EVENT_BREAKDOWN_KEYS = [
  "source_breakdown",
  "deposit_breakdown",
  "tile_source_breakdown_after",
  "tile_fresh_kill_source_breakdown_after",
];

export function buildReplayModel(payload, decodeAgent) {
  const frames = payload?.viewer?.frames ?? [];
  const events = payload?.events ?? [];
  const decodedFrames = frames.map((frame, frameIndex) => {
    const tick = Number(frame.tick);
    return {
      frame,
      frameIndex,
      tick,
      agents: frame.agents.map((encoded) => decodeAgent(encoded)),
    };
  });
  const frameIndexByTick = new Map(decodedFrames.map((entry) => [entry.tick, entry.frameIndex]));
  const eventsByTick = groupEventsByTick(events);
  const significantEventTicks = buildSignificantEventTicks(events);
  const episodes = buildEventEpisodes(significantEventTicks);
  const agentEventsById = buildAgentEventsById(events);
  const agentPositionsById = buildAgentPositionsById(decodedFrames);
  const trajectoryRecordsByAgentId = buildTrajectoryRecordsByAgentId(
    payload?.viewer?.trajectory?.records ?? [],
  );

  return {
    decodedFrames,
    frameIndexByTick,
    eventsByTick,
    significantEventTicks,
    episodes,
    agentEventsById,
    agentPositionsById,
    trajectoryRecordsByAgentId,
    cacheStats: {
      decodedFrameCount: decodedFrames.length,
      eventTickCount: eventsByTick.size,
      significantTickCount: significantEventTicks.length,
      episodeCount: episodes.length,
      agentEventIndexSize: agentEventsById.size,
      agentPositionIndexSize: agentPositionsById.size,
      trajectoryAgentIndexSize: trajectoryRecordsByAgentId.size,
    },
  };
}

function groupEventsByTick(events) {
  const grouped = new Map();
  for (const event of events) {
    const tick = Number(event.tick);
    if (!Number.isFinite(tick)) continue;
    if (!grouped.has(tick)) {
      grouped.set(tick, []);
    }
    grouped.get(tick).push(event);
  }
  return grouped;
}

function buildAgentEventsById(events) {
  const byId = new Map();
  for (const event of events) {
    for (const agentId of agentIdsForEvent(event)) {
      if (!byId.has(agentId)) {
        byId.set(agentId, []);
      }
      byId.get(agentId).push(event);
    }
  }
  for (const records of byId.values()) {
    records.sort((left, right) => {
      if (left.tick !== right.tick) return left.tick - right.tick;
      return String(left.type).localeCompare(String(right.type));
    });
  }
  return byId;
}

function buildAgentPositionsById(decodedFrames) {
  const byId = new Map();
  for (const { tick, frameIndex, agents } of decodedFrames) {
    for (const agent of agents) {
      if (!byId.has(agent.agentId)) {
        byId.set(agent.agentId, []);
      }
      byId.get(agent.agentId).push({ ...agent, tick, frameIndex });
    }
  }
  return byId;
}

function buildTrajectoryRecordsByAgentId(records) {
  const byId = new Map();
  for (const record of records) {
    const agentId = Number(record.agent_id);
    if (!Number.isFinite(agentId)) continue;
    if (!byId.has(agentId)) {
      byId.set(agentId, []);
    }
    byId.get(agentId).push(record);
  }
  for (const agentRecords of byId.values()) {
    agentRecords.sort((left, right) => {
      if (left.tick !== right.tick) return left.tick - right.tick;
      return String(left.requested_action).localeCompare(String(right.requested_action));
    });
  }
  return byId;
}

function agentIdsForEvent(event) {
  const ids = new Set();
  addFiniteId(ids, event.agent_id);
  const data = event.data ?? {};
  for (const key of AGENT_EVENT_DATA_KEYS) {
    addFiniteId(ids, data[key]);
  }
  for (const key of AGENT_EVENT_ARRAY_KEYS) {
    for (const value of data[key] ?? []) {
      addFiniteId(ids, value);
    }
  }
  for (const key of AGENT_EVENT_AGENT_OBJECT_KEYS) {
    for (const entry of data[key] ?? []) {
      addFiniteId(ids, entry?.agent_id);
    }
  }
  for (const key of AGENT_EVENT_BREAKDOWN_KEYS) {
    for (const entry of data[key] ?? []) {
      addFiniteId(ids, entry?.source_agent_id);
      addFiniteId(ids, entry?.killer_id);
    }
  }
  return ids;
}

function addFiniteId(ids, value) {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    ids.add(numeric);
  }
}

function buildSignificantEventTicks(events) {
  const byTick = new Map();
  for (const event of events) {
    if (!SIGNIFICANT_EVENT_TYPES.has(event.type)) continue;
    const tick = Number(event.tick);
    if (!Number.isFinite(tick)) continue;
    if (!byTick.has(tick)) {
      byTick.set(tick, {
        tick,
        births: 0,
        deaths: 0,
        attacks: 0,
        carcasses: 0,
        total: 0,
        hasBirths: false,
        hasDeaths: false,
        hasCombat: false,
        hasCarcass: false,
      });
    }
    const entry = byTick.get(tick);
    entry.total += 1;
    if (event.type === "agent_reproduced") {
      entry.births += 1;
      entry.hasBirths = true;
    } else if (event.type === "agent_died") {
      entry.deaths += 1;
      entry.hasDeaths = true;
    } else if (event.type === "agent_attacked" || event.type === "agent_damaged") {
      entry.attacks += 1;
      entry.hasCombat = true;
    } else if (event.type === "carcass_deposited") {
      entry.carcasses += 1;
      entry.hasCarcass = true;
    }
  }
  return [...byTick.values()].sort((left, right) => left.tick - right.tick);
}

function buildEventEpisodes(entries) {
  return entries.map((entry) => {
    const kind = entry.hasDeaths
      ? "death"
      : entry.hasBirths
        ? "birth"
        : entry.hasCarcass
          ? "carcass"
          : entry.hasCombat
            ? "combat"
            : "event";
    return {
      ...entry,
      kind,
      severity: entry.hasDeaths || entry.hasCombat ? "high" : entry.hasCarcass ? "medium" : "low",
    };
  });
}
