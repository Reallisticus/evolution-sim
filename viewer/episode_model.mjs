export const EPISODE_SUPPORT_EVENT_TYPES = new Set([
  "agent_moved",
  "agent_damaged",
  "agent_healed",
  "agent_ate",
  "agent_drank",
]);

const EVENT_MARKER_TYPES = new Set([
  "agent_reproduced",
  "agent_died",
  "carcass_deposited",
  "agent_attacked",
  "agent_damaged",
  "agent_healed",
  "agent_ate",
  "agent_drank",
]);

const EVENT_TYPE_PRIORITY = {
  agent_reproduced: 0,
  agent_died: 0,
  carcass_deposited: 1,
  agent_attacked: 1,
  agent_damaged: 2,
  agent_healed: 3,
  agent_ate: 3,
  agent_drank: 3,
};

export function eventLensCategory(event) {
  if (event.type === "agent_reproduced") return "birth";
  if (event.type === "agent_died") return "death";
  if (event.type === "agent_attacked") return "combat";
  if (event.type === "carcass_deposited") return "carcass";
  if (
    event.type === "agent_ate" &&
    ["carcass", "fresh_kill"].includes(String(event.data?.food_source ?? ""))
  ) {
    return "carcass";
  }
  return null;
}

export function eventLensTicks(currentTick, mode, playback, frameWindow = 24) {
  if (mode === "current") return [currentTick];
  if (playback) {
    const startTick = Number(playback.startTick ?? currentTick);
    const endTick = Math.min(Number(currentTick), Number(playback.endTick ?? currentTick));
    if (!Number.isFinite(startTick) || !Number.isFinite(endTick) || endTick < startTick) return [currentTick];
    return Array.from({ length: endTick - startTick + 1 }, (_, index) => startTick + index);
  }
  return Array.from({ length: frameWindow + 1 }, (_, index) => currentTick - index)
    .filter((tick) => tick >= 0)
    .reverse();
}

export function eventMatchesLens(event, mode) {
  if (mode === "current") return EVENT_MARKER_TYPES.has(event.type);
  const category = eventLensCategory(event);
  if (mode === "causal") return category != null;
  if (mode === "births") return category === "birth";
  if (mode === "deaths") return category === "death";
  if (mode === "combat") return category === "combat";
  if (mode === "carcass") return category === "carcass";
  return false;
}

export function primaryEpisodeEvents(events) {
  const primary = events.filter((event) => eventLensCategory(event) != null);
  if (primary.length > 0) return [...primary].sort(sortEpisodeEvents);
  return events.filter(isEpisodeSupportEvent).sort(sortEpisodeEvents);
}

export function isEpisodeSupportEvent(event) {
  if (!EPISODE_SUPPORT_EVENT_TYPES.has(event.type)) return false;
  return eventLensCategory(event) == null;
}

export function sortEpisodeEvents(left, right) {
  const leftPriority = EVENT_TYPE_PRIORITY[left.type] ?? 9;
  const rightPriority = EVENT_TYPE_PRIORITY[right.type] ?? 9;
  return leftPriority - rightPriority || String(left.type).localeCompare(String(right.type));
}

export function buildEpisodePlaybackScope(events, markerForEvent) {
  const primary = primaryEpisodeEvents(events).filter((event) => eventLensCategory(event) != null);
  const agentIds = new Set();
  const tileKeys = new Set();
  for (const event of primary) {
    for (const { agentId } of eventAgentRoles(event)) {
      agentIds.add(agentId);
    }
    const marker = markerForEvent(event);
    addPlaybackTileKey(tileKeys, marker);
    addPlaybackTileKey(tileKeys, marker?.from);
  }
  return {
    agentIds: [...agentIds].sort((left, right) => left - right),
    tileKeys: [...tileKeys].sort(),
    primaryEventCount: primary.length,
  };
}

export function eventTouchesPlaybackScope(event, scope, markerForEvent) {
  const agentIds = new Set(scope?.agentIds ?? []);
  const tileKeys = new Set(scope?.tileKeys ?? []);
  if (agentIds.size === 0 && tileKeys.size === 0) return true;

  for (const { agentId } of eventAgentRoles(event)) {
    if (agentIds.has(agentId)) return true;
  }

  const marker = markerForEvent(event);
  return playbackTileInScope(marker, tileKeys) || playbackTileInScope(marker?.from, tileKeys);
}

function addPlaybackTileKey(tileKeys, point) {
  const key = playbackTileKey(point);
  if (key) tileKeys.add(key);
}

function playbackTileInScope(point, tileKeys) {
  const key = playbackTileKey(point);
  return key ? tileKeys.has(key) : false;
}

export function playbackTileKey(point) {
  if (!point) return null;
  const x = Number(point.x);
  const y = Number(point.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return `${Math.trunc(x)},${Math.trunc(y)}`;
}

export function eventCategoryCounts(events) {
  const counts = {
    birth: 0,
    death: 0,
    combat: 0,
    carcass: 0,
    moves: 0,
    meatMeals: 0,
    support: 0,
  };
  for (const event of events) {
    const category = eventLensCategory(event);
    if (category && counts[category] != null) {
      counts[category] += 1;
    } else if (event.type === "agent_moved") {
      counts.moves += 1;
    } else {
      counts.support += 1;
    }
    if (
      event.type === "agent_ate" &&
      ["carcass", "fresh_kill"].includes(String(event.data?.food_source ?? ""))
    ) {
      counts.meatMeals += 1;
    }
  }
  return counts;
}

export function episodeCausalitySentence(counts, deltas, options = {}) {
  const signedValue = options.signedValue ?? defaultSignedValue;
  if (counts.combat > 0 && counts.death > 0) {
    return `Predation is visible end-to-end: ${counts.combat} attack${counts.combat === 1 ? "" : "s"} led into ${counts.death} death${counts.death === 1 ? "" : "s"}, with ${counts.carcass} carcass-flow event${counts.carcass === 1 ? "" : "s"} for the food web.`;
  }
  if (counts.birth > 0) {
    return `Reproduction expanded the population by ${counts.birth} birth${counts.birth === 1 ? "" : "s"} while the live-agent delta was ${signedValue(deltas.aliveDelta)}.`;
  }
  if (counts.death > 0 || counts.carcass > 0) {
    return `Mortality changed the resource layer: ${counts.death} death${counts.death === 1 ? "" : "s"}, ${counts.carcass} carcass-flow event${counts.carcass === 1 ? "" : "s"}, and a carcass tile delta of ${signedValue(deltas.carcassTileDelta)}.`;
  }
  if (counts.combat > 0) {
    return `${counts.combat} attack${counts.combat === 1 ? "" : "s"} created local pressure without an immediate recorded death on this tick.`;
  }
  return `The episode is mostly support motion: ${counts.moves} movement${counts.moves === 1 ? "" : "s"} and ${deltas.tileChanges} changed tile layer${deltas.tileChanges === 1 ? "" : "s"}.`;
}

export function episodeLensModes(counts) {
  const modes = ["causal", "delta"];
  if (counts.combat > 0) modes.push("combat");
  if (counts.birth > 0) modes.push("births");
  if (counts.death > 0) modes.push("deaths");
  if (counts.carcass > 0 || counts.meatMeals > 0) modes.push("carcass");
  return [...new Set(modes)].slice(0, 5);
}

export function episodeContextForTick(episodes, tick) {
  if (episodes.length === 0) return null;
  const current = episodes.find((episode) => Number(episode.tick) === Number(tick));
  if (current) return { episode: current, relation: "current", distance: 0 };
  const previous = [...episodes].reverse().find((episode) => Number(episode.tick) < Number(tick));
  const next = episodes.find((episode) => Number(episode.tick) > Number(tick));
  if (!previous) {
    return { episode: next, relation: "future", distance: Math.abs(Number(next.tick) - Number(tick)) };
  }
  if (!next) {
    return { episode: previous, relation: "past", distance: Math.abs(Number(tick) - Number(previous.tick)) };
  }
  const previousDistance = Math.abs(Number(tick) - Number(previous.tick));
  const nextDistance = Math.abs(Number(next.tick) - Number(tick));
  return previousDistance <= nextDistance
    ? { episode: previous, relation: "past", distance: previousDistance }
    : { episode: next, relation: "future", distance: nextDistance };
}

export function episodeAgentRoleEntries(events, options = {}) {
  const limit = Number(options.limit ?? 10);
  const agents = new Map();
  for (const event of primaryEpisodeEvents(events)) {
    for (const { agentId, role } of eventAgentRoles(event)) {
      if (!agents.has(agentId)) {
        agents.set(agentId, { agentId, roles: new Set() });
      }
      agents.get(agentId).roles.add(role);
    }
  }
  return [...agents.values()]
    .map((entry) => ({
      agentId: entry.agentId,
      roles: [...entry.roles].sort(sortAgentEpisodeRoles),
      roleLabel: [...entry.roles].sort(sortAgentEpisodeRoles).join(" / "),
    }))
    .sort((left, right) => sortAgentEpisodeRoles(left.roleLabel, right.roleLabel) || left.agentId - right.agentId)
    .slice(0, Number.isFinite(limit) ? limit : 10);
}

export function eventAgentRoles(event) {
  const data = event.data ?? {};
  const roles = [];
  const add = (id, role) => {
    const agentId = Number(id);
    if (!Number.isFinite(agentId)) return;
    roles.push({ agentId, role });
  };
  const addCarcassBreakdownRoles = () => {
    const entries = [...(data.source_breakdown ?? []), ...(data.deposit_breakdown ?? [])];
    for (const entry of entries) {
      add(entry.source_agent_id, "source");
      add(entry.killer_id, "killer");
    }
  };
  if (event.type === "agent_attacked") {
    add(event.agent_id, "attacker");
    add(data.target_id, data.kill ? "killed" : "target");
    return uniqueAgentRoles(roles);
  }
  if (event.type === "agent_reproduced") {
    for (const parentId of data.parent_ids ?? [event.agent_id]) add(parentId, "parent");
    add(data.child_id, "child");
    return uniqueAgentRoles(roles);
  }
  if (event.type === "agent_died") {
    add(event.agent_id, "died");
    add(data.killer_id, "killer");
    return uniqueAgentRoles(roles);
  }
  if (event.type === "carcass_deposited") {
    add(data.source_agent_id ?? event.agent_id, "source");
    add(data.killer_id, "killer");
    return uniqueAgentRoles(roles);
  }
  if (event.type === "agent_ate") {
    add(event.agent_id, "consumer");
    addCarcassBreakdownRoles();
    return uniqueAgentRoles(roles);
  }
  if (event.type === "agent_moved") add(event.agent_id, "mover");
  else if (event.type === "agent_damaged") add(event.agent_id, "damaged");
  else if (event.type === "agent_healed") add(event.agent_id, "healed");
  else if (event.type === "agent_drank") add(event.agent_id, "drinker");
  else add(event.agent_id, "actor");
  add(data.target_id, "target");
  add(data.source_agent_id, "source");
  add(data.killer_id, "killer");
  return uniqueAgentRoles(roles);
}

function uniqueAgentRoles(roles) {
  const seen = new Set();
  return roles.filter(({ agentId, role }) => {
    const key = `${agentId}:${role}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function sortAgentEpisodeRoles(left, right) {
  return episodeRolePriority(left) - episodeRolePriority(right) || String(left).localeCompare(String(right));
}

export function episodeRolePriority(roleLabel) {
  const role = String(roleLabel).split(" / ")[0];
  const priority = {
    killed: 0,
    died: 1,
    attacker: 2,
    killer: 3,
    child: 4,
    parent: 5,
    consumer: 6,
    target: 7,
    source: 8,
    damaged: 9,
    healed: 10,
    drinker: 11,
    mover: 12,
    actor: 13,
  };
  return priority[role] ?? 99;
}

export function episodeLedger(events, options = {}) {
  const primary = primaryEpisodeEvents(events).filter((event) => eventLensCategory(event) != null);
  const support = events.filter(isEpisodeSupportEvent);
  const detailedEvents = primary.length > 0 ? primary : support.slice(0, 6);
  const supportSummary = supportEventSummary(support);
  return [
    ...detailedEvents.map((event) => episodeLedgerItem(event, options)),
    ...(supportSummary ? [supportSummary] : []),
  ];
}

function episodeLedgerItem(event, options) {
  const data = event.data ?? {};
  const formatters = ledgerFormatters(options);
  if (event.type === "agent_attacked") {
    return {
      category: "combat",
      title: "Attack",
      detail: `Agent ${event.agent_id} ${data.success ? "hit" : "missed"} agent ${data.target_id} for ${formatters.roundValue(data.damage ?? 0)}${data.kill ? " and killed it" : ""}`,
    };
  }
  if (event.type === "agent_reproduced") {
    return {
      category: "birth",
      title: "Birth",
      detail: `Agent ${data.child_id} born by ${formatters.reproductionModeLabel(data.reproduction_mode)} near ${data.child_x},${data.child_y}`,
    };
  }
  if (event.type === "agent_died") {
    return {
      category: "death",
      title: "Death",
      detail: `Agent ${event.agent_id} died from ${formatters.deathCauseLabel(data.cause)} at ${data.x},${data.y}`,
    };
  }
  if (event.type === "carcass_deposited") {
    return {
      category: "carcass",
      title: "Carcass",
      detail: `${formatters.roundValue(data.deposited_energy ?? 0)} energy deposited at ${data.x},${data.y}`,
    };
  }
  if (event.type === "agent_ate") {
    return {
      category: eventLensCategory(event) ?? "support",
      title: "Feeding",
      detail: `Agent ${event.agent_id} ate ${formatters.foodSourceLabel(data.food_source)} for ${formatters.roundValue(data.gained_energy ?? 0)} energy`,
    };
  }
  if (event.type === "agent_moved") {
    return {
      category: "support",
      title: "Movement",
      detail: `Agent ${event.agent_id} moved to ${data.x},${data.y}`,
    };
  }
  return {
    category: eventLensCategory(event) ?? "support",
    title: titleCase(event.type),
    detail: `Agent ${event.agent_id ?? formatters.emptyLabel}`,
  };
}

function ledgerFormatters(options) {
  return {
    roundValue: options.roundValue ?? defaultRoundValue,
    reproductionModeLabel: options.reproductionModeLabel ?? defaultLabel,
    deathCauseLabel: options.deathCauseLabel ?? defaultLabel,
    foodSourceLabel: options.foodSourceLabel ?? defaultLabel,
    emptyLabel: options.emptyLabel ?? "N/A",
  };
}

export function supportEventSummary(events) {
  if (events.length === 0) return null;
  const counts = {
    moved: 0,
    damaged: 0,
    healed: 0,
    plantMeals: 0,
    drinks: 0,
    other: 0,
  };
  for (const event of events) {
    if (event.type === "agent_moved") counts.moved += 1;
    else if (event.type === "agent_damaged") counts.damaged += 1;
    else if (event.type === "agent_healed") counts.healed += 1;
    else if (event.type === "agent_ate") counts.plantMeals += 1;
    else if (event.type === "agent_drank") counts.drinks += 1;
    else counts.other += 1;
  }
  const parts = [
    supportCountLabel(counts.moved, "moved"),
    supportCountLabel(counts.damaged, "damaged"),
    supportCountLabel(counts.healed, "healed"),
    supportCountLabel(counts.plantMeals, "plant meal", "plant meals"),
    supportCountLabel(counts.drinks, "drink", "drinks"),
    supportCountLabel(counts.other, "other event", "other events"),
  ].filter(Boolean);
  return {
    category: "support",
    title: "Support Events",
    detail: parts.join(" · "),
  };
}

function supportCountLabel(count, singular, plural = null) {
  if (!count) return null;
  return `${count} ${count === 1 ? singular : plural ?? singular}`;
}

export function buildStoryboardExportModel(options) {
  const runId = String(options.runId ?? "unknown-run");
  const bookmarks = options.eventBookmarks ?? [];
  const episodes = options.episodes ?? [];
  const summarizeEventTick = options.eventTickSummary ?? eventTickSummary;
  const sourceEntries =
    bookmarks.length > 0
      ? bookmarks.map((bookmark) => ({
          tick: bookmark.tick,
          note: bookmark.summary,
          bookmarked_at: bookmark.createdAt,
        }))
      : episodes.slice(0, 12).map((episode) => ({
          tick: episode.tick,
          note: summarizeEventTick(episode),
          bookmarked_at: null,
        }));
  const entries = sourceEntries.map((entry) => options.entryForTick(entry));
  return {
    schema_version: "viewer_storyboard_v1",
    run_id: runId,
    replay_source: String(options.replaySource ?? ""),
    source: bookmarks.length > 0 ? "bookmarks" : "major_episodes",
    exported_at: options.exportedAt ?? new Date().toISOString(),
    entry_count: entries.length,
    entries,
  };
}

export function eventBookmarkSummary(tick, significantEventTicks, events) {
  const significant = significantEventTicks.find((entry) => Number(entry.tick) === Number(tick));
  if (significant) return eventTickSummary(significant);
  const counts = events.reduce((accumulator, event) => {
    const category = eventLensCategory(event) ?? "other";
    accumulator[category] = (accumulator[category] ?? 0) + 1;
    return accumulator;
  }, {});
  const parts = [
    counts.birth ? `${counts.birth} birth${counts.birth === 1 ? "" : "s"}` : null,
    counts.death ? `${counts.death} death${counts.death === 1 ? "" : "s"}` : null,
    counts.combat ? `${counts.combat} combat event${counts.combat === 1 ? "" : "s"}` : null,
    counts.carcass ? `${counts.carcass} carcass flow event${counts.carcass === 1 ? "" : "s"}` : null,
    counts.other ? `${counts.other} support event${counts.other === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(", ") : "No recorded event on this tick";
}

export function episodeLabel(episode) {
  if (episode.hasDeaths) return "Mortality pressure";
  if (episode.hasBirths) return "Population growth";
  if (episode.hasCarcass) return "Food-web transfer";
  if (episode.hasCombat) return "Predation pressure";
  return "World event";
}

export function eventTickSummary(entry) {
  if (!entry) return "no event";
  const parts = [
    entry.births ? `${entry.births} birth${entry.births === 1 ? "" : "s"}` : null,
    entry.deaths ? `${entry.deaths} death${entry.deaths === 1 ? "" : "s"}` : null,
    entry.attacks ? `${entry.attacks} attack${entry.attacks === 1 ? "" : "s"}` : null,
    entry.carcasses ? `${entry.carcasses} carcass deposit${entry.carcasses === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(", ") : `${entry.total} event${entry.total === 1 ? "" : "s"}`;
}

export function tilePressureTags(context, agents, eventCounts) {
  const tags = [];
  if ((context.hazardLevel ?? 0) > 0.2 || context.hazardType !== "none") {
    tags.push({ label: "Hazard pressure", kind: "danger" });
  }
  if ((context.carcassEnergy ?? 0) > 0 || eventCounts.carcass > 0 || eventCounts.meatMeals > 0) {
    tags.push({ label: "Carrion flow", kind: "warning" });
  }
  if ((context.vegetation ?? 0) > 0.65) {
    tags.push({ label: "Vegetation rich", kind: "good" });
  }
  if (agents.length > 0) {
    tags.push({ label: `${agents.length} occupant${agents.length === 1 ? "" : "s"}`, kind: "info" });
  }
  if (eventCounts.combat > 0) {
    tags.push({ label: "Combat recent", kind: "danger" });
  }
  if (tags.length === 0) {
    tags.push({ label: "Quiet tile", kind: "info" });
  }
  return tags.slice(0, 5);
}

export function tileExplanationSentence(context, agents, eventCounts, options = {}) {
  const hazardLabel = options.hazardLabel ?? defaultLabel;
  const formatPercent = options.formatPercent ?? defaultPercent;
  const occupants = agents.length
    ? `${agents.length} living agent${agents.length === 1 ? "" : "s"} occupy it`
    : "no living agents occupy it";
  const hazard = context.hazardType !== "none"
    ? `${hazardLabel(context.hazardType).toLowerCase()} hazard is active`
    : "hazard is quiet";
  const food =
    (context.carcassEnergy ?? 0) > 0
      ? `carcass energy is available at ${formatPercent(context.carcassEnergy)}`
      : `vegetation is at ${formatPercent(context.vegetation ?? 0)}`;
  const events = [
    eventCounts.birth ? `${eventCounts.birth} birth${eventCounts.birth === 1 ? "" : "s"}` : null,
    eventCounts.death ? `${eventCounts.death} death${eventCounts.death === 1 ? "" : "s"}` : null,
    eventCounts.combat ? `${eventCounts.combat} attack${eventCounts.combat === 1 ? "" : "s"}` : null,
    eventCounts.carcass ? `${eventCounts.carcass} carcass-flow event${eventCounts.carcass === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return `${occupants}; ${hazard}; ${food}.${events.length ? ` Recent local events: ${events.join(", ")}.` : ""}`;
}

function defaultSignedValue(value) {
  const numeric = Number(value) || 0;
  return numeric > 0 ? `+${defaultRoundValue(numeric)}` : String(defaultRoundValue(numeric));
}

function defaultRoundValue(value) {
  return Math.round(Number(value ?? 0) * 10000) / 10000;
}

function defaultPercent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function defaultLabel(value) {
  return titleCase(value ?? "unknown");
}

function titleCase(value) {
  return String(value ?? "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
