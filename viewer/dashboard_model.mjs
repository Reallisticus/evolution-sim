export const VIEWER_STATE_STORAGE_VERSION = 1;
export const VIEWER_STATE_STORAGE_KEY = "evolution-sim:viewer-ui-state:v1";
export const VIEWER_URL_STATE_VERSION = 1;

export const OVERLAY_CONTRACTS = {
  terrain: {
    label: "Terrain",
    fields: ["viewer.map.terrain_codes"],
    summary: "Base terrain classes from the replay map contract.",
    keys: ["Plain", "Forest", "Wetland", "Rocky", "Water"],
    defaultOpacity: 0.72,
  },
  fertility: {
    label: "Fertility",
    fields: ["field_state.fertility_codes"],
    summary: "Plant productivity scalar used by feeding and recovery surfaces.",
    keys: ["Low", "Medium", "High"],
    defaultOpacity: 0.66,
  },
  moisture: {
    label: "Moisture",
    fields: ["field_state.moisture_codes"],
    summary: "Moisture scalar that shapes vegetation and hydrology pressure.",
    keys: ["Dry", "Balanced", "Wet"],
    defaultOpacity: 0.66,
  },
  heat: {
    label: "Heat",
    fields: ["field_state.heat_codes"],
    summary: "Heat scalar used by disturbance and tile stress calculations.",
    keys: ["Cool", "Moderate", "Hot"],
    defaultOpacity: 0.68,
  },
  hydrology: {
    label: "Hydrology",
    fields: ["hydrology_primary_codes"],
    summary: "Direct water access category for each tile.",
    keys: ["No access", "Adjacent", "Wetland", "Flooded"],
    defaultOpacity: 0.84,
  },
  shoreline: {
    label: "Shoreline",
    fields: ["hydrology_support_codes"],
    summary: "Water-adjacent support bits for movement and habitat context.",
    keys: ["None", "Shoreline", "Wetland", "Flooded"],
    defaultOpacity: 0.82,
  },
  refuge: {
    label: "Refuge",
    fields: ["soft_refuge_reason_codes", "refuge_score_codes"],
    summary: "Soft refuge reason plus local refuge score.",
    keys: ["No refuge", "Canopy", "High score"],
    defaultOpacity: 0.78,
  },
  hazard: {
    label: "Hazard",
    fields: ["hazard_type_codes", "hazard_level_codes"],
    summary: "Hazard class and intensity used by damage and pathing pressure.",
    keys: ["No hazard", "Exposure", "Instability"],
    defaultOpacity: 0.86,
  },
  carcass: {
    label: "Carcass",
    fields: ["carcass_energy_codes", "carcass_freshness_codes"],
    summary: "Meat resource energy and freshness on each tile.",
    keys: ["Absent", "Fresh", "Spent"],
    defaultOpacity: 0.9,
  },
  reproductive_signal: {
    label: "Reproductive Signal",
    fields: ["signal_fields.reproductive_signal"],
    summary: "Opt-in reproductive signal field emitted by ready agents.",
    keys: ["Quiet", "Trace", "Strong"],
    defaultOpacity: 0.88,
  },
  trophic: {
    label: "Trophic",
    fields: ["trophic_role_codes"],
    summary: "Dominant trophic role pressure visible on occupied tiles.",
    keys: ["None", "Herbivore", "Omnivore", "Carnivore"],
    defaultOpacity: 0.78,
  },
  habitat: {
    label: "Habitat",
    fields: ["habitat_state_codes"],
    summary: "Habitat state from the runtime ecology surface.",
    keys: ["Stable", "Bloom", "Flooded", "Parched"],
    defaultOpacity: 0.76,
  },
  ecology: {
    label: "Ecology",
    fields: ["ecology_state_codes"],
    summary: "Vegetation recovery state for each tile.",
    keys: ["Stable", "Lush", "Recovering", "Depleted"],
    defaultOpacity: 0.8,
  },
};

export const OVERLAY_OPACITY_PRESETS = {
  soft: 0.46,
  strong: 0.92,
};

export function overlayContractForMode(mode) {
  return OVERLAY_CONTRACTS[mode] ?? OVERLAY_CONTRACTS.terrain;
}

export function defaultOverlayOpacityForMode(mode) {
  return overlayContractForMode(mode).defaultOpacity ?? OVERLAY_CONTRACTS.terrain.defaultOpacity;
}

export function overlayOpacityPresetValue(mode, preset) {
  if (preset === "default") return defaultOverlayOpacityForMode(mode);
  return OVERLAY_OPACITY_PRESETS[preset] ?? defaultOverlayOpacityForMode(mode);
}

export function frameComparison(frame, baseline, countChangedTileDeltas) {
  return {
    alive: Number(frame?.alive_agents ?? 0) - Number(baseline?.alive_agents ?? frame?.alive_agents ?? 0),
    species:
      Number(frame?.species_counts?.length ?? 0) -
      Number(baseline?.species_counts?.length ?? frame?.species_counts?.length ?? 0),
    births: Number(frame?.births ?? 0) - Number(baseline?.births ?? frame?.births ?? 0),
    deaths: Number(frame?.deaths ?? 0) - Number(baseline?.deaths ?? frame?.deaths ?? 0),
    tiles: typeof countChangedTileDeltas === "function" ? countChangedTileDeltas(frame, baseline ?? frame) : 0,
  };
}

export function comparisonBaselineIndexForPreset({
  preset,
  frames,
  currentFrameIndex,
  significantEventTicks,
}) {
  const currentIndex = clampIndex(currentFrameIndex, frames.length);
  const currentTick = Number(frames[currentIndex]?.tick ?? 0);
  if (preset === "previous") {
    return Math.max(0, currentIndex - 1);
  }
  if (preset === "run-start") {
    return 0;
  }
  if (preset === "previous-event") {
    const previous = [...(significantEventTicks ?? [])]
      .reverse()
      .find((entry) => Number(entry.tick) < currentTick);
    if (!previous) return currentIndex;
    const tick = Number(previous.tick);
    const index = frames.findIndex((frame) => Number(frame.tick) === tick);
    return index >= 0 ? index : currentIndex;
  }
  return currentIndex;
}

export function eventClusterDetailModel(clusterEntry, significantEventTicks, options = {}) {
  const limit = Number(options.limit ?? 8);
  const startTick = Number(clusterEntry?.startTick ?? clusterEntry?.tick ?? 0);
  const endTick = Number(clusterEntry?.endTick ?? startTick);
  const ticks = (significantEventTicks ?? [])
    .filter((entry) => Number(entry.tick) >= startTick && Number(entry.tick) <= endTick)
    .map((entry) => ({
      tick: Number(entry.tick),
      total: Number(entry.total ?? 0),
      label: eventTickLabel(entry),
    }))
    .slice(0, Number.isFinite(limit) ? Math.max(1, limit) : 8);
  const count = Number(clusterEntry?.entryCount ?? ticks.length);
  return {
    startTick,
    endTick,
    rangeLabel: startTick === endTick ? `Tick ${startTick}` : `Ticks ${startTick}-${endTick}`,
    summary: `${count} event tick${count === 1 ? "" : "s"} grouped by density: ${eventTickLabel(clusterEntry)}.`,
    tickCount: ticks.length,
    ticks,
  };
}

export function episodeCauseEffect(episode, counts, deltas, options = {}) {
  const formatValue = options.formatValue ?? String;
  const signedValue = options.signedValue ?? defaultSignedValue;
  const episodeLabel = options.episodeLabel ?? ((entry) => entry?.kind ?? "Episode");
  if ((counts.combat ?? 0) > 0 && ((counts.death ?? 0) > 0 || (counts.carcass ?? 0) > 0)) {
    return {
      cause: `${formatValue(counts.combat)} attack${counts.combat === 1 ? "" : "s"} concentrated pressure on local prey.`,
      effect: `${formatValue(counts.death)} death${counts.death === 1 ? "" : "s"} and ${formatValue(
        counts.carcass,
      )} carcass-flow event${counts.carcass === 1 ? "" : "s"} changed food availability.`,
    };
  }
  if ((counts.birth ?? 0) > 0) {
    return {
      cause: "Reproduction readiness and placement opened a birth opportunity.",
      effect: `${formatValue(counts.birth)} birth${counts.birth === 1 ? "" : "s"} shifted live population by ${signedValue(
        deltas.aliveDelta,
      )}.`,
    };
  }
  if ((counts.death ?? 0) > 0 || (counts.carcass ?? 0) > 0) {
    return {
      cause: "Mortality and carcass deposition changed local resource pressure.",
      effect: `Carcass tile delta ${signedValue(deltas.carcassTileDelta)} with ${formatValue(
        counts.carcass,
      )} carcass-flow event${counts.carcass === 1 ? "" : "s"}.`,
    };
  }
  if ((counts.combat ?? 0) > 0) {
    return {
      cause: `${formatValue(counts.combat)} attack${counts.combat === 1 ? "" : "s"} produced immediate tactical pressure.`,
      effect: "No same-tick death was recorded, so the map shows pressure before mortality.",
    };
  }
  return {
    cause: episode ? `${episodeLabel(episode)} support movement changed local positions.` : "Support movement changed local positions.",
    effect: `${formatValue(deltas.tileChanges)} tile layer change${deltas.tileChanges === 1 ? "" : "s"} are visible from the prior tick.`,
  };
}

export function serializeViewerPreferences(state) {
  const overlayMode = validOverlayMode(state.overlayMode) ? state.overlayMode : "terrain";
  const opacityByMode = normalizeOpacityByMode(state.overlayOpacityByMode);
  opacityByMode[overlayMode] = normalizeOpacity(state.overlayOpacity, defaultOverlayOpacityForMode(overlayMode));
  return {
    version: VIEWER_STATE_STORAGE_VERSION,
    overlayMode,
    overlayOpacity: opacityByMode[overlayMode],
    overlayOpacityByMode: opacityByMode,
    blendTerrain: Boolean(state.blendTerrain),
    decisionOverlayMode: validChoice(state.decisionOverlayMode, ["selected", "recent", "reward", "mind", "off"], "selected"),
    eventLensMode: validChoice(state.eventLensMode, ["current", "causal", "births", "deaths", "combat", "carcass", "delta"], "current"),
    activeDetailView: validChoice(state.activeDetailView, ["overview", "agent", "species", "events"], "overview"),
    activeCameraPreset: validChoice(state.activeCameraPreset, ["world", "selected", "episode", "pressure", "custom"], "world"),
    activeView: validChoice(state.activeView, ["story", "advanced"], "story"),
    presentationMode: Boolean(state.presentationMode),
  };
}

export function normalizeViewerPreferences(raw) {
  if (!raw || typeof raw !== "object") return null;
  if (Number(raw.version) !== VIEWER_STATE_STORAGE_VERSION) return null;
  return serializeViewerPreferences({
    overlayMode: raw.overlayMode,
    overlayOpacity: raw.overlayOpacity,
    overlayOpacityByMode: raw.overlayOpacityByMode,
    blendTerrain: raw.blendTerrain ?? true,
    decisionOverlayMode: raw.decisionOverlayMode,
    eventLensMode: raw.eventLensMode,
    activeDetailView: raw.activeDetailView,
    activeCameraPreset: raw.activeCameraPreset,
    activeView: raw.activeView,
    presentationMode: raw.presentationMode,
  });
}

export function loadViewerPreferences(storage, key = VIEWER_STATE_STORAGE_KEY) {
  try {
    const raw = storage?.getItem(key);
    return raw ? normalizeViewerPreferences(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function saveViewerPreferences(storage, preferences, key = VIEWER_STATE_STORAGE_KEY) {
  try {
    const normalized = normalizeViewerPreferences(preferences);
    if (!normalized) return false;
    storage?.setItem(key, JSON.stringify(normalized));
    return true;
  } catch {
    return false;
  }
}

export function serializeViewerUrlState(state) {
  const frames = state.payload?.viewer?.frames ?? [];
  const frame = frames[state.currentFrameIndex] ?? frames[0] ?? null;
  const result = {
    v: VIEWER_URL_STATE_VERSION,
    tick: normalizeInteger(frame?.tick, 0),
    overlay: validOverlayMode(state.overlayMode) ? state.overlayMode : "terrain",
    lens: validChoice(state.eventLensMode, ["current", "causal", "births", "deaths", "combat", "carcass", "delta"], "current"),
    detail: validChoice(state.activeDetailView, ["overview", "agent", "species", "events"], "overview"),
    camera: validChoice(state.activeCameraPreset, ["world", "selected", "episode", "pressure", "custom"], "world"),
    view: validChoice(state.activeView, ["story", "advanced"], "story"),
    present: state.presentationMode ? 1 : 0,
  };
  const agentId = optionalInteger(state.selectedAgentId);
  const speciesId = optionalInteger(state.selectedSpeciesId);
  if (agentId != null) result.agent = agentId;
  if (speciesId != null) result.species = speciesId;
  return result;
}

export function normalizeViewerUrlState(raw) {
  if (!raw) return null;
  const get = (key) => (typeof raw.get === "function" ? raw.get(key) : raw[key]);
  if (Number(get("v")) !== VIEWER_URL_STATE_VERSION) return null;
  const result = {
    v: VIEWER_URL_STATE_VERSION,
    tick: normalizeInteger(get("tick"), 0),
    overlay: validOverlayMode(get("overlay")) ? String(get("overlay")) : "terrain",
    lens: validChoice(String(get("lens") ?? ""), ["current", "causal", "births", "deaths", "combat", "carcass", "delta"], "current"),
    detail: validChoice(String(get("detail") ?? ""), ["overview", "agent", "species", "events"], "overview"),
    camera: validChoice(String(get("camera") ?? ""), ["world", "selected", "episode", "pressure", "custom"], "world"),
    view: validChoice(String(get("view") ?? ""), ["story", "advanced"], "story"),
    present: ["1", "true", "yes"].includes(String(get("present") ?? "0")) ? 1 : 0,
  };
  const agentId = optionalInteger(get("agent"));
  const speciesId = optionalInteger(get("species"));
  if (agentId != null) result.agent = agentId;
  if (speciesId != null) result.species = speciesId;
  return result;
}

export function episodeFilterModel(episodes, options = {}) {
  const allEpisodes = Array.isArray(episodes) ? episodes : [];
  const query = normalizeSearch(options.query);
  const kind = String(options.kind ?? "all");
  const currentTick = Number(options.currentTick ?? 0);
  const limit = Math.max(1, Math.trunc(Number(options.limit ?? 10)) || 10);
  const agentId = optionalInteger(options.agentId);
  const eventAgentIdsByTick = options.eventAgentIdsByTick;
  const matches = allEpisodes.filter((episode) => {
    if (kind !== "all" && String(episode.kind ?? "") !== kind) return false;
    if (agentId != null && !eventAgentIdsByTick?.get?.(Number(episode.tick))?.has?.(agentId)) return false;
    if (query && !episodeSearchText(episode).includes(query)) return false;
    return true;
  });
  let anchor = matches.findIndex((episode) => Number(episode.tick) >= currentTick);
  if (anchor < 0) anchor = Math.max(0, matches.length - 1);
  const start = Math.max(0, Math.min(Math.max(0, matches.length - limit), anchor - Math.floor(limit / 2)));
  const visibleEpisodes = matches.slice(start, start + limit).map((episode) => {
    const tick = Number(episode.tick);
    return {
      ...episode,
      relative: tick < currentTick ? "past" : tick > currentTick ? "future" : "current",
    };
  });
  return {
    totalEpisodes: allEpisodes.length,
    totalMatches: matches.length,
    visibleEpisodes,
    query,
    kind,
    agentId,
    summary: `${matches.length} of ${allEpisodes.length} episode${allEpisodes.length === 1 ? "" : "s"} shown`,
  };
}

function eventTickLabel(entry) {
  const parts = [];
  if ((entry?.births ?? 0) > 0) parts.push(`${entry.births} birth${entry.births === 1 ? "" : "s"}`);
  if ((entry?.deaths ?? 0) > 0) parts.push(`${entry.deaths} death${entry.deaths === 1 ? "" : "s"}`);
  if ((entry?.attacks ?? 0) > 0) parts.push(`${entry.attacks} attack${entry.attacks === 1 ? "" : "s"}`);
  if ((entry?.carcasses ?? 0) > 0) parts.push(`${entry.carcasses} carcass${entry.carcasses === 1 ? "" : "es"}`);
  return parts.length ? parts.join(", ") : `${Number(entry?.total ?? 0)} event${Number(entry?.total ?? 0) === 1 ? "" : "s"}`;
}

function defaultSignedValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric === 0) return "0";
  return numeric > 0 ? `+${numeric}` : String(numeric);
}

function clampIndex(index, length) {
  if (length <= 0) return 0;
  return Math.max(0, Math.min(Number(index) || 0, length - 1));
}

function normalizeOpacity(value, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(0.2, Math.min(numeric, 1));
}

function normalizeOpacityByMode(raw) {
  const result = {};
  if (!raw || typeof raw !== "object") return result;
  for (const mode of Object.keys(OVERLAY_CONTRACTS)) {
    if (raw[mode] != null) {
      result[mode] = normalizeOpacity(raw[mode], defaultOverlayOpacityForMode(mode));
    }
  }
  return result;
}

function validOverlayMode(value) {
  return Object.hasOwn(OVERLAY_CONTRACTS, value);
}

function validChoice(value, choices, fallback) {
  return choices.includes(value) ? value : fallback;
}

function optionalInteger(value) {
  if (value == null || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.trunc(numeric) : null;
}

function normalizeInteger(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.trunc(numeric) : fallback;
}

function normalizeSearch(value) {
  return String(value ?? "").trim().toLowerCase();
}

function episodeSearchText(episode) {
  return [
    episode.kind,
    episode.tick,
    episode.hasBirths ? "birth births reproduction population growth" : "",
    episode.hasDeaths ? "death deaths mortality collapse" : "",
    episode.hasCombat ? "attack attacks combat predation" : "",
    episode.hasCarcass ? "carcass carrion food web" : "",
    eventTickLabel(episode),
  ]
    .join(" ")
    .toLowerCase();
}
