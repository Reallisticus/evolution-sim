import assert from "node:assert/strict";
import {
  VIEWER_STATE_STORAGE_KEY,
  VIEWER_STATE_STORAGE_VERSION,
  comparisonBaselineIndexForPreset,
  defaultOverlayOpacityForMode,
  episodeFilterModel,
  eventClusterDetailModel,
  frameComparison,
  loadViewerPreferences,
  normalizeViewerUrlState,
  overlayContractForMode,
  saveViewerPreferences,
  serializeViewerUrlState,
  serializeViewerPreferences,
} from "./dashboard_model.mjs";

const frames = [
  { tick: 0, alive_agents: 3, species_counts: [{ id: 1 }], births: 0, deaths: 0 },
  { tick: 1, alive_agents: 4, species_counts: [{ id: 1 }, { id: 2 }], births: 1, deaths: 0 },
  { tick: 2, alive_agents: 2, species_counts: [{ id: 1 }], births: 1, deaths: 2 },
];

assert.equal(overlayContractForMode("hazard").label, "Hazard");
assert.equal(overlayContractForMode("missing").label, "Terrain");
assert(defaultOverlayOpacityForMode("hazard") > defaultOverlayOpacityForMode("fertility"));

const comparison = frameComparison(frames[2], frames[0], () => 7);
assert.deepEqual(comparison, { alive: -1, species: 0, births: 1, deaths: 2, tiles: 7 });

assert.equal(
  comparisonBaselineIndexForPreset({
    preset: "previous",
    frames,
    currentFrameIndex: 2,
    significantEventTicks: [],
  }),
  1,
);
assert.equal(
  comparisonBaselineIndexForPreset({
    preset: "previous-event",
    frames,
    currentFrameIndex: 2,
    significantEventTicks: [{ tick: 0 }, { tick: 1 }],
  }),
  1,
);
assert.equal(
  comparisonBaselineIndexForPreset({
    preset: "previous-event",
    frames,
    currentFrameIndex: 1,
    significantEventTicks: [{ tick: 2 }],
  }),
  1,
);
assert.equal(
  comparisonBaselineIndexForPreset({
    preset: "run-start",
    frames,
    currentFrameIndex: 2,
    significantEventTicks: [],
  }),
  0,
);

const cluster = eventClusterDetailModel(
  { startTick: 10, endTick: 15, entryCount: 3, attacks: 4, deaths: 2, births: 0, carcasses: 1 },
  [
    { tick: 9, total: 1 },
    { tick: 10, attacks: 2, total: 2 },
    { tick: 12, deaths: 1, carcasses: 1, total: 2 },
    { tick: 15, attacks: 2, deaths: 1, total: 3 },
  ],
);
assert.equal(cluster.rangeLabel, "Ticks 10-15");
assert.equal(cluster.ticks.length, 3);
assert(cluster.summary.includes("3 event ticks"));
assert(cluster.ticks[0].label.includes("attack"));

const storage = new Map();
const storageAdapter = {
  getItem: (key) => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value),
};
const prefs = serializeViewerPreferences({
  overlayMode: "hazard",
  overlayOpacity: 0.86,
  overlayOpacityByMode: { hazard: 0.86 },
  blendTerrain: false,
  decisionOverlayMode: "reward",
  eventLensMode: "causal",
  activeDetailView: "events",
  activeCameraPreset: "pressure",
  activeView: "story",
});
assert.equal(prefs.version, VIEWER_STATE_STORAGE_VERSION);
saveViewerPreferences(storageAdapter, prefs);
assert(storage.get(VIEWER_STATE_STORAGE_KEY).includes('"overlayMode":"hazard"'));
assert.deepEqual(loadViewerPreferences(storageAdapter), prefs);

const urlState = serializeViewerUrlState({
  payload: { viewer: { frames: [{ tick: 0 }, { tick: 5 }, { tick: 10 }] } },
  currentFrameIndex: 2,
  selectedAgentId: 42,
  selectedSpeciesId: 7,
  overlayMode: "hazard",
  eventLensMode: "delta",
  activeDetailView: "events",
  activeCameraPreset: "pressure",
  activeView: "story",
  presentationMode: true,
});
assert.deepEqual(urlState, {
  v: 1,
  tick: 10,
  agent: 42,
  species: 7,
  overlay: "hazard",
  lens: "delta",
  detail: "events",
  camera: "pressure",
  view: "story",
  present: 1,
});
assert.deepEqual(
  normalizeViewerUrlState(new URLSearchParams("v=1&tick=10&agent=42&species=7&overlay=hazard&lens=delta&detail=events&camera=pressure&view=story&present=1")),
  urlState,
);
assert.equal(normalizeViewerUrlState(new URLSearchParams("v=99&tick=10")), null);

const episodeFilter = episodeFilterModel(
  [
    { tick: 1, kind: "combat", attacks: 2, deaths: 0, births: 0, carcasses: 0 },
    { tick: 3, kind: "birth", attacks: 0, deaths: 0, births: 1, carcasses: 0 },
    { tick: 5, kind: "death", attacks: 1, deaths: 1, births: 0, carcasses: 1 },
  ],
  {
    currentTick: 4,
    query: "death",
    kind: "death",
    eventAgentIdsByTick: new Map([[5, new Set([8, 9])]]),
    agentId: 8,
    limit: 5,
  },
);
assert.equal(episodeFilter.totalMatches, 1);
assert.equal(episodeFilter.visibleEpisodes[0].tick, 5);
assert.equal(episodeFilter.visibleEpisodes[0].relative, "future");
assert(episodeFilter.summary.includes("1 of 3"));

console.log("dashboard_model_test_ok");
