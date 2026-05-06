import * as PIXI from "./vendor/pixi.min.mjs";
import {
  VIEWER_DISPLAY_LABELS,
  VIEWER_EMPTY_LABELS,
} from "./contracts.generated.mjs?assetVersion=display-v01";
import {
  buildEpisodePlaybackScope,
  buildStoryboardExportModel,
  episodeAgentRoleEntries,
  episodeCausalitySentence,
  episodeContextForTick as modelEpisodeContextForTick,
  episodeLabel,
  episodeLedger as modelEpisodeLedger,
  episodeLensModes,
  eventBookmarkSummary as modelEventBookmarkSummary,
  eventCategoryCounts,
  eventLensCategory,
  eventLensTicks,
  eventMatchesLens,
  eventTickSummary,
  eventTouchesPlaybackScope,
  tileExplanationSentence,
  tilePressureTags,
} from "./episode_model.mjs";
import {
  OVERLAY_CONTRACTS,
  VIEWER_STATE_STORAGE_VERSION,
  comparisonBaselineIndexForPreset,
  defaultOverlayOpacityForMode,
  episodeCauseEffect as modelEpisodeCauseEffect,
  episodeFilterModel,
  eventClusterDetailModel,
  frameComparison as modelFrameComparison,
  loadViewerPreferences,
  normalizeViewerUrlState,
  overlayContractForMode,
  overlayOpacityPresetValue,
  saveViewerPreferences,
  serializeViewerUrlState,
  serializeViewerPreferences,
} from "./dashboard_model.mjs";
import {
  EVENT_MARKER_STYLES,
  countChangedTileDeltas,
  renderAgentLayer,
  renderDecisionOverlayLayer,
  renderEventMapLayer,
  renderTerrainLayer,
  renderTrailLayer,
} from "./map_layers.mjs";
import { buildReplayModel } from "./replay_model.mjs";
import { validateReplayPayload } from "./replay_validator.mjs";
import {
  DEFAULT_SNAPSHOT_BUDGET_MS,
  DEFAULT_SNAPSHOT_CACHE_LIMIT,
  buildSnapshotBudgetReport,
  changedTileCoordinates,
  snapshotCacheKey,
  snapshotScaleForMap,
  trimSnapshotCache,
} from "./snapshot_model.mjs";
import {
  blendColor,
  clamp,
  clamp01,
  clampToOrderedRange,
  escapeHtml,
  formatClimateField,
  formatInteger,
  formatPercent,
  formatValue,
  hslToRgb,
  roundValue,
  titleCase,
  yesNoLabel,
} from "./view_helpers.mjs";

const state = {
  payload: null,
  currentFrameIndex: 0,
  selectedAgentId: null,
  selectedSpeciesId: null,
  overlayMode: "terrain",
  overlayOpacityByMode: {},
  decisionOverlayMode: "selected",
  eventLensMode: "current",
  overlayOpacity: 0.72,
  blendTerrain: true,
  hoveredTile: null,
  explainedTile: null,
  mapView: {
    zoom: 1,
    panX: 0,
    panY: 0,
  },
  activeCameraPreset: "world",
  comparisonBaselineFrameIndex: 0,
  mapPointer: null,
  activeView: "story",
  playing: false,
  timerId: null,
  episodePlayback: null,
  episodeTimerId: null,
  pixiApp: null,
  terrainLayer: null,
  trailLayer: null,
  agentLayer: null,
  eventLayer: null,
  agentEncoding: null,
  replayModel: null,
  loadGeneration: 0,
  replaySource: null,
  eventsByTick: new Map(),
  significantEventTicks: [],
  eventBookmarks: [],
  activeEventCluster: null,
  lastStoryboardExport: null,
  lastShareUrl: null,
  pendingUrlState: normalizeViewerUrlState(new URLSearchParams(window.location.search)),
  presentationMode: false,
  episodeFilters: {
    query: "",
    kind: "all",
    selectedAgentOnly: false,
  },
  snapshotCache: new Map(),
  snapshotStats: {
    renders: 0,
    hits: 0,
    lastBudget: null,
  },
  activeDetailView: "overview",
  visualStats: {
    glyphs: 0,
    trailSegments: 0,
    clusterHalos: 0,
    eventMarkers: 0,
    eventLensMarkers: 0,
    eventFlowLinks: 0,
    deltaMarkers: 0,
    decisionOverlays: 0,
    terrainBlends: 0,
    lineageBranches: 0,
    eventBands: 0,
  },
};

const UNKNOWN_TERRAIN_COLOR = 0xff00ff;
const TRAIL_FRAME_WINDOW = 14;
const MAX_TRAIL_AGENTS = 32;
const MAX_EVENT_MARKERS = 64;
const MAX_DECISION_OVERLAYS = 56;
const EVENT_LENS_FRAME_WINDOW = 24;
const EPISODE_PLAYBACK_WINDOW = 10;
const EPISODE_PLAYBACK_INTERVAL_MS = 320;
const MAX_EVENT_LENS_MARKERS = 96;
const EVENT_STRIP_CLUSTER_PX = 120;
const EVENT_STRIP_MIN_BUCKETS = 4;
const MAP_MIN_ZOOM = 0.72;
const MAP_MAX_ZOOM = 3.2;
const MAP_ZOOM_STEP = 1.22;
const MISSING_LABEL = VIEWER_EMPTY_LABELS.missing;
const UNKNOWN_LABEL = VIEWER_EMPTY_LABELS.unknown;
const SPECIES_NAME_PARTS = {
  plain: {
    adjectives: ["Amber", "Dawn", "Vale", "Aster", "Lumen", "Sunfield"],
    grazer: ["Grazer", "Runner", "Browser", "Meadowling"],
    forager: ["Forager", "Seeker", "Wanderer", "Gatherer"],
    scavenger: ["Scavenger", "Gleaner", "Carrioner", "Drifter"],
    predator: ["Hunter", "Stalker", "Courser", "Pouncer"],
  },
  forest: {
    adjectives: ["Moss", "Briar", "Canopy", "Fern", "Lichen", "Shade"],
    grazer: ["Browser", "Grazer", "Leafling", "Thicket"],
    forager: ["Forager", "Nester", "Rootseeker", "Mender"],
    scavenger: ["Gleaner", "Hollows", "Carrioner", "Drifter"],
    predator: ["Stalker", "Hunter", "Pouncer", "Nightstep"],
  },
  wetland: {
    adjectives: ["Reed", "Fen", "Mist", "Marsh", "Delta", "Willow"],
    grazer: ["Grazer", "Skimmer", "Mireling", "Browse"],
    forager: ["Forager", "Wader", "Seeker", "Nester"],
    scavenger: ["Gleaner", "Scavenger", "Carrioner", "Mudwalker"],
    predator: ["Hunter", "Stalker", "Striker", "Skulker"],
  },
  rocky: {
    adjectives: ["Slate", "Quartz", "Ridge", "Cinder", "Ember", "Basalt"],
    grazer: ["Grazer", "Climber", "Cragling", "Browser"],
    forager: ["Forager", "Strider", "Seeker", "Shelterer"],
    scavenger: ["Scavenger", "Gleaner", "Drifter", "Carrioner"],
    predator: ["Hunter", "Stalker", "Ravager", "Pouncer"],
  },
};
const elements = {
  replayUrl: document.getElementById("replay-url"),
  replayFile: document.getElementById("replay-file"),
  loadUrl: document.getElementById("load-url"),
  loadStatus: document.getElementById("load-status"),
  viewTabs: document.getElementById("view-tabs"),
  presentationToggle: document.getElementById("presentation-toggle"),
  shareViewLink: document.getElementById("share-view-link"),
  shareViewStatus: document.getElementById("share-view-status"),
  storyPanel: document.getElementById("story-panel"),
  advancedPanel: document.getElementById("advanced-panel"),
  advancedSidebarGrid: document.getElementById("advanced-sidebar-grid"),
  storyMapSlot: document.getElementById("story-map-slot"),
  canvasPanel: document.getElementById("canvas-panel"),
  analyticsGrid: document.getElementById("analytics-grid"),
  playToggle: document.getElementById("play-toggle"),
  playbackSpeed: document.getElementById("playback-speed"),
  overlayMode: document.getElementById("overlay-mode"),
  decisionOverlayMode: document.getElementById("decision-overlay-mode"),
  eventLensMode: document.getElementById("event-lens-mode"),
  eventLensSummary: document.getElementById("event-lens-summary"),
  overlayOpacity: document.getElementById("overlay-opacity"),
  overlayOpacityPresets: document.getElementById("overlay-opacity-presets"),
  blendTerrainToggle: document.getElementById("blend-terrain-toggle"),
  mapNavigation: document.getElementById("map-navigation"),
  mapZoomOut: document.getElementById("map-zoom-out"),
  mapZoomIn: document.getElementById("map-zoom-in"),
  mapResetView: document.getElementById("map-reset-view"),
  mapFitFocus: document.getElementById("map-fit-focus"),
  mapZoomLabel: document.getElementById("map-zoom-label"),
  timeline: document.getElementById("timeline"),
  frameLabel: document.getElementById("frame-label"),
  aliveLabel: document.getElementById("alive-label"),
  speciesLabel: document.getElementById("species-label"),
  seasonLabel: document.getElementById("season-label"),
  climateLabel: document.getElementById("climate-label"),
  birthsLabel: document.getElementById("births-label"),
  deathsLabel: document.getElementById("deaths-label"),
  overlayLabel: document.getElementById("overlay-label"),
  overlayMeaning: document.getElementById("overlay-meaning"),
  mapCameraPresets: document.getElementById("map-camera-presets"),
  terrainLegend: document.getElementById("terrain-legend"),
  summaryGrid: document.getElementById("summary-grid"),
  traitGrid: document.getElementById("trait-grid"),
  climateGrid: document.getElementById("climate-grid"),
  hydrologyPrimaryGrid: document.getElementById("hydrology-primary-grid"),
  hydrologySupportGrid: document.getElementById("hydrology-support-grid"),
  hazardGrid: document.getElementById("hazard-grid"),
  carcassGrid: document.getElementById("carcass-grid"),
  trophicGrid: document.getElementById("trophic-grid"),
  habitatGrid: document.getElementById("habitat-grid"),
  ecologyGrid: document.getElementById("ecology-grid"),
  speciesEmpty: document.getElementById("species-empty"),
  speciesList: document.getElementById("species-list"),
  inspectorEmpty: document.getElementById("inspector-empty"),
  inspectorGrid: document.getElementById("inspector-grid"),
  populationChart: document.getElementById("population-chart"),
  speciesChart: document.getElementById("species-chart"),
  turnoverChart: document.getElementById("turnover-chart"),
  traitChart: document.getElementById("trait-chart"),
  habitatChart: document.getElementById("habitat-chart"),
  hydrologyPrimaryChart: document.getElementById("hydrology-primary-chart"),
  hydrologySupportChart: document.getElementById("hydrology-support-chart"),
  hazardChart: document.getElementById("hazard-chart"),
  carcassChart: document.getElementById("carcass-chart"),
  combatChart: document.getElementById("combat-chart"),
  ecologyChart: document.getElementById("ecology-chart"),
  speciesEcologyEmpty: document.getElementById("species-ecology-empty"),
  speciesEcologyGrid: document.getElementById("species-ecology-grid"),
  speciesOccupancyBars: document.getElementById("species-occupancy-bars"),
  speciesTrendChart: document.getElementById("species-trend-chart"),
  collapseEmpty: document.getElementById("collapse-empty"),
  collapseList: document.getElementById("collapse-list"),
  sceneSummary: document.getElementById("scene-summary"),
  sceneStats: document.getElementById("scene-stats"),
  storyFeed: document.getElementById("story-feed"),
  focusCards: document.getElementById("focus-cards"),
  lineageTree: document.getElementById("lineage-tree"),
  agentDossier: document.getElementById("agent-dossier"),
  agentDossierBody: document.getElementById("agent-dossier-body"),
  clearAgentFocus: document.getElementById("clear-agent-focus"),
  detailTabs: document.getElementById("story-detail-tabs"),
  detailPanels: [...document.querySelectorAll("[data-detail-panel]")],
  episodeInspector: document.getElementById("episode-inspector"),
  episodeSearch: document.getElementById("episode-search"),
  episodeKindFilter: document.getElementById("episode-kind-filter"),
  episodeSelectedAgentFilter: document.getElementById("episode-selected-agent-filter"),
  episodeFilterSummary: document.getElementById("episode-filter-summary"),
  episodeList: document.getElementById("episode-list"),
  bookmarkCurrentEvent: document.getElementById("bookmark-current-event"),
  exportStoryboard: document.getElementById("export-storyboard"),
  storyboardExportStatus: document.getElementById("storyboard-export-status"),
  storyboardPreview: document.getElementById("storyboard-preview"),
  bookmarkList: document.getElementById("bookmark-list"),
  canvasHost: document.getElementById("canvas-host"),
  mapAnnotations: document.getElementById("map-annotations"),
  mapMinimap: document.getElementById("map-minimap"),
  tileHover: document.getElementById("tile-hover"),
  tileInspector: document.getElementById("tile-inspector"),
  tileExplainer: document.getElementById("tile-explainer"),
  eventSummary: document.getElementById("event-summary"),
  eventStrip: document.getElementById("event-strip"),
  eventClusterDrilldown: document.getElementById("event-cluster-drilldown"),
  comparisonPanel: document.getElementById("comparison-panel"),
  jumpPrevEvent: document.getElementById("jump-prev-event"),
  jumpNextEvent: document.getElementById("jump-next-event"),
};

const HYDROLOGY_SUPPORT_BITS = {
  adjacent_to_water: 1,
  wetland: 2,
  flooded: 4,
};

setupSimplifiedLayout();
applyStoredViewerPreferences();
await initPixi();
bindEvents();
await bootstrapDefaultReplay();

function setupSimplifiedLayout() {
  if (elements.storyMapSlot && elements.canvasPanel) {
    elements.storyMapSlot.appendChild(elements.canvasPanel);
  }
  if (elements.advancedPanel && elements.analyticsGrid) {
    elements.advancedPanel.appendChild(elements.analyticsGrid);
  }
  if (elements.advancedSidebarGrid) {
    const sidebarPanels = [...document.querySelectorAll(".sidebar > .panel")];
    sidebarPanels.slice(3).forEach((panel) => {
      elements.advancedSidebarGrid.appendChild(panel);
    });
  }
}

function applyStoredViewerPreferences() {
  const preferences = loadViewerPreferences(window.localStorage);
  if (!preferences) {
    state.overlayOpacityByMode = {
      terrain: state.overlayOpacity,
    };
    applyUrlPreferenceState(state.pendingUrlState);
    updatePreferenceControls();
    return;
  }
  state.overlayMode = preferences.overlayMode;
  state.overlayOpacityByMode = { ...preferences.overlayOpacityByMode };
  state.overlayOpacity = preferences.overlayOpacity;
  state.blendTerrain = preferences.blendTerrain;
  state.decisionOverlayMode = preferences.decisionOverlayMode;
  state.eventLensMode = preferences.eventLensMode;
  state.activeDetailView = preferences.activeDetailView;
  state.activeCameraPreset = preferences.activeCameraPreset;
  state.activeView = preferences.activeView;
  state.presentationMode = Boolean(preferences.presentationMode);
  applyUrlPreferenceState(state.pendingUrlState);
  updatePreferenceControls();
}

function applyUrlPreferenceState(urlState) {
  if (!urlState) return;
  state.overlayMode = urlState.overlay;
  state.overlayOpacity = state.overlayOpacityByMode[urlState.overlay] ?? defaultOverlayOpacityForMode(urlState.overlay);
  state.eventLensMode = urlState.lens;
  state.activeDetailView = urlState.detail;
  state.activeCameraPreset = urlState.camera;
  state.activeView = urlState.view;
  state.presentationMode = Boolean(urlState.present);
}

function updatePreferenceControls() {
  if (elements.overlayMode) elements.overlayMode.value = state.overlayMode;
  if (elements.overlayOpacity) elements.overlayOpacity.value = String(Math.round(state.overlayOpacity * 100));
  if (elements.blendTerrainToggle) elements.blendTerrainToggle.checked = Boolean(state.blendTerrain);
  if (elements.decisionOverlayMode) elements.decisionOverlayMode.value = state.decisionOverlayMode;
  if (elements.eventLensMode) elements.eventLensMode.value = state.eventLensMode;
  if (elements.episodeSearch) elements.episodeSearch.value = state.episodeFilters.query;
  if (elements.episodeKindFilter) elements.episodeKindFilter.value = state.episodeFilters.kind;
  if (elements.episodeSelectedAgentFilter) {
    elements.episodeSelectedAgentFilter.checked = Boolean(state.episodeFilters.selectedAgentOnly);
  }
  renderOverlayOpacityPresets();
  renderPresentationMode();
}

function persistViewerPreferences() {
  const preferences = serializeViewerPreferences(state);
  state.overlayOpacityByMode = { ...preferences.overlayOpacityByMode };
  saveViewerPreferences(window.localStorage, preferences);
  syncDebugState();
}

async function initPixi() {
  const app = new PIXI.Application();
  await app.init({
    antialias: true,
    background: "#08101c",
    resizeTo: elements.canvasHost,
  });

  app.stage.eventMode = "static";
  app.stage.hitArea = app.screen;
  app.stage.on("pointerdown", onCanvasPointerDown);
  app.stage.on("pointermove", onCanvasPointerMove);
  app.stage.on("pointerup", onCanvasPointerUp);
  app.stage.on("pointerupoutside", onCanvasPointerUp);

  elements.canvasHost.appendChild(app.canvas);

  state.pixiApp = app;
  state.terrainLayer = new PIXI.Container();
  state.trailLayer = new PIXI.Container();
  state.agentLayer = new PIXI.Container();
  state.eventLayer = new PIXI.Container();
  app.stage.addChild(state.terrainLayer);
  app.stage.addChild(state.trailLayer);
  app.stage.addChild(state.agentLayer);
  app.stage.addChild(state.eventLayer);

  window.addEventListener("resize", () => {
    if (state.payload) {
      drawTerrain();
      renderFrame(state.currentFrameIndex);
    }
  });
}

function bindEvents() {
  elements.loadUrl.addEventListener("click", async () => {
    const url = elements.replayUrl.value.trim();
    if (!url) return;
    await loadReplayFromUrl(url);
  });

  elements.replayFile.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      loadReplay(JSON.parse(text), file.name);
    } catch (error) {
      setStatus(`Failed to load replay: ${error.message}`);
    }
  });

  elements.timeline.addEventListener("input", () => {
    stopPlayback();
    renderFrame(Number(elements.timeline.value));
  });

  elements.playToggle.addEventListener("click", () => {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  elements.overlayMode.addEventListener("change", () => {
    setOverlayMode(elements.overlayMode.value);
  });

  elements.decisionOverlayMode?.addEventListener("change", () => {
    state.decisionOverlayMode = elements.decisionOverlayMode.value;
    persistViewerPreferences();
    renderFrame(state.currentFrameIndex);
  });

  elements.eventLensMode?.addEventListener("change", () => {
    state.eventLensMode = elements.eventLensMode.value;
    persistViewerPreferences();
    renderFrame(state.currentFrameIndex);
  });

  elements.overlayOpacity?.addEventListener("input", () => {
    setOverlayOpacity(Number(elements.overlayOpacity.value) / 100);
  });

  elements.overlayOpacityPresets?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-overlay-opacity-preset]");
    if (!button) return;
    setOverlayOpacity(overlayOpacityPresetValue(state.overlayMode, button.dataset.overlayOpacityPreset));
  });

  elements.blendTerrainToggle?.addEventListener("change", () => {
    state.blendTerrain = Boolean(elements.blendTerrainToggle.checked);
    persistViewerPreferences();
    drawTerrain();
    renderFrame(state.currentFrameIndex);
  });

  elements.mapZoomOut?.addEventListener("click", () => {
    setMapZoom(state.mapView.zoom / MAP_ZOOM_STEP);
  });

  elements.mapZoomIn?.addEventListener("click", () => {
    setMapZoom(state.mapView.zoom * MAP_ZOOM_STEP);
  });

  elements.mapResetView?.addEventListener("click", () => {
    resetMapView();
  });

  elements.mapFitFocus?.addEventListener("click", () => {
    fitMapToFocus();
  });

  elements.mapCameraPresets?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-map-camera-preset]");
    if (!button) return;
    applyMapCameraPreset(button.dataset.mapCameraPreset);
  });

  elements.comparisonPanel?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-compare-preset]");
    if (!button) return;
    applyComparisonPreset(button.dataset.comparePreset);
  });

  elements.viewTabs.addEventListener("click", (event) => {
    const presentation = event.target.closest("#presentation-toggle");
    if (presentation) {
      setPresentationMode(!state.presentationMode);
      return;
    }
    const share = event.target.closest("#share-view-link");
    if (share) {
      shareCurrentView();
      return;
    }
    const button = event.target.closest("[data-view]");
    if (!button) return;
    setActiveView(button.dataset.view);
  });

  elements.detailTabs?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-detail-view]");
    if (!button) return;
    setActiveDetailView(button.dataset.detailView);
  });
  elements.detailTabs?.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const buttons = [...elements.detailTabs.querySelectorAll("[data-detail-view]")];
    const currentIndex = buttons.findIndex((button) => button.dataset.detailView === state.activeDetailView);
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? buttons.length - 1
          : (currentIndex + (event.key === "ArrowLeft" ? -1 : 1) + buttons.length) % buttons.length;
    const nextButton = buttons[nextIndex];
    if (!nextButton) return;
    event.preventDefault();
    setActiveDetailView(nextButton.dataset.detailView);
    nextButton.focus();
  });

  elements.eventStrip?.addEventListener("click", (event) => {
    const marker = event.target.closest("[data-event-tick]");
    if (!marker) return;
    if (marker.dataset.eventRange) {
      state.activeEventCluster = eventClusterFromRange(marker.dataset.eventRange);
    } else {
      state.activeEventCluster = null;
    }
    jumpToTick(Number(marker.dataset.eventTick));
  });

  elements.episodeList?.addEventListener("click", (event) => {
    const item = event.target.closest("[data-episode-tick]");
    if (!item) return;
    jumpToTick(Number(item.dataset.episodeTick));
  });

  elements.episodeInspector?.addEventListener("click", (event) => {
    const play = event.target.closest("[data-episode-play]");
    if (play) {
      toggleEpisodePlayback(Number(play.dataset.episodePlay));
      return;
    }
    const lens = event.target.closest("[data-episode-lens]");
    if (lens) {
      setEventLensMode(lens.dataset.episodeLens);
      return;
    }
    const agent = event.target.closest("[data-episode-agent]");
    if (agent) {
      focusEpisodeAgent(Number(agent.dataset.episodeAgent));
      return;
    }
    const jump = event.target.closest("[data-episode-jump]");
    if (jump) {
      jumpToTick(Number(jump.dataset.episodeJump));
    }
  });

  elements.bookmarkCurrentEvent?.addEventListener("click", () => {
    toggleCurrentEventBookmark();
  });

  elements.exportStoryboard?.addEventListener("click", () => {
    exportStoryboard();
  });

  elements.bookmarkList?.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-bookmark-remove]");
    if (remove) {
      removeEventBookmark(Number(remove.dataset.bookmarkRemove));
      return;
    }
    const item = event.target.closest("[data-bookmark-tick]");
    if (!item) return;
    jumpToTick(Number(item.dataset.bookmarkTick));
  });

  elements.storyboardPreview?.addEventListener("click", (event) => {
    const item = event.target.closest("[data-storyboard-preview-tick]");
    if (!item) return;
    jumpToTick(Number(item.dataset.storyboardPreviewTick));
  });

  elements.eventClusterDrilldown?.addEventListener("click", (event) => {
    const item = event.target.closest("[data-cluster-tick]");
    if (!item) return;
    jumpToTick(Number(item.dataset.clusterTick));
  });

  elements.episodeSearch?.addEventListener("input", () => {
    state.episodeFilters.query = elements.episodeSearch.value;
    renderEventEpisodes(state.payload?.viewer?.frames?.[state.currentFrameIndex] ?? null);
    syncDebugState();
  });

  elements.episodeKindFilter?.addEventListener("change", () => {
    state.episodeFilters.kind = elements.episodeKindFilter.value;
    renderEventEpisodes(state.payload?.viewer?.frames?.[state.currentFrameIndex] ?? null);
    syncDebugState();
  });

  elements.episodeSelectedAgentFilter?.addEventListener("change", () => {
    state.episodeFilters.selectedAgentOnly = Boolean(elements.episodeSelectedAgentFilter.checked);
    renderEventEpisodes(state.payload?.viewer?.frames?.[state.currentFrameIndex] ?? null);
    syncDebugState();
  });

  elements.tileExplainer?.addEventListener("click", (event) => {
    const agent = event.target.closest("[data-episode-agent]");
    if (agent) {
      focusEpisodeAgent(Number(agent.dataset.episodeAgent));
      return;
    }
    const jump = event.target.closest("[data-episode-jump]");
    if (jump) {
      jumpToTick(Number(jump.dataset.episodeJump));
    }
  });

  elements.jumpPrevEvent?.addEventListener("click", () => {
    jumpToAdjacentEvent(-1);
  });

  elements.jumpNextEvent?.addEventListener("click", () => {
    jumpToAdjacentEvent(1);
  });

  elements.clearAgentFocus?.addEventListener("click", () => {
    state.selectedAgentId = null;
    state.selectedSpeciesId = null;
    setActiveDetailView("overview");
    renderFrame(state.currentFrameIndex);
  });

  elements.agentDossier?.addEventListener("click", (event) => {
    const item = event.target.closest("[data-agent-event-tick], [data-decision-tick]");
    if (!item) return;
    jumpToTick(Number(item.dataset.agentEventTick ?? item.dataset.decisionTick));
  });

  elements.canvasHost?.addEventListener("mousemove", (event) => {
    updateHoveredTileFromPointer(event);
  });

  elements.canvasHost?.addEventListener(
    "wheel",
    (event) => {
      if (!state.payload) return;
      event.preventDefault();
      const nextZoom =
        event.deltaY > 0 ? state.mapView.zoom / MAP_ZOOM_STEP : state.mapView.zoom * MAP_ZOOM_STEP;
      setMapZoom(nextZoom);
    },
    { passive: false },
  );

  elements.canvasHost?.addEventListener("keydown", onCanvasKeyDown);

  elements.canvasHost?.addEventListener("mouseleave", () => {
    state.hoveredTile = null;
    renderTileHover();
    renderTileInspector();
    syncDebugState();
  });

  elements.mapAnnotations?.addEventListener("click", (event) => {
    const marker = event.target.closest("[data-map-species-id]");
    if (!marker) return;
    const speciesId = Number(marker.dataset.mapSpeciesId);
    const match = decodedAgentsForFrame(state.currentFrameIndex).find(
      (agent) => agent.speciesId === speciesId,
    );
    state.selectedSpeciesId = speciesId;
    state.selectedAgentId = match?.agentId ?? null;
    setActiveDetailView(match ? "agent" : "species");
    renderFrame(state.currentFrameIndex);
  });

  elements.speciesList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-species-id]");
    if (!button || !state.payload) return;

    const speciesId = Number(button.dataset.speciesId);
    const match = decodedAgentsForFrame(state.currentFrameIndex).find(
      (agent) => agent.speciesId === speciesId,
    );

    state.selectedSpeciesId = speciesId;
    state.selectedAgentId = match?.agentId ?? null;
    setActiveDetailView(match ? "agent" : "species");
    renderFrame(state.currentFrameIndex);
  });

  elements.lineageTree?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-lineage-species-id]");
    if (!button || !state.payload) return;
    state.selectedSpeciesId = Number(button.dataset.lineageSpeciesId);
    state.selectedAgentId = null;
    setActiveDetailView("species");
    renderFrame(state.currentFrameIndex);
  });
}

function setActiveView(viewName) {
  const view = viewName === "advanced" ? "advanced" : "story";
  state.activeView = view;

  const isStory = view === "story";
  elements.storyPanel.classList.toggle("hidden", !isStory);
  elements.storyPanel.classList.toggle("active", isStory);
  elements.advancedPanel.classList.toggle("hidden", isStory);
  elements.advancedPanel.classList.toggle("active", !isStory);

  [...elements.viewTabs.querySelectorAll("[data-view]")].forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });

  if (isStory && state.payload) {
    requestAnimationFrame(() => {
      drawTerrain();
      renderFrame(state.currentFrameIndex);
    });
  }
  persistViewerPreferences();
  updateShareUrlState();
}

function setActiveDetailView(viewName) {
  const validViews = new Set(["overview", "agent", "species", "events"]);
  const view = validViews.has(viewName) ? viewName : "overview";
  state.activeDetailView = view;

  elements.detailPanels.forEach((panel) => {
    const active = panel.dataset.detailPanel === view;
    panel.classList.toggle("hidden", !active);
  });
  elements.detailTabs?.querySelectorAll("[data-detail-view]").forEach((button) => {
    const active = button.dataset.detailView === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
    button.setAttribute("tabindex", active ? "0" : "-1");
  });
  persistViewerPreferences();
  updateShareUrlState();
  syncDebugState();
}

function setPresentationMode(enabled) {
  state.presentationMode = Boolean(enabled);
  renderPresentationMode();
  persistViewerPreferences();
  updateShareUrlState();
  if (state.payload) {
    requestAnimationFrame(() => {
      drawTerrain();
      renderFrame(state.currentFrameIndex);
    });
  }
}

function renderPresentationMode() {
  document.body.classList.toggle("presentation-mode", Boolean(state.presentationMode));
  if (elements.presentationToggle) {
    elements.presentationToggle.classList.toggle("active", Boolean(state.presentationMode));
    elements.presentationToggle.textContent = state.presentationMode ? "Exit Present" : "Present";
  }
}

async function shareCurrentView() {
  const url = updateShareUrlState({ returnUrl: true });
  state.lastShareUrl = url;
  try {
    await navigator.clipboard?.writeText(url);
    if (elements.shareViewStatus) elements.shareViewStatus.textContent = "Link copied";
  } catch {
    if (elements.shareViewStatus) elements.shareViewStatus.textContent = "URL updated";
  }
  syncDebugState();
}

function updateShareUrlState(options = {}) {
  if (!state.payload) return window.location.href;
  const params = new URLSearchParams(window.location.search);
  const urlState = serializeViewerUrlState(state);
  for (const [key, value] of Object.entries(urlState)) {
    params.set(key, String(value));
  }
  const nextUrl = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
  const absoluteUrl = `${window.location.origin}${nextUrl}`;
  if (!options.returnOnly) {
    window.history.replaceState(null, "", nextUrl);
  }
  return absoluteUrl;
}

async function bootstrapDefaultReplay() {
  const replayParam = new URLSearchParams(window.location.search).get("replay");
  if (replayParam) {
    elements.replayUrl.value = replayParam;
  }

  const initialUrl = elements.replayUrl.value.trim();
  if (initialUrl) {
    await loadReplayFromUrl(initialUrl);
  }
}

async function loadReplayFromUrl(url) {
  setStatus(`Loading ${url} ...`);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const payload = await response.json();
    loadReplay(payload, url);
  } catch (error) {
    setStatus(`Failed to load replay: ${error.message}`);
  }
}

function loadReplay(payload, sourceLabel) {
  stopPlayback();
  const preferredDetailView = state.activeDetailView;
  const preferredCameraPreset = state.activeCameraPreset;
  state.payload = validateReplayPayload(payload);
  state.loadGeneration += 1;
  state.replaySource = sourceLabel;
  state.selectedAgentId = null;
  state.selectedSpeciesId = null;
  state.hoveredTile = null;
  state.explainedTile = null;
  state.currentFrameIndex = 0;
  state.overlayMode = elements.overlayMode.value;
  state.decisionOverlayMode = elements.decisionOverlayMode?.value ?? "selected";
  state.eventLensMode = elements.eventLensMode?.value ?? "current";
  state.overlayOpacity = clamp(Number(elements.overlayOpacity?.value ?? 72) / 100, 0.2, 1);
  state.overlayOpacityByMode[state.overlayMode] = state.overlayOpacity;
  state.blendTerrain = Boolean(elements.blendTerrainToggle?.checked ?? true);
  state.mapView = { zoom: 1, panX: 0, panY: 0 };
  state.activeCameraPreset = ["world", "selected", "episode", "pressure"].includes(preferredCameraPreset)
    ? preferredCameraPreset
    : "world";
  state.comparisonBaselineFrameIndex = 0;
  state.activeEventCluster = null;
  state.mapPointer = null;
  state.agentEncoding = buildEncodingMap(state.payload.viewer.agent_encoding);
  state.replayModel = buildReplayModel(state.payload, decodeAgent);
  state.eventsByTick = state.replayModel.eventsByTick;
  state.significantEventTicks = state.replayModel.significantEventTicks;
  state.eventBookmarks = loadEventBookmarks();
  state.lastStoryboardExport = null;
  state.snapshotCache.clear();
  state.snapshotStats = { renders: 0, hits: 0, lastBudget: null };
  state.activeDetailView = ["overview", "agent", "species", "events"].includes(preferredDetailView)
    ? preferredDetailView
    : "overview";
  const initialFrameIndex = applyLoadedUrlState(state.pendingUrlState);
  state.visualStats = {
    glyphs: 0,
    trailSegments: 0,
    clusterHalos: 0,
    eventMarkers: 0,
    eventLensMarkers: 0,
    eventFlowLinks: 0,
    deltaMarkers: 0,
    decisionOverlays: 0,
    terrainBlends: 0,
    lineageBranches: 0,
    eventBands: 0,
  };

  const frames = state.payload.viewer.frames;
  elements.timeline.disabled = frames.length === 0;
  elements.playToggle.disabled = frames.length === 0;
  elements.timeline.min = "0";
  elements.timeline.max = String(Math.max(frames.length - 1, 0));
  elements.timeline.value = "0";
  if (elements.storyboardExportStatus) {
    elements.storyboardExportStatus.textContent = "";
  }

  populateSummary(state.payload.summary);
  populateTraits(state.payload.summary);
  populateTerrainLegend(state.payload.viewer.map);
  setActiveView(state.activeView);
  setActiveDetailView(state.activeDetailView);
  drawTerrain();
  renderFrame(initialFrameIndex);
  if (state.activeCameraPreset !== "world") {
    requestAnimationFrame(() => applyMapCameraPreset(state.activeCameraPreset));
  }
  setStatus(`Loaded ${sourceLabel}`);
}

function applyLoadedUrlState(urlState) {
  if (!urlState || !state.payload) return 0;
  state.overlayMode = urlState.overlay;
  state.overlayOpacity = state.overlayOpacityByMode[urlState.overlay] ?? defaultOverlayOpacityForMode(urlState.overlay);
  state.eventLensMode = urlState.lens;
  state.activeDetailView = urlState.detail;
  state.activeCameraPreset = urlState.camera;
  state.activeView = urlState.view;
  state.presentationMode = Boolean(urlState.present);
  state.selectedAgentId = urlState.agent ?? null;
  state.selectedSpeciesId = urlState.species ?? null;
  updatePreferenceControls();
  const index = frameIndexForTick(urlState.tick);
  return index ?? 0;
}

function populateSummary(summary) {
  [...elements.summaryGrid.querySelectorAll("[data-summary-field]")].forEach((node) => {
    const field = node.dataset.summaryField;
    node.textContent = formatValue(summary[field]);
  });
}

function populateTraits(summary) {
  [...elements.traitGrid.querySelectorAll("[data-trait-field]")].forEach((node) => {
    const field = node.dataset.traitField;
    node.textContent = formatValue(summary[field], true);
  });
}

function populateTerrainLegend(map) {
  if (!elements.terrainLegend) return;
  const totalTiles = map.width * map.height;
  const items = terrainEntries(map).map(({ code, name }) => {
    const count = map.terrain_counts?.[name];
    const share = count != null ? count / Math.max(totalTiles, 1) : 0;
    const shareLabel = count != null ? formatPercent(share) : UNKNOWN_LABEL;
    return `
      <span
        class="terrain-legend-item"
        data-terrain-key="${escapeHtml(name)}"
        data-terrain-share-label="${escapeHtml(shareLabel)}"
        style="--terrain-share:${share};--terrain-color:${terrainCssColor(Number(code))};"
      >
        <i class="swatch ${escapeHtml(name)}" style="background:${terrainCssColor(Number(code))};"></i>
        <span>${escapeHtml(titleCase(name))}</span>
        ${count != null ? `<small>${escapeHtml(formatValue(count))} · ${escapeHtml(shareLabel)}</small>` : ""}
      </span>
    `;
  });
  elements.terrainLegend.innerHTML = `
    <div class="map-key-group terrain-key">
      <strong>Terrain coverage</strong>
      <span class="terrain-grid-note">${escapeHtml(formatValue(map.width))} x ${escapeHtml(
        formatValue(map.height),
      )} grid · ${escapeHtml(formatInteger(totalTiles))} tiles</span>
      <div class="terrain-key-grid">${items.join("")}</div>
    </div>
  `;
}

function renderOverlayMeaning(frame) {
  if (!elements.overlayMeaning) return;
  const contract = overlayContractForMode(state.overlayMode);
  const opacity = `${Math.round(state.overlayOpacity * 100)}%`;
  const fieldText = contract.fields.join(" + ");
  const keyItems = contract.keys
    .slice(0, 4)
    .map((label) => `<span data-overlay-key="${escapeHtml(label)}">${escapeHtml(label)}</span>`)
    .join("");
  const frameText = frame ? `tick ${escapeHtml(formatValue(frame.tick))}` : "current tick";
  elements.overlayMeaning.dataset.overlayContract = state.overlayMode;
  elements.overlayMeaning.innerHTML = `
    <div>
      <span>Backend ${escapeHtml(contract.label)} · ${escapeHtml(frameText)}</span>
      <strong>${escapeHtml(contract.summary)}</strong>
      <small>${escapeHtml(fieldText)} · opacity ${escapeHtml(opacity)}</small>
    </div>
    <div class="overlay-meaning-keys">${keyItems}</div>
  `;
}

function renderOverlayOpacityPresets() {
  if (!elements.overlayOpacityPresets) return;
  const contract = overlayContractForMode(state.overlayMode);
  elements.overlayOpacityPresets.querySelector("span").textContent =
    `${contract.label} preset`;
  const activeDefault = Math.round(defaultOverlayOpacityForMode(state.overlayMode) * 100);
  const current = Math.round(state.overlayOpacity * 100);
  elements.overlayOpacityPresets.querySelectorAll("[data-overlay-opacity-preset]").forEach((button) => {
    const value = Math.round(overlayOpacityPresetValue(state.overlayMode, button.dataset.overlayOpacityPreset) * 100);
    button.classList.toggle("selected", value === current);
    button.title = `${contract.label} ${button.textContent.trim()} opacity ${value}%`;
    if (button.dataset.overlayOpacityPreset === "default") {
      button.textContent = `Default ${activeDefault}%`;
    }
  });
}

function setOverlayMode(mode) {
  const nextMode = Object.hasOwn(OVERLAY_CONTRACTS, mode) ? mode : "terrain";
  state.overlayOpacityByMode[state.overlayMode] = state.overlayOpacity;
  state.overlayMode = nextMode;
  state.overlayOpacity =
    state.overlayOpacityByMode[nextMode] ?? defaultOverlayOpacityForMode(nextMode);
  if (elements.overlayMode) elements.overlayMode.value = nextMode;
  if (elements.overlayOpacity) elements.overlayOpacity.value = String(Math.round(state.overlayOpacity * 100));
  renderOverlayOpacityPresets();
  persistViewerPreferences();
  drawTerrain();
  renderFrame(state.currentFrameIndex);
}

function setOverlayOpacity(value) {
  state.overlayOpacity = clamp(Number(value), 0.2, 1);
  state.overlayOpacityByMode[state.overlayMode] = state.overlayOpacity;
  if (elements.overlayOpacity) elements.overlayOpacity.value = String(Math.round(state.overlayOpacity * 100));
  renderOverlayOpacityPresets();
  persistViewerPreferences();
  drawTerrain();
  renderFrame(state.currentFrameIndex);
}

function renderComparisonPanel(frame) {
  if (!elements.comparisonPanel || !state.payload) return;
  const frames = state.payload.viewer.frames;
  const currentIndex = state.currentFrameIndex;
  const baselineIndex = clamp(
    Number.isFinite(state.comparisonBaselineFrameIndex)
      ? Number(state.comparisonBaselineFrameIndex)
      : Math.max(0, currentIndex - 1),
    0,
    Math.max(0, frames.length - 1),
  );
  state.comparisonBaselineFrameIndex = baselineIndex;
  const baseline = frames[baselineIndex] ?? frame;
  const comparison = frameComparison(frame, baseline);
  const baselineMap = mapSnapshotDataUrl(baselineIndex, { maxPixels: 180 });
  const currentMap = mapSnapshotDataUrl(currentIndex, {
    maxPixels: 180,
    baselineIndex,
    delta: true,
  });
  elements.comparisonPanel.innerHTML = `
    <div class="comparison-map-pair">
      ${renderComparisonMap("baseline", baseline, baselineMap)}
      ${renderComparisonMap("current", frame, currentMap)}
    </div>
    <div class="comparison-heading">
      <span>Compare</span>
      <strong>Tick ${escapeHtml(formatValue(frame.tick))} vs tick ${escapeHtml(formatValue(baseline.tick))}</strong>
    </div>
    <dl class="comparison-deltas">
      ${[
        ["alive", "Alive", comparison.alive],
        ["species", "Sp", comparison.species],
        ["births", "Born", comparison.births],
        ["deaths", "Dead", comparison.deaths],
        ["tiles", "Tiles", comparison.tiles],
      ]
        .map(
          ([key, label, value]) => `
            <div data-comparison-delta="${escapeHtml(key)}">
              <dt>${escapeHtml(label)}</dt>
              <dd>${escapeHtml(signedValue(value))}</dd>
            </div>
          `,
        )
        .join("")}
    </dl>
    <div class="comparison-actions">
      <button type="button" data-compare-preset="previous">Prev Tick</button>
      <button type="button" data-compare-preset="previous-event">Prev Event</button>
      <button type="button" data-compare-preset="run-start">Start</button>
    </div>
  `;
}

function renderComparisonMap(kind, frame, dataUrl) {
  const label = kind === "current" ? "Current + Delta" : titleCase(kind);
  return `
    <div data-comparison-map="${escapeHtml(kind)}">
      ${dataUrl ? `<img alt="${escapeHtml(kind)} tick ${escapeHtml(formatValue(frame.tick))}" src="${dataUrl}" />` : ""}
      <span>${escapeHtml(label)}</span>
      <strong>Tick ${escapeHtml(formatValue(frame.tick))}</strong>
    </div>
  `;
}

function frameComparison(frame, baseline) {
  return modelFrameComparison(frame, baseline, countChangedTileDeltas);
}

function applyComparisonPreset(preset) {
  if (!state.payload) return;
  const frames = state.payload.viewer.frames;
  state.comparisonBaselineFrameIndex = comparisonBaselineIndexForPreset({
    preset,
    frames,
    currentFrameIndex: state.currentFrameIndex,
    significantEventTicks: state.significantEventTicks,
  });
  renderFrame(state.currentFrameIndex);
}

function drawTerrain() {
  const app = state.pixiApp;
  const payload = state.payload;
  if (!app || !payload) return;

  const map = payload.viewer.map;
  const frame = payload.viewer.frames[state.currentFrameIndex] ?? null;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);
  const stats = renderTerrainLayer({
    layer: state.terrainLayer,
    map,
    frame,
    tileSize,
    offset,
    overlayMode: state.overlayMode,
    overlayOpacity: state.overlayOpacity,
    blendTerrain: state.blendTerrain,
    terrainColor,
    terrainBaseColor,
    overlayColorForTile: (targetFrame, x, y, code) => overlayColorForTile(payload, targetFrame, x, y, code),
    terrainNameFromCode: (code) => terrainNameFromCode(map, code),
    waterCode: terrainCodeByName("water"),
    blendColor,
  });
  state.visualStats.terrainBlends = stats.terrainBlends;
  elements.overlayLabel.textContent = `Overlay: ${titleCase(state.overlayMode)}`;
  renderOverlayMeaning(frame);
  updateMapNavigation();
  syncDebugState();
}

function renderFrame(frameIndex) {
  const payload = state.payload;
  const app = state.pixiApp;
  if (!payload || !app) return;

  const frames = payload.viewer.frames;
  if (frames.length === 0) return;

  state.currentFrameIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
  elements.timeline.value = String(state.currentFrameIndex);
  drawTerrain();

  const frame = frames[state.currentFrameIndex];
  const decodedAgents = decodedAgentsForFrame(state.currentFrameIndex);
  const map = payload.viewer.map;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);

  renderMapTrails(decodedAgents, tileSize, offset);
  renderMapAgents(decodedAgents, tileSize, offset);
  renderMapEvents(frame, decodedAgents, tileSize, offset);
  renderMapDecisionOverlay(frame, decodedAgents, tileSize, offset);
  renderMapAnnotations(frame, decodedAgents, tileSize, offset);
  renderMapMinimap(map, tileSize, offset);
  renderTileHover();
  renderTileInspector();
  renderTileExplainer();

  elements.frameLabel.textContent = `Tick ${frame.tick}`;
  elements.aliveLabel.textContent = `Alive ${frame.alive_agents}`;
  elements.speciesLabel.textContent = `Species ${frame.species_counts.length}`;
  elements.seasonLabel.textContent = `Season ${frame.season}`;
  elements.climateLabel.textContent = `Climate ${frame.field_state?.disturbance_type ?? "-"}`;
  elements.birthsLabel.textContent = `Births ${frame.births}`;
  elements.deathsLabel.textContent = `Deaths ${frame.deaths}`;
  renderStoryFrame(frame, decodedAgents);
  renderAgentDossier(frame, decodedAgents);
  renderLineageTree(frame);
  renderStoryTimeline(frame);
  renderComparisonPanel(frame);
  renderEpisodeInspector(frame);
  renderEventEpisodes(frame);
  renderEventBookmarks(frame);
  renderSpeciesList(frame, decodedAgents);
  renderClimateState(frame);
  renderHydrologyState(frame);
  renderHazardState(frame);
  renderCarcassState(frame);
  renderTrophicState(decodedAgents);
  renderHabitatState(frame);
  renderEcologyState(frame);
  updateInspector();
  renderAnalyticsPanels(frame);
  updateShareUrlState();
  syncDebugState();
}

function decodedAgentsForFrame(frameIndex) {
  return state.replayModel?.decodedFrames?.[frameIndex]?.agents ?? [];
}

function renderMapTrails(decodedAgents, tileSize, offset) {
  if (!state.payload) return;
  const startIndex = Math.max(0, state.currentFrameIndex - TRAIL_FRAME_WINDOW);
  const recentFrames = [];
  for (let index = startIndex; index <= state.currentFrameIndex; index += 1) {
    recentFrames.push(decodedAgentsForFrame(index));
  }
  const stats = renderTrailLayer({
    layer: state.trailLayer,
    decodedAgents,
    recentFrames,
    selectedPositions: state.selectedAgentId == null ? [] : selectedAgentPositions(state.selectedAgentId),
    selectedAgentId: state.selectedAgentId,
    selectedSpeciesId: state.selectedSpeciesId,
    dominantSpeciesId: dominantSpeciesIdForFrame(),
    tileSize,
    offset,
    colorForSpecies,
    maxTrailAgents: MAX_TRAIL_AGENTS,
  });
  state.visualStats.trailSegments = stats.trailSegments;
  state.visualStats.clusterHalos = stats.clusterHalos;
}

function selectedAgentPositions(agentId) {
  const positions = state.replayModel?.agentPositionsById?.get(agentId) ?? [];
  const frames = state.payload?.viewer?.frames ?? [];
  const currentTick = frames[state.currentFrameIndex]?.tick ?? 0;
  return positions.filter((position) => position.tick <= currentTick);
}

function dominantSpeciesIdForFrame() {
  const frame = state.payload?.viewer?.frames?.[state.currentFrameIndex];
  return [...(frame?.species_counts ?? [])].sort((left, right) => right[1] - left[1])[0]?.[0] ?? null;
}

function renderMapAgents(decodedAgents, tileSize, offset) {
  const stats = renderAgentLayer({
    layer: state.agentLayer,
    decodedAgents,
    tileSize,
    offset,
    selectedAgentId: state.selectedAgentId,
    selectedSpeciesId: state.selectedSpeciesId,
    colorForSpecies,
  });
  state.visualStats.glyphs = stats.glyphs;
}

function renderMapEvents(frame, decodedAgents, tileSize, offset) {
  if (!state.payload) return;
  const previousIndex = Math.max(0, state.currentFrameIndex - 1);
  const previousFrame = previousIndex === state.currentFrameIndex ? null : state.payload.viewer.frames[previousIndex];
  const stats = renderEventMapLayer({
    layer: state.eventLayer,
    frame,
    decodedAgents,
    previousFrame,
    previousAgents: previousFrame ? decodedAgentsForFrame(previousIndex) : [],
    tileSize,
    offset,
    map: state.payload.viewer.map,
    mode: state.eventLensMode,
    events: state.eventLensMode === "delta" ? [] : eventLensEvents(frame.tick),
    markerForEvent: eventMarkerPosition,
    colorForSpecies,
    eventLensFrameWindow: EVENT_LENS_FRAME_WINDOW,
  });
  state.visualStats.eventMarkers = stats.eventMarkers;
  state.visualStats.eventLensMarkers = stats.eventLensMarkers;
  state.visualStats.eventFlowLinks = stats.eventFlowLinks;
  state.visualStats.deltaMarkers = stats.deltaMarkers;
  updateEventLensSummary(stats.markerCount, frame.tick);
}

function eventLensEvents(currentTick) {
  const mode = state.eventLensMode;
  const playback = mode === "causal" ? state.episodePlayback : null;
  const ticks = eventLensTicks(currentTick, mode, playback);
  const scopeMarkerForEvent = (event) => eventMarkerPosition(event, new Map(), event.tick);
  const events = [];
  for (const tick of ticks) {
    for (const event of state.eventsByTick.get(tick) ?? []) {
      if (!eventMatchesLens(event, mode)) continue;
      if (playback && !eventTouchesPlaybackScope(event, playback.scope, scopeMarkerForEvent)) continue;
      events.push(event);
    }
  }
  return events
    .sort((left, right) => {
      if (left.tick !== right.tick) return left.tick - right.tick;
      const leftPriority = EVENT_MARKER_STYLES[left.type]?.priority ?? 99;
      const rightPriority = EVENT_MARKER_STYLES[right.type]?.priority ?? 99;
      if (leftPriority !== rightPriority) return leftPriority - rightPriority;
      return String(left.type).localeCompare(String(right.type));
    })
    .slice(-MAX_EVENT_LENS_MARKERS);
}

function episodePlaybackScope(episodeTick) {
  return buildEpisodePlaybackScope(
    state.eventsByTick.get(Number(episodeTick)) ?? [],
    (event) => eventMarkerPosition(event, new Map(), event.tick),
  );
}

function updateEventLensSummary(markerCount, currentTick) {
  if (!elements.eventLensSummary) return;
  const playback = state.eventLensMode === "causal" ? state.episodePlayback : null;
  const labels = {
    current: "current tick",
    causal: playback ? `episode ${playback.startTick}-${playback.endTick} causal window` : `${EVENT_LENS_FRAME_WINDOW}-tick causal trail`,
    births: "birth trail",
    deaths: "death trail",
    combat: "combat trail",
    carcass: "carcass flow",
    delta: "tick delta",
  };
  elements.eventLensSummary.textContent = `Lens: ${labels[state.eventLensMode] ?? "events"} · ${formatValue(
    markerCount,
  )} mark${markerCount === 1 ? "" : "s"} at tick ${formatValue(currentTick)}`;
}

function renderMapDecisionOverlay(frame, decodedAgents, tileSize, offset) {
  const stats = renderDecisionOverlayLayer({
    layer: state.eventLayer,
    decodedAgents,
    tick: frame.tick,
    tileSize,
    offset,
    mode: state.decisionOverlayMode,
    selectedAgentId: state.selectedAgentId,
    selectedSpeciesId: state.selectedSpeciesId,
    recordForAgent: (agentId, tick) => decisionRecordAtOrBefore(agentId, tick),
    isPointOnMap,
    maxDecisionOverlays: MAX_DECISION_OVERLAYS,
  });
  state.visualStats.decisionOverlays = stats.decisionOverlays;
}

function renderMapAnnotations(frame, decodedAgents, tileSize, offset) {
  if (!elements.mapAnnotations || !state.payload) return;
  const selectedAgent = decodedAgents.find((agent) => agent.agentId === state.selectedAgentId);
  const topSpeciesIds = new Set(
    [...(frame.species_counts ?? [])]
      .sort((left, right) => right[1] - left[1])
      .slice(0, 3)
      .map(([speciesId]) => speciesId),
  );
  if (state.selectedSpeciesId != null) {
    topSpeciesIds.add(state.selectedSpeciesId);
  }
  if (selectedAgent) {
    topSpeciesIds.delete(selectedAgent.speciesId);
  }
  const bySpecies = new Map();
  for (const agent of decodedAgents) {
    if (!topSpeciesIds.has(agent.speciesId)) continue;
    if (!bySpecies.has(agent.speciesId)) {
      bySpecies.set(agent.speciesId, { x: 0, y: 0, count: 0 });
    }
    const centroid = bySpecies.get(agent.speciesId);
    centroid.x += agent.x;
    centroid.y += agent.y;
    centroid.count += 1;
  }

  const placedBoxes = [];
  const selectedAgentAnnotation = selectedAgent
    ? (() => {
        const center = tileCenter(selectedAgent.x, selectedAgent.y, tileSize, offset);
        return placeMapAnnotation(
          {
            agentId: selectedAgent.agentId,
            speciesId: selectedAgent.speciesId,
            x: Math.round(center.x),
            y: Math.round(center.y),
            boxWidth: 148,
            boxHeight: 48,
          },
          placedBoxes,
          state.payload.viewer.map,
          tileSize,
          offset,
        );
      })()
    : null;
  const annotations = [...bySpecies.entries()]
    .filter(([, centroid]) => centroid.count > 0)
    .map(([speciesId, centroid]) => {
      const x = centroid.x / centroid.count;
      const y = centroid.y / centroid.count;
      const center = tileCenter(x, y, tileSize, offset);
      const selected = state.selectedSpeciesId === speciesId;
      return {
        speciesId,
        selected,
        count: centroid.count,
        x: Math.round(center.x),
        y: Math.round(center.y),
      };
    })
    .sort((left, right) => {
      if (left.selected !== right.selected) return left.selected ? -1 : 1;
      return right.count - left.count;
    })
    .map((annotation) =>
      placeMapAnnotation(annotation, placedBoxes, state.payload.viewer.map, tileSize, offset),
    )
    .filter(Boolean);

  elements.mapAnnotations.innerHTML = `
    ${
      selectedAgentAnnotation
        ? `
          <div
            class="map-annotation agent-callout${selectedAgentAnnotation.mode === "pin" ? " compact-pin" : ""}"
            data-callout-mode="${selectedAgentAnnotation.mode}"
            style="left:${selectedAgentAnnotation.x}px;top:${selectedAgentAnnotation.y}px;--annotation-width:${selectedAgentAnnotation.boxWidth}px;"
          >
            <span>${escapeHtml(selectedAgentAnnotation.mode === "pin" ? "A" : `Agent ${selectedAgentAnnotation.agentId}`)}</span>
            <small>${escapeHtml(speciesLabel(selectedAgentAnnotation.speciesId))}</small>
          </div>
        `
        : ""
    }
    ${annotations
      .map(
        (annotation) => `
          <button
            type="button"
            class="map-annotation species-callout${annotation.selected ? " selected" : ""}${annotation.mode === "pin" ? " compact-pin" : ""}"
            data-map-species-id="${annotation.speciesId}"
            data-callout-mode="${annotation.mode}"
            aria-label="Focus ${escapeHtml(speciesLabel(annotation.speciesId))} species"
            style="left:${annotation.x}px;top:${annotation.y}px;--species-color:#${colorForSpecies(annotation.speciesId)
              .toString(16)
              .padStart(6, "0")};--annotation-width:${annotation.boxWidth}px;"
          >
            <span>${escapeHtml(annotation.mode === "pin" ? String(annotation.count) : speciesLabel(annotation.speciesId))}</span>
            <small>${escapeHtml(speciesCode(annotation.speciesId))} · ${annotation.count}</small>
          </button>
        `,
      )
      .join("")}
  `;
}

function renderMapMinimap(map, tileSize, offset) {
  if (!elements.mapMinimap || !map?.terrain_codes) return;
  const maxWidth = 158;
  const maxHeight = 112;
  const scale = Math.min(maxWidth / Math.max(map.width, 1), maxHeight / Math.max(map.height, 1));
  const width = Math.round(map.width * scale);
  const height = Math.round(map.height * scale);
  const hostWidth = elements.canvasHost.clientWidth;
  const hostHeight = elements.canvasHost.clientHeight;
  const viewLeft = clamp((0 - offset.x) / tileSize, 0, map.width);
  const viewTop = clamp((0 - offset.y) / tileSize, 0, map.height);
  const viewRight = clamp((hostWidth - offset.x) / tileSize, 0, map.width);
  const viewBottom = clamp((hostHeight - offset.y) / tileSize, 0, map.height);
  const cells = [];
  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const code = map.terrain_codes[y][x];
      cells.push(`<i style="background:${terrainCssColor(Number(code))};"></i>`);
    }
  }

  elements.mapMinimap.innerHTML = `
    <div class="minimap-shell" style="width:${width}px;height:${height}px;">
      <div
        class="minimap-terrain"
        style="grid-template-columns:repeat(${map.width}, 1fr);"
      >${cells.join("")}</div>
      <div
        class="minimap-viewport"
        data-minimap-viewport
        style="
          left:${viewLeft * scale}px;
          top:${viewTop * scale}px;
          width:${Math.max(8, (viewRight - viewLeft) * scale)}px;
          height:${Math.max(8, (viewBottom - viewTop) * scale)}px;
        "
      ></div>
    </div>
  `;
}

function placeMapAnnotation(annotation, placed, map, tileSize, offset) {
  const labelPlacement = placeMapAnnotationCandidate(
    annotation,
    placed,
    map,
    tileSize,
    offset,
    {
      mode: "label",
      boxWidth: annotation.boxWidth ?? 176,
      boxHeight: annotation.boxHeight ?? 50,
    },
  );
  if (labelPlacement) return labelPlacement;
  return placeMapAnnotationCandidate(
    annotation,
    placed,
    map,
    tileSize,
    offset,
    {
      mode: "pin",
      boxWidth: 34,
      boxHeight: 34,
    },
  );
}

function placeMapAnnotationCandidate(annotation, placed, map, tileSize, offset, options) {
  const mapWidth = map.width * tileSize;
  const mapHeight = map.height * tileSize;
  const { mode, boxWidth, boxHeight } = options;
  const hostWidth = elements.canvasHost?.clientWidth ?? offset.x + mapWidth;
  const hostHeight = elements.canvasHost?.clientHeight ?? offset.y + mapHeight;
  const minX = Math.max(12 + boxWidth / 2, offset.x + boxWidth / 2);
  const maxX = Math.min(hostWidth - 12 - boxWidth / 2, offset.x + mapWidth - boxWidth / 2);
  const minY =
    mode === "pin"
      ? Math.max(12 + boxHeight / 2, offset.y + boxHeight / 2)
      : Math.max(12 + boxHeight + 12, offset.y + boxHeight + 12);
  const maxY =
    mode === "pin"
      ? Math.min(hostHeight - 12 - boxHeight / 2, offset.y + mapHeight - boxHeight / 2)
      : Math.min(hostHeight - 8, offset.y + mapHeight - 6);
  const spread = Math.max(54, tileSize * 4.6);
  const candidates =
    mode === "pin"
      ? [
          [0, 0],
          [tileSize * 1.2, 0],
          [-tileSize * 1.2, 0],
          [0, tileSize * 1.2],
          [0, -tileSize * 1.2],
          [tileSize * 1.8, tileSize * 1.8],
          [-tileSize * 1.8, tileSize * 1.8],
          [tileSize * 1.8, -tileSize * 1.8],
          [-tileSize * 1.8, -tileSize * 1.8],
        ]
      : [
          [0, 0],
          [spread, 0],
          [-spread, 0],
          [0, boxHeight + 12],
          [spread, boxHeight + 12],
          [-spread, boxHeight + 12],
          [spread * 1.25, boxHeight * 0.45],
          [-spread * 1.25, boxHeight * 0.45],
          [spread * 0.7, -(boxHeight + 12)],
          [-spread * 0.7, -(boxHeight + 12)],
          [0, -(boxHeight + 18)],
          [spread * 1.6, -(boxHeight + 10)],
          [-spread * 1.6, -(boxHeight + 10)],
        ];
  for (const [dx, dy] of candidates) {
    const x = clampToOrderedRange(annotation.x + dx, minX, maxX);
    const y = clampToOrderedRange(annotation.y + dy, minY, maxY);
    const box =
      mode === "pin"
        ? {
            left: x - boxWidth / 2,
            right: x + boxWidth / 2,
            top: y - boxHeight / 2,
            bottom: y + boxHeight / 2,
          }
        : {
            left: x - boxWidth / 2,
            right: x + boxWidth / 2,
            top: y - boxHeight - 12,
            bottom: y + 8,
          };
    if (!placed.some((candidate) => boxesOverlap(box, candidate))) {
      placed.push(box);
      return { ...annotation, x: Math.round(x), y: Math.round(y), mode, boxWidth, boxHeight };
    }
  }
  return null;
}

function boxesOverlap(left, right, margin = 8) {
  return !(
    left.right + margin < right.left ||
    left.left - margin > right.right ||
    left.bottom + margin < right.top ||
    left.top - margin > right.bottom
  );
}

function updateHoveredTileFromPointer(event) {
  if (!state.payload || !elements.canvasHost) return;
  const map = state.payload.viewer.map;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);
  const rect = elements.canvasHost.getBoundingClientRect();
  const localX = event.clientX - rect.left;
  const localY = event.clientY - rect.top;
  const x = Math.floor((localX - offset.x) / tileSize);
  const y = Math.floor((localY - offset.y) / tileSize);
  if (!isPointOnMap(x, y)) {
    state.hoveredTile = null;
  } else {
    state.hoveredTile = { x, y };
  }
  renderTileHover();
  renderTileInspector();
  syncDebugState();
}

function renderTileHover() {
  const tile = state.hoveredTile ?? state.explainedTile;
  if (!elements.tileHover || !state.payload || !tile) {
    elements.tileHover?.classList.add("hidden");
    return;
  }
  const map = state.payload.viewer.map;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);
  elements.tileHover.classList.remove("hidden");
  elements.tileHover.classList.toggle("locked", Boolean(!state.hoveredTile && state.explainedTile));
  elements.tileHover.style.left = `${Math.round(offset.x + tile.x * tileSize)}px`;
  elements.tileHover.style.top = `${Math.round(offset.y + tile.y * tileSize)}px`;
  elements.tileHover.style.width = `${tileSize}px`;
  elements.tileHover.style.height = `${tileSize}px`;
}

function renderTileInspector() {
  if (!elements.tileInspector || !state.payload) return;
  if (!state.hoveredTile) {
    elements.tileInspector.innerHTML = '<span>Tile</span><strong>Hover the map</strong>';
    return;
  }
  const context = tileContextForPoint(state.hoveredTile.x, state.hoveredTile.y, state.currentFrameIndex);
  const support = hydrologySupportFromCode(context.hydrologySupportCode);
  elements.tileInspector.innerHTML = `
    <span>Tile ${escapeHtml(String(state.hoveredTile.x))}, ${escapeHtml(String(state.hoveredTile.y))}</span>
    <strong>${escapeHtml(terrainLabel(context.terrainName))}</strong>
    <dl>
      <div><dt>Terrain</dt><dd>${escapeHtml(terrainLabel(context.terrainName))}</dd></div>
      <div><dt>Hydrology</dt><dd>${escapeHtml(hydrologyLabel(context.hydrologyReason))}</dd></div>
      <div><dt>Support</dt><dd>${escapeHtml(tileSupportLabel(support))}</dd></div>
      <div><dt>Hazard</dt><dd>${escapeHtml(hazardLabel(context.hazardType))} · ${escapeHtml(formatPercent(context.hazardLevel ?? 0))}</dd></div>
      <div><dt>Ecology</dt><dd>${escapeHtml(ecologyLabel(context.ecologyState))}</dd></div>
      <div><dt>Habitat</dt><dd>${escapeHtml(habitatLabel(context.habitatState))}</dd></div>
      <div><dt>Carcass</dt><dd>${escapeHtml(formatPercent(context.carcassEnergy ?? 0))} energy</dd></div>
      <div><dt>Signal</dt><dd>Repro ${escapeHtml(roundValue(context.reproductiveSignal ?? 0).toString())} · comm ${escapeHtml(roundValue(context.communicationSignal ?? 0).toString())}</dd></div>
    </dl>
  `;
}

function renderTileExplainer() {
  if (!elements.tileExplainer || !state.payload) return;
  const tile = state.explainedTile;
  if (!tile) {
    elements.tileExplainer.innerHTML = `
      <span>Tile Explanation</span>
      <strong>Click a tile to explain what is happening there.</strong>
    `;
    return;
  }

  const context = tileContextForPoint(tile.x, tile.y, state.currentFrameIndex);
  const support = hydrologySupportFromCode(context.hydrologySupportCode);
  const agents = decodedAgentsForFrame(state.currentFrameIndex)
    .filter((agent) => agent.x === tile.x && agent.y === tile.y)
    .sort((left, right) => left.agentId - right.agentId);
  const events = eventsForTile(tile.x, tile.y, state.payload.viewer.frames[state.currentFrameIndex]?.tick ?? 0);
  const eventCounts = eventCategoryCounts(events);
  const pressureTags = tilePressureTags(context, agents, eventCounts);

  elements.tileExplainer.innerHTML = `
    <span>Tile Explanation</span>
    <strong>Tile ${escapeHtml(String(tile.x))}, ${escapeHtml(String(tile.y))}: ${escapeHtml(terrainLabel(context.terrainName))}</strong>
    <p>${escapeHtml(tileExplanationSentence(context, agents, eventCounts, { hazardLabel, formatPercent }))}</p>
    <div class="tile-pressure-tags">
      ${pressureTags.map((tag) => `<i class="${escapeHtml(tag.kind)}">${escapeHtml(tag.label)}</i>`).join("")}
    </div>
    <dl>
      <div><dt>Hydrology</dt><dd>${escapeHtml(hydrologyLabel(context.hydrologyReason))}</dd></div>
      <div><dt>Support</dt><dd>${escapeHtml(tileSupportLabel(support))}</dd></div>
      <div><dt>Hazard</dt><dd>${escapeHtml(hazardLabel(context.hazardType))} · ${escapeHtml(formatPercent(context.hazardLevel ?? 0))}</dd></div>
      <div><dt>Ecology</dt><dd>${escapeHtml(ecologyLabel(context.ecologyState))}</dd></div>
      <div><dt>Habitat</dt><dd>${escapeHtml(habitatLabel(context.habitatState))}</dd></div>
      <div><dt>Vegetation</dt><dd>${escapeHtml(formatPercent(context.vegetation ?? 0))}</dd></div>
      <div><dt>Carcass</dt><dd>${escapeHtml(formatPercent(context.carcassEnergy ?? 0))} energy · ${escapeHtml(formatPercent(context.carcassFreshness ?? 0))} fresh</dd></div>
      <div><dt>Signals</dt><dd>Repro ${escapeHtml(roundValue(context.reproductiveSignal ?? 0).toString())} · comm ${escapeHtml(roundValue(context.communicationSignal ?? 0).toString())}</dd></div>
    </dl>
    <div class="tile-agent-list">
      <strong>Agents Here</strong>
      ${
        agents.length
          ? agents
              .map(
                (agent) => `
                  <button type="button" data-episode-agent="${escapeHtml(String(agent.agentId))}">
                    <span>Agent ${escapeHtml(String(agent.agentId))}</span>
                    <small>${escapeHtml(trophicRoleLabel(agent.trophicRole))} · ${escapeHtml(speciesLabel(agent.speciesId))}</small>
                  </button>
                `,
              )
              .join("")
          : '<div class="muted">No living agent occupies this tile on the selected tick.</div>'
      }
    </div>
    <div class="tile-event-list">
      <strong>Recent Local Events</strong>
      ${
        events.length
          ? events
              .slice(0, 8)
              .map((event) => {
                const item = tileEventLine(event);
                return `
                  <button type="button" data-episode-jump="${escapeHtml(String(event.tick))}">
                    <span>Tick ${escapeHtml(String(event.tick))}</span>
                    <strong>${escapeHtml(item.title)}</strong>
                    <small>${escapeHtml(item.detail)}</small>
                  </button>
                `;
              })
              .join("")
          : '<div class="muted">No local birth, death, attack, feeding, or carcass event was recorded recently.</div>'
      }
    </div>
  `;
}

function eventsForTile(x, y, currentTick) {
  const startTick = Math.max(0, Number(currentTick) - EVENT_LENS_FRAME_WINDOW);
  const matches = [];
  for (let tick = Number(currentTick); tick >= startTick; tick -= 1) {
    const frameIndex = frameIndexForTick(tick) ?? state.currentFrameIndex;
    const agentById = new Map(decodedAgentsForFrame(frameIndex).map((agent) => [agent.agentId, agent]));
    for (const event of state.eventsByTick.get(tick) ?? []) {
      if (!eventLensCategory(event) && !["agent_ate", "agent_drank", "agent_moved"].includes(event.type)) {
        continue;
      }
      const marker = eventMarkerPosition(event, agentById, tick);
      if (!marker || marker.x !== x || marker.y !== y) continue;
      matches.push(event);
    }
  }
  return matches.sort((left, right) => {
    if (left.tick !== right.tick) return right.tick - left.tick;
    return String(left.type).localeCompare(String(right.type));
  });
}

function tileEventLine(event) {
  const data = event.data ?? {};
  if (event.type === "agent_attacked") {
    return {
      title: "Attack",
      detail: `Agent ${event.agent_id} ${data.success ? "hit" : "missed"} agent ${data.target_id}`,
    };
  }
  if (event.type === "agent_reproduced") {
    return { title: "Birth", detail: `Child ${data.child_id} born by ${reproductionModeLabel(data.reproduction_mode)}` };
  }
  if (event.type === "agent_died") {
    return { title: "Death", detail: `Agent ${event.agent_id} died from ${deathCauseLabel(data.cause)}` };
  }
  if (event.type === "carcass_deposited") {
    return { title: "Carcass", detail: `${roundValue(data.deposited_energy ?? 0)} energy deposited` };
  }
  if (event.type === "agent_ate") {
    return { title: "Feeding", detail: `${foodSourceLabel(data.food_source)} for agent ${event.agent_id}` };
  }
  if (event.type === "agent_drank") {
    return { title: "Drinking", detail: `${waterAccessLabel(data.water_access_reason)} water access` };
  }
  if (event.type === "agent_moved") {
    return { title: "Movement", detail: `Agent ${event.agent_id} moved here` };
  }
  return { title: titleCase(event.type), detail: "Recorded local event" };
}

function tileSupportLabel(support) {
  const labels = [];
  if (support.adjacentToWater) labels.push("Shoreline");
  if (support.wetland) labels.push("Wetland");
  if (support.flooded) labels.push("Flooded");
  return labels.length ? labels.join(", ") : "No soft support";
}

function eventMarkerPosition(event, agentById, tick = event.tick) {
  const data = event.data ?? {};
  if (event.type === "agent_reproduced") {
    return { x: data.child_x, y: data.child_y };
  }
  if (event.type === "agent_died" || event.type === "carcass_deposited") {
    return { x: data.x, y: data.y };
  }
  if (data.x != null && data.y != null) {
    return { x: data.x, y: data.y };
  }
  if (event.type === "agent_drank" && data.source_x != null && data.source_y != null) {
    return { x: data.source_x, y: data.source_y };
  }
  if (event.type === "agent_attacked") {
    const attacker = agentById.get(event.agent_id) ?? agentSnapshotAtTick(event.agent_id, tick);
    const target = agentById.get(data.target_id) ?? agentSnapshotAtTick(data.target_id, tick);
    if (target) {
      return { x: target.x, y: target.y, from: attacker };
    }
  }
  const agent = agentById.get(event.agent_id) ?? agentSnapshotAtTick(event.agent_id, tick);
  return agent ? { x: agent.x, y: agent.y } : null;
}

function agentSnapshotAtTick(agentId, tick) {
  const positions = state.replayModel?.agentPositionsById?.get(Number(agentId)) ?? [];
  for (let index = positions.length - 1; index >= 0; index -= 1) {
    if (Number(positions[index].tick) <= Number(tick)) return positions[index];
  }
  return null;
}

function tileCenter(x, y, tileSize, offset) {
  return {
    x: offset.x + x * tileSize + tileSize / 2,
    y: offset.y + y * tileSize + tileSize / 2,
  };
}

function isPointOnMap(x, y) {
  const map = state.payload?.viewer?.map;
  return (
    map &&
    Number.isFinite(x) &&
    Number.isFinite(y) &&
    x >= 0 &&
    y >= 0 &&
    x < map.width &&
    y < map.height
  );
}

function renderStoryFrame(frame, decodedAgents) {
  if (!elements.sceneSummary || !elements.storyFeed || !state.payload) return;

  const payload = state.payload;
  const previousFrame =
    state.currentFrameIndex > 0 ? payload.viewer.frames[state.currentFrameIndex - 1] : null;
  const birthDelta = Math.max(0, frame.births - (previousFrame?.births ?? frame.births));
  const deathDelta = Math.max(0, frame.deaths - (previousFrame?.deaths ?? frame.deaths));
  const roleCounts = frame.trophic_role_counts ?? countAgentRoles(decodedAgents);
  const dominantRole = dominantCountEntry(roleCounts);
  const speciesCounts = [...frame.species_counts].sort((left, right) => right[1] - left[1]);
  const topSpecies = speciesCounts[0] ?? null;
  const hazardTiles = frame.hazard_stats?.hazardous_tiles ?? 0;
  const mapTiles = Math.max(payload.viewer.map.width * payload.viewer.map.height, 1);
  const hazardShare = hazardTiles / mapTiles;
  const carcassTiles = frame.carcass_stats?.carcass_tiles ?? 0;
  const hardWaterTiles = frame.hydrology_primary_stats?.hard_access_tiles ?? 0;
  const climate = frame.field_state?.disturbance_type ?? "quiet";
  const season = frame.season ?? "unknown";
  const pressure = describePrimaryPressure(frame, hazardShare, carcassTiles);
  const tickEventCounts = eventCountsForTick(frame.tick);

  elements.sceneSummary.textContent = [
    `Tick ${frame.tick}: ${frame.alive_agents} agents across ${frame.species_counts.length} live species.`,
    `The world is in ${season} season with ${climate} pressure.`,
    pressure,
  ].join(" ");

  elements.sceneStats.innerHTML = [
    ["Alive", frame.alive_agents],
    ["Species", frame.species_counts.length],
    ["Births Now", birthDelta],
    ["Deaths Now", deathDelta],
    ["Main Diet", titleCase(dominantRole?.key ?? "-")],
    ["Hard Water", hardWaterTiles],
  ]
    .map(
      ([label, value]) => `
        <div class="scene-stat">
          <span>${escapeHtml(String(label))}</span>
          <strong>${escapeHtml(String(formatValue(value)))}</strong>
        </div>
      `,
    )
    .join("");

  const notes = [
    {
      kind: "world",
      title: "World",
      text: describeWorld(frame, hazardShare, hardWaterTiles),
    },
    {
      kind: "population",
      title: "Population",
      text: describePopulation(frame, birthDelta, deathDelta, topSpecies),
    },
    {
      kind: "food",
      title: "Food Web",
      text: describeFoodWeb(roleCounts, carcassTiles),
    },
    {
      kind: "event",
      title: "This Tick",
      text: describeTickEvents(frame, birthDelta, deathDelta, tickEventCounts),
    },
  ];
  const selectedNote = describeSelectedFocus(frame, decodedAgents);
  if (selectedNote) {
    notes.push(selectedNote);
  }

  elements.storyFeed.innerHTML = notes
    .map(
      (note) => `
        <div class="story-bubble ${escapeHtml(note.kind)}">
          <span>${escapeHtml(note.title)}</span>
          <p>${escapeHtml(note.text)}</p>
        </div>
      `,
    )
    .join("");

  renderFocusCards(frame, decodedAgents, speciesCounts);
}

function renderAgentDossier(frame, decodedAgents) {
  if (!elements.agentDossier || !elements.agentDossierBody || !state.payload) return;
  if (state.selectedAgentId == null) {
    elements.agentDossierBody.innerHTML =
      '<div class="focus-note">Select an agent on the map to inspect its condition and life timeline.</div>';
    return;
  }

  const payload = state.payload;
  const agentId = state.selectedAgentId;
  const catalog = payload.viewer.agent_catalog[String(agentId)];
  if (!catalog) {
    elements.agentDossierBody.innerHTML =
      '<div class="focus-note">The selected agent is missing from the replay catalog.</div>';
    return;
  }

  const current = decodedAgents.find((agent) => agent.agentId === agentId) ?? null;
  const latest = current ?? selectedAgentSnapshotAtOrBefore(agentId, state.currentFrameIndex);
  const snapshot = current ?? latest;
  const snapshotFrameIndex = current ? state.currentFrameIndex : latest?.frameIndex ?? state.currentFrameIndex;
  const tileContext = tileContextForAgent(snapshot, snapshotFrameIndex);
  const speciesId = current?.speciesId ?? latest?.speciesId ?? state.selectedSpeciesId ?? catalog.lineage_id;
  const speciesRecord =
    speciesId != null ? payload.viewer.species_catalog[String(speciesId)] : null;
  const deathEvent =
    agentEvents(agentId).find(
      (event) => event.type === "agent_died" && event.agent_id === agentId,
    ) ?? null;
  const lifeEvents = agentEvents(agentId).slice(-18).reverse();
  const decisionRecords = trajectoryRecords(agentId);
  const currentDecision = decisionRecordAtOrBefore(agentId, frame.tick);
  const speciesTrend = describeSpeciesTrend(speciesId);
  const stressTags = agentStressTags(current, deathEvent);
  const statusText = current
    ? `Alive at ${current.x},${current.y}`
    : deathEvent
      ? `Dead since tick ${deathEvent.tick}`
      : "Not alive on this tick";
  const conditionHeading = current ? "Current Condition" : "Last Known Condition";

  const metrics = [
    ["Energy", current?.energyRatio ?? latest?.energyRatio ?? 0, "energy"],
    ["Hydration", current?.hydrationRatio ?? latest?.hydrationRatio ?? 0, "hydration"],
    ["Health", current?.healthRatio ?? latest?.healthRatio ?? 0, "health"],
  ];
  const keyFacts = [
    ["Species", speciesId != null ? `${speciesLabel(speciesId)} (${speciesCode(speciesId)})` : MISSING_LABEL],
    ["Lineage", catalog.lineage_id],
    ["Age", current?.age ?? latest?.age ?? MISSING_LABEL],
    ["Role", trophicRoleLabel(current?.trophicRole ?? latest?.trophicRole)],
    ["Mode", meatModeLabel(current?.meatMode ?? latest?.meatMode)],
    ["Water", waterAccessLabel(current?.waterAccessReason ?? latest?.waterAccessReason)],
    ["Terrain", terrainLabel(tileContext.terrainName)],
    ["Habitat", habitatLabel(tileContext.habitatState)],
    ["Hazard", hazardLabel(tileContext.hazardType)],
    ["Parents", Array.isArray(catalog.parent_ids) && catalog.parent_ids.length ? catalog.parent_ids.join(", ") : "root"],
    ["Birth Tick", catalog.birth_tick],
    [
      "Death",
      deathStatusLabel(deathEvent, current, frame.tick),
    ],
  ];

  elements.agentDossierBody.innerHTML = `
    <div class="agent-dossier-lede">
      <div>
        <div class="agent-title-row">
          <span class="agent-token" style="background:#${colorForSpecies(speciesId).toString(16).padStart(6, "0")};"></span>
          <strong>Agent ${escapeHtml(String(agentId))}</strong>
          <span class="agent-status ${current ? "alive" : "dead"}">${escapeHtml(statusText)}</span>
        </div>
        <p>${escapeHtml(agentFocusSentence(current, latest, speciesTrend, stressTags, deathEvent))}</p>
      </div>
    </div>

    ${renderAgentDossierSummary({
      current,
      latest,
      currentDecision,
      tileContext,
      speciesTrend,
    })}

    ${renderDossierSection(conditionHeading, `
      <div class="condition-bars">
        ${metrics
          .map(
            ([label, value, kind]) => `
              <div class="condition-row ${escapeHtml(kind)}">
                <span>${escapeHtml(label)}</span>
                <div class="condition-track">
                  <i style="width:${clamp01(Number(value ?? 0)) * 100}%;"></i>
                </div>
                <strong>${escapeHtml(formatPercent(value ?? 0))}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
      <div class="stress-tags">
        ${stressTags.map((tag) => `<span class="${escapeHtml(tag.kind)}">${escapeHtml(tag.label)}</span>`).join("")}
      </div>
    `, { priority: "primary", open: true })}

    ${renderDossierSection("Agent Context", `
      <dl class="agent-fact-grid">
        ${keyFacts
          .map(
            ([label, value]) => `
              <div>
                <dt>${escapeHtml(String(label))}</dt>
                <dd>${escapeHtml(String(value ?? MISSING_LABEL))}</dd>
              </div>
            `,
          )
          .join("")}
      </dl>
    `, { priority: "primary", open: true })}

    ${renderDossierSection("Decision Trace", `
      ${renderAgentDecisionTrace(decisionRecords, currentDecision)}
    `, { priority: "secondary", open: true })}

    ${renderDossierSection("Reward Components", renderRewardComponents(currentDecision), {
      priority: "deep",
      open: false,
    })}

    ${renderDossierSection("Observation Patch", renderObservationPatch(currentDecision), {
      priority: "deep",
      open: false,
    })}

    ${renderDossierSection("Life Timeline", `
      <div id="agent-action-timeline" class="agent-action-timeline">
        ${
          lifeEvents.length
            ? lifeEvents.map((event) => renderAgentTimelineItem(event, agentId)).join("")
            : '<div class="muted">No recorded agent events before this tick.</div>'
        }
      </div>
    `, { priority: "secondary", open: false })}
  `;
}

function renderDossierSection(title, bodyHtml, options = {}) {
  const priority = options.priority ?? "secondary";
  const open = options.open ? " open" : "";
  return `
    <details
      class="dossier-section"
      data-dossier-priority="${escapeHtml(priority)}"
      data-dossier-section="${escapeHtml(safeFilePart(title))}"
      ${open}
    >
      <summary>
        <h3>${escapeHtml(title)}</h3>
        <span>${escapeHtml(priority === "primary" ? "Primary" : priority === "deep" ? "Details" : "Trace")}</span>
      </summary>
      <div class="dossier-section-body">
        ${bodyHtml}
      </div>
    </details>
  `;
}

function renderAgentDossierSummary({ current, latest, currentDecision, tileContext, speciesTrend }) {
  const vitalSource = current ?? latest ?? {};
  const vitals = [
    ["energy", "Energy", vitalSource.energyRatio ?? 0],
    ["hydration", "Hydration", vitalSource.hydrationRatio ?? 0],
    ["health", "Health", vitalSource.healthRatio ?? 0],
  ];
  const requested = currentDecision ? actionLabel(currentDecision.requested_action) : VIEWER_EMPTY_LABELS.not_applicable;
  const resolved = currentDecision ? actionLabel(currentDecision.resolved_action) : VIEWER_EMPTY_LABELS.not_applicable;
  const rewardTotal = currentDecision
    ? roundValue(currentDecision.reward?.total ?? 0).toString()
    : VIEWER_EMPTY_LABELS.not_applicable;
  return `
    <div class="agent-dossier-summary">
      <div class="agent-primary-card" data-agent-primary-card="decision">
        <span>Decision</span>
        <strong>${escapeHtml(resolved)}</strong>
        <small>Requested ${escapeHtml(requested)} · reward ${escapeHtml(rewardTotal)}</small>
      </div>
      <div class="agent-primary-card" data-agent-primary-card="environment">
        <span>Tile Context</span>
        <strong>${escapeHtml(terrainLabel(tileContext.terrainName))}</strong>
        <small>${escapeHtml(hydrologyLabel(tileContext.hydrologyReason))} · ${escapeHtml(hazardLabel(tileContext.hazardType))}</small>
      </div>
      <div class="agent-primary-card" data-agent-primary-card="trend">
        <span>Species Trend</span>
        <strong>${escapeHtml(titleCase(speciesTrend))}</strong>
        <small>${escapeHtml(habitatLabel(tileContext.habitatState))} habitat</small>
      </div>
      <div class="agent-vital-grid">
        ${vitals
          .map(
            ([kind, label, value]) => `
              <div class="agent-vital-card ${escapeHtml(kind)}" data-agent-vital="${escapeHtml(kind)}">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(formatPercent(value ?? 0))}</strong>
                <i style="width:${clamp01(Number(value ?? 0)) * 100}%;"></i>
              </div>
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function selectedAgentSnapshotAtOrBefore(agentId, frameIndex) {
  const positions = state.replayModel?.agentPositionsById?.get(agentId) ?? [];
  const frame = state.payload?.viewer?.frames?.[frameIndex] ?? null;
  const currentTick = frame?.tick ?? Number.POSITIVE_INFINITY;
  for (let index = positions.length - 1; index >= 0; index -= 1) {
    if (positions[index].tick <= currentTick) return positions[index];
  }
  return null;
}

function tileContextForAgent(agent, frameIndex = state.currentFrameIndex) {
  const payload = state.payload;
  const frame = payload?.viewer?.frames?.[frameIndex] ?? null;
  const map = payload?.viewer?.map ?? null;
  if (!payload || !frame || !map || !agent || !isPointOnMap(agent.x, agent.y)) {
    return {
      terrainName: null,
      habitatState: null,
      ecologyState: null,
      hazardType: null,
      hazardLevel: null,
      carcassEnergy: null,
      reproductiveSignal: null,
      communicationSignal: null,
      carcassPatch: null,
    };
  }

  const x = Number(agent.x);
  const y = Number(agent.y);
  return tileContextForPoint(x, y, frameIndex);
}

function tileContextForPoint(x, y, frameIndex = state.currentFrameIndex) {
  const payload = state.payload;
  const frame = payload?.viewer?.frames?.[frameIndex] ?? null;
  const map = payload?.viewer?.map ?? null;
  if (!payload || !frame || !map || !isPointOnMap(x, y)) {
    return {
      onMap: false,
      terrainName: null,
      hydrologyReason: null,
      hydrologySupportCode: 0,
      habitatState: null,
      ecologyState: null,
      hazardType: null,
      hazardLevel: null,
      carcassEnergy: null,
      carcassFreshness: null,
      reproductiveSignal: null,
      communicationSignal: null,
      vegetation: null,
      recoveryDebt: null,
      carcassPatch: null,
    };
  }

  const gridX = Number(x);
  const gridY = Number(y);
  return {
    onMap: true,
    x: gridX,
    y: gridY,
    terrainName: terrainNameFromCode(map, map.terrain_codes?.[gridY]?.[gridX]),
    hydrologyReason: hydrologyReasonNameFromCode(frame.hydrology_primary_codes?.[gridY]?.[gridX] ?? 0),
    hydrologySupportCode: frame.hydrology_support_codes?.[gridY]?.[gridX] ?? 0,
    habitatState: habitatStateNameFromCode(frame.habitat_state_codes?.[gridY]?.[gridX] ?? 0),
    ecologyState: ecologyStateNameFromCode(frame.ecology_state_codes?.[gridY]?.[gridX] ?? 0),
    hazardType: hazardTypeNameFromCode(frame.hazard_type_codes?.[gridY]?.[gridX] ?? 0),
    hazardLevel: (frame.hazard_level_codes?.[gridY]?.[gridX] ?? 0) / 100,
    carcassEnergy: (frame.carcass_energy_codes?.[gridY]?.[gridX] ?? 0) / 100,
    carcassFreshness: (frame.carcass_freshness_codes?.[gridY]?.[gridX] ?? 0) / 100,
    reproductiveSignal: frame.signal_fields?.reproductive_signal?.[gridY]?.[gridX] ?? 0,
    communicationSignal: frame.signal_fields?.communication_signal?.[gridY]?.[gridX] ?? 0,
    vegetation: effectiveFieldValue(payload, frame, "fertility", gridX, gridY),
    recoveryDebt: frame.tile_recovery_debt_codes?.[gridY]?.[gridX] ?? null,
    carcassPatch:
      frame.carcass_patches?.find((patch) => patch.x === gridX && patch.y === gridY) ?? null,
  };
}

function observationCellTitle(context, x, y) {
  if (!context?.onMap) return `Tile ${x},${y} is outside the replay map`;
  return [
    `Tile ${x},${y}`,
    `Terrain ${terrainLabel(context.terrainName)}`,
    `Hydrology ${hydrologyLabel(context.hydrologyReason)}`,
    `Hazard ${hazardLabel(context.hazardType)}`,
    `Ecology ${ecologyLabel(context.ecologyState)}`,
  ].join(" · ");
}

function agentEvents(agentId) {
  return state.replayModel?.agentEventsById?.get(agentId) ?? [];
}

function trajectoryRecords(agentId) {
  return state.replayModel?.trajectoryRecordsByAgentId?.get(Number(agentId)) ?? [];
}

function decisionRecordAtOrBefore(agentId, tick) {
  const records = trajectoryRecords(agentId);
  let best = null;
  for (const record of records) {
    if (Number(record.tick) <= Number(tick)) {
      best = record;
    } else {
      break;
    }
  }
  return best ?? records[0] ?? null;
}

function renderAgentDecisionTrace(records, currentDecision) {
  if (!records.length) {
    return '<div id="agent-decision-trace" class="decision-trace"><div class="muted">No trajectory decisions were recorded for this agent.</div></div>';
  }
  const currentTick = state.payload.viewer.frames[state.currentFrameIndex]?.tick ?? 0;
  const visible = records
    .filter((record) => Number(record.tick) <= currentTick)
    .slice(-8)
    .reverse();
  const trace = visible.length ? visible : records.slice(0, 8);
  return `
    <div id="agent-decision-trace" class="decision-trace">
      ${trace
        .map((record) => {
          const selected = currentDecision && Number(record.tick) === Number(currentDecision.tick);
          const validity = record.action_valid && record.resolution_action_valid ? "valid" : "invalid";
          return `
            <button
              type="button"
              class="decision-trace-item ${validity}${selected ? " current" : ""}"
              data-decision-tick="${escapeHtml(String(record.tick))}"
            >
              <span>Tick ${escapeHtml(String(record.tick))}</span>
              <strong>Requested ${escapeHtml(actionLabel(record.requested_action))}</strong>
              <small>Resolved ${escapeHtml(actionLabel(record.resolved_action))} · ${escapeHtml(policyLabel(record))}</small>
            </button>
          `;
        })
        .join("")}
    </div>
    ${renderDecisionSummary(currentDecision)}
  `;
}

function renderDecisionSummary(record) {
  if (!record) return "";
  const actionMask = maskSummary(record.action_mask);
  const resolutionMask = maskSummary(record.resolution_action_mask);
  const invalidReason =
    record.outcome?.invalid_reason ??
    record.outcome?.signal?.invalid_reason ??
    null;
  return `
    <dl class="decision-summary-grid">
      <div><dt>Requested</dt><dd>${escapeHtml(actionLabel(record.requested_action))}</dd></div>
      <div><dt>Resolved</dt><dd>${escapeHtml(actionLabel(record.resolved_action))}</dd></div>
      <div><dt>Source</dt><dd>${escapeHtml(policyLabel(record))}</dd></div>
      <div><dt>Validity</dt><dd>${record.action_valid ? "Requested valid" : "Requested invalid"} · ${record.resolution_action_valid ? "resolved valid" : "resolved invalid"}</dd></div>
      <div><dt>Action Mask</dt><dd>${escapeHtml(actionMask)}</dd></div>
      <div><dt>Resolution Mask</dt><dd>${escapeHtml(resolutionMask)}</dd></div>
      <div><dt>Reward Total</dt><dd>${escapeHtml(roundValue(record.reward?.total ?? 0).toString())}</dd></div>
      <div><dt>Invalid Reason</dt><dd>${escapeHtml(invalidReason ? titleCase(invalidReason) : VIEWER_EMPTY_LABELS.not_applicable)}</dd></div>
    </dl>
  `;
}

function renderRewardComponents(record) {
  const components = record?.reward?.components ?? null;
  if (!components) {
    return '<div class="reward-components"><strong>Reward Components</strong><div class="muted">No reward component breakdown recorded.</div></div>';
  }
  const entries = Object.entries(components).sort((left, right) => {
    const magnitude = Math.abs(right[1]) - Math.abs(left[1]);
    return magnitude || left[0].localeCompare(right[0]);
  });
  const maxMagnitude = Math.max(0.001, ...entries.map(([, value]) => Math.abs(Number(value) || 0)));
  return `
    <div class="reward-components">
      <strong>Reward Components</strong>
      ${entries
        .map(([name, value]) => {
          const numeric = Number(value) || 0;
          const width = Math.max(3, (Math.abs(numeric) / maxMagnitude) * 100);
          return `
            <div class="reward-row ${numeric < 0 ? "negative" : "positive"}">
              <span>${escapeHtml(titleCase(name))}</span>
              <i style="width:${width}%;"></i>
              <b>${escapeHtml(roundValue(numeric).toString())}</b>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderObservationPatch(record) {
  if (!record || !state.payload) {
    return '<div id="agent-observation-patch" class="observation-patch"><strong>Observation Patch</strong><div class="muted">No observation patch available.</div></div>';
  }
  const point = record.before ?? record.after ?? {};
  const frameIndex = frameIndexForTick(Number(record.tick)) ?? state.currentFrameIndex;
  const cells = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const x = Number(point.x) + dx;
      const y = Number(point.y) + dy;
      const context = tileContextForPoint(x, y, frameIndex);
      cells.push({ x, y, dx, dy, context });
    }
  }
  return `
    <div id="agent-observation-patch" class="observation-patch">
      <strong>Observation Patch</strong>
      <span>Terrain, hydrology, hazard, and ecology around the decision tile.</span>
      <div class="observation-grid">
        ${cells
          .map(({ x, y, dx, dy, context }) => {
            const offMap = !context.onMap;
            const center = dx === 0 && dy === 0;
            return `
              <div
                class="observation-cell${center ? " center" : ""}${offMap ? " off-map" : ""}"
                data-observation-cell
                title="${escapeHtml(observationCellTitle(context, x, y))}"
              >
                <span>${escapeHtml(offMap ? "Off map" : terrainLabel(context.terrainName))}</span>
                <small>${escapeHtml(offMap ? `${x},${y}` : `${hydrologyLabel(context.hydrologyReason)} · ${hazardLabel(context.hazardType)}`)}</small>
                <em>${escapeHtml(offMap ? VIEWER_EMPTY_LABELS.not_applicable : ecologyLabel(context.ecologyState))}</em>
              </div>
            `;
          })
          .join("")}
      </div>
    </div>
  `;
}

function actionLabel(value) {
  return titleCase(String(value ?? UNKNOWN_LABEL));
}

function policyLabel(record) {
  const sourceKey = record?.action_source ?? "unknown_source";
  const policyKey = record?.policy_id ?? null;
  if (String(sourceKey).includes("observation_heuristic_confidence_delegate_v1")) {
    return "Mind confidence delegate";
  }
  if (String(sourceKey).includes("observation_heuristic_safety_floor_v1")) {
    return "Mind hard guard";
  }
  if (String(sourceKey).includes("mind_v1_learned_policy")) {
    return "Mind learned";
  }
  const source = titleCase(sourceKey);
  const policy = policyKey ? titleCase(policyKey) : null;
  if (policyKey && normalizedLabelKey(sourceKey) === normalizedLabelKey(policyKey)) {
    return normalizedLabelKey(policyKey).includes("heuristic") ? "Heuristic" : policy;
  }
  return policy ? `${source} / ${policy}` : source;
}

function normalizedLabelKey(value) {
  return String(value ?? "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean)
    .sort()
    .join("_");
}

function maskSummary(mask) {
  if (!mask || typeof mask !== "object") return UNKNOWN_LABEL;
  const entries = Array.isArray(mask) ? mask.map((value, index) => [String(index), value]) : Object.entries(mask);
  const allowed = entries.filter(([, value]) => Boolean(value)).length;
  return `${allowed}/${entries.length} allowed`;
}

function renderAgentTimelineItem(event, selectedAgentId) {
  const summary = describeAgentEvent(event, selectedAgentId);
  const selected = event.tick === state.payload.viewer.frames[state.currentFrameIndex]?.tick;
  return `
    <button
      type="button"
      class="agent-event-item ${escapeHtml(event.type)}${selected ? " current" : ""}"
      data-agent-event-tick="${event.tick}"
    >
      <span>Tick ${escapeHtml(String(event.tick))}</span>
      <strong>${escapeHtml(summary.title)}</strong>
      <small>${escapeHtml(summary.detail)}</small>
    </button>
  `;
}

function describeAgentEvent(event, selectedAgentId) {
  const data = event.data ?? {};
  if (event.type === "agent_moved") {
    return { title: "Moved", detail: `${titleCase(data.action ?? "move")} to ${data.x},${data.y}` };
  }
  if (event.type === "agent_ate") {
    return {
      title: "Ate",
      detail: `${foodSourceLabel(data.food_source)} gained ${roundValue(data.gained_energy ?? 0)} energy`,
    };
  }
  if (event.type === "agent_drank") {
    return {
      title: "Drank",
      detail: `${waterAccessLabel(data.water_access_reason)} water access`,
    };
  }
  if (event.type === "agent_damaged") {
    return {
      title: "Damaged",
      detail: `${roundValue(data.amount ?? 0)} from ${damageSourceLabel(data.source)}`,
    };
  }
  if (event.type === "agent_healed") {
    return { title: "Healed", detail: `${roundValue(data.amount ?? 0)} health recovered` };
  }
  if (event.type === "agent_attacked") {
    const role = event.agent_id === selectedAgentId ? "Attacked" : "Was attacked";
    return {
      title: role,
      detail: `${data.success ? "Hit" : "Miss"}${data.kill ? ", kill" : ""} for ${roundValue(data.damage ?? 0)}`,
    };
  }
  if (event.type === "agent_reproduced") {
    const title = data.child_id === selectedAgentId ? "Born" : "Reproduced";
    return {
      title,
      detail: `${reproductionModeLabel(data.reproduction_mode)} child ${data.child_id ?? "-"}`,
    };
  }
  if (event.type === "agent_died") {
    return {
      title: "Died",
      detail: `${deathCauseLabel(data.cause)} at ${data.x},${data.y}`,
    };
  }
  if (event.type === "carcass_deposited") {
    return {
      title: "Carcass",
      detail: `${roundValue(data.deposited_energy ?? 0)} energy deposited`,
    };
  }
  return { title: titleCase(event.type), detail: "Recorded event" };
}

function agentStressTags(current, deathEvent) {
  if (!current) {
    return [
      {
        label: deathEvent ? `Dead: ${deathCauseLabel(deathEvent.data?.cause)}` : "Not alive",
        kind: "danger",
      },
    ];
  }
  const tags = [];
  if ((current.energyRatio ?? 0) < 0.35) tags.push({ label: "Starving", kind: "danger" });
  if ((current.hydrationRatio ?? 0) < 0.35) tags.push({ label: "Dehydrated", kind: "danger" });
  if ((current.healthRatio ?? 0) < 0.55 || (current.injuryLoad ?? 0) > 0.25) {
    tags.push({ label: "Injured", kind: "warning" });
  }
  if (current.reproductionReady) tags.push({ label: "Ready to reproduce", kind: "good" });
  if (current.lastDamageSource && current.lastDamageSource !== "none") {
    tags.push({ label: damageSourceLabel(current.lastDamageSource), kind: "warning" });
  }
  if (tags.length === 0) tags.push({ label: "Stable", kind: "good" });
  return tags.slice(0, 4);
}

function describeSpeciesTrend(speciesId) {
  if (speciesId == null) return "unknown";
  const series = state.payload?.viewer?.analytics?.species_population?.[String(speciesId)];
  if (!series || series.length === 0) return "unknown";
  const current = series[state.currentFrameIndex] ?? 0;
  const previous = series[Math.max(0, state.currentFrameIndex - 12)] ?? current;
  if (current > previous) return "growing";
  if (current < previous) return "shrinking";
  return "stable";
}

function agentFocusSentence(current, latest, speciesTrend, stressTags, deathEvent) {
  if (!current) {
    const lastSeen = latest ? `Last seen at tick ${latest.tick} near ${latest.x},${latest.y}. ` : "";
    const death = deathEvent ? `It died from ${deathCauseLabel(deathEvent.data?.cause)}.` : "";
    return `${lastSeen}${death}`.trim() || "This agent is not alive on the selected tick.";
  }
  const stress = stressTags.map((tag) => tag.label.toLowerCase()).join(", ");
  return `This ${trophicRoleLabel(current.trophicRole).toLowerCase()} is ${stress}; its species trend is ${speciesTrend}.`;
}

function eventStripMarkerClass(entry, currentTick, cluster) {
  return [
    "event-strip-marker",
    cluster ? "clustered" : "",
    currentTick >= entry.startTick && currentTick <= entry.endTick ? "current" : "",
    entry.endTick < currentTick ? "past" : "",
    entry.hasDeaths ? "death" : "",
    entry.hasBirths ? "birth" : "",
    entry.hasCombat ? "combat" : "",
    entry.hasCarcass ? "carcass" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function aggregateEventStripEntries(entries, maxTick, bucketIndex) {
  const startTick = Math.min(...entries.map((entry) => Number(entry.tick)));
  const endTick = Math.max(...entries.map((entry) => Number(entry.tick)));
  const aggregate = entries.reduce(
    (total, entry) => ({
      tick: total.tick,
      startTick,
      endTick,
      total: total.total + (Number(entry.total) || 0),
      births: total.births + (Number(entry.births) || 0),
      deaths: total.deaths + (Number(entry.deaths) || 0),
      attacks: total.attacks + (Number(entry.attacks) || 0),
      carcasses: total.carcasses + (Number(entry.carcasses) || 0),
      hasBirths: total.hasBirths || Boolean(entry.hasBirths),
      hasDeaths: total.hasDeaths || Boolean(entry.hasDeaths),
      hasCombat: total.hasCombat || Boolean(entry.hasCombat),
      hasCarcass: total.hasCarcass || Boolean(entry.hasCarcass),
    }),
    {
      tick: entries[0]?.tick ?? 0,
      startTick,
      endTick,
      total: 0,
      births: 0,
      deaths: 0,
      attacks: 0,
      carcasses: 0,
      hasBirths: false,
      hasDeaths: false,
      hasCombat: false,
      hasCarcass: false,
    },
  );
  const busiest = entries.reduce((best, entry) => {
    const bestTotal = Number(best.total) || 0;
    const entryTotal = Number(entry.total) || 0;
    return entryTotal > bestTotal ? entry : best;
  }, entries[0]);
  return {
    ...aggregate,
    tick: Number(busiest?.tick ?? aggregate.tick),
    startTick,
    endTick,
    bucketIndex,
    entryCount: entries.length,
    positionTick: (startTick + endTick) / 2,
    percent: maxTick > 0 ? ((startTick + endTick) / 2 / maxTick) * 100 : 0,
    startPercent: maxTick > 0 ? (startTick / maxTick) * 100 : 0,
    endPercent: maxTick > 0 ? (endTick / maxTick) * 100 : 0,
  };
}

function buildEventStripMarkers(eventTicks, maxTick) {
  if (!elements.eventStrip || eventTicks.length === 0) return [];
  const stripWidth = elements.eventStrip.clientWidth || 720;
  const bucketCount = Math.max(EVENT_STRIP_MIN_BUCKETS, Math.floor(stripWidth / EVENT_STRIP_CLUSTER_PX));
  if (eventTicks.length <= bucketCount) {
    return eventTicks.map((entry) => ({
      ...entry,
      startTick: Number(entry.tick),
      endTick: Number(entry.tick),
      entryCount: 1,
      positionTick: Number(entry.tick),
      percent: maxTick > 0 ? (Number(entry.tick) / maxTick) * 100 : 0,
      startPercent: maxTick > 0 ? (Number(entry.tick) / maxTick) * 100 : 0,
      endPercent: maxTick > 0 ? (Number(entry.tick) / maxTick) * 100 : 0,
    }));
  }

  const buckets = new Map();
  for (const entry of eventTicks) {
    const tick = Number(entry.tick);
    const bucketIndex = Math.min(bucketCount - 1, Math.max(0, Math.floor((tick / maxTick) * bucketCount)));
    const bucket = buckets.get(bucketIndex) ?? [];
    bucket.push(entry);
    buckets.set(bucketIndex, bucket);
  }
  return [...buckets.entries()]
    .sort(([left], [right]) => left - right)
    .map(([bucketIndex, entries]) => aggregateEventStripEntries(entries, maxTick, bucketIndex));
}

function eventStripMarkerTitle(entry) {
  if ((entry.entryCount ?? 1) <= 1 || entry.startTick === entry.endTick) {
    return `Tick ${entry.tick}: ${eventTickSummary(entry)}`;
  }
  return `Ticks ${entry.startTick}-${entry.endTick}: ${entry.entryCount} event ticks, ${eventTickSummary(entry)}`;
}

function eventBandKind(entry) {
  if ((entry.attacks ?? 0) > 0 || entry.hasCombat) return "combat";
  if ((entry.deaths ?? 0) > 0 || entry.hasDeaths) return "death";
  if ((entry.births ?? 0) > 0 || entry.hasBirths) return "birth";
  if ((entry.carcasses ?? 0) > 0 || entry.hasCarcass) return "carcass";
  return "event";
}

function eventBandLabel(entry) {
  const kind = eventBandKind(entry);
  const labels = {
    birth: "Births",
    death: "Deaths",
    combat: "Combat",
    carcass: "Carcass",
    event: "Events",
  };
  return labels[kind] ?? labels.event;
}

function eventBandWidth(entry) {
  return Math.max(1.8, (Number(entry.endPercent) || 0) - (Number(entry.startPercent) || 0));
}

function renderStoryTimeline(frame) {
  if (!elements.eventStrip || !elements.eventSummary || !state.payload) return;
  const eventTicks = state.significantEventTicks;
  const maxTick = Math.max(state.payload.summary.ticks_executed - 1, 1);
  if (eventTicks.length === 0) {
    elements.eventSummary.textContent = "No major replay events in this run.";
    elements.eventStrip.innerHTML = "";
    if (elements.eventClusterDrilldown) elements.eventClusterDrilldown.innerHTML = "";
    elements.jumpPrevEvent.disabled = true;
    elements.jumpNextEvent.disabled = true;
    return;
  }

  const currentTick = Number(frame.tick);
  const currentEntry = eventTicks.find((entry) => entry.tick === currentTick);
  const previousEntry = [...eventTicks].reverse().find((entry) => entry.tick < currentTick);
  const nextEntry = eventTicks.find((entry) => entry.tick > currentTick);
  elements.eventSummary.textContent = currentEntry
    ? `Tick ${currentTick}: ${eventTickSummary(currentEntry)}`
    : nextEntry
      ? `Next major event at tick ${nextEntry.tick}: ${eventTickSummary(nextEntry)}`
      : `Last major event was tick ${previousEntry?.tick ?? "-"}: ${eventTickSummary(previousEntry)}`;

  elements.jumpPrevEvent.disabled = !previousEntry;
  elements.jumpNextEvent.disabled = !nextEntry;

  const timelineEntries = buildEventStripMarkers(eventTicks, maxTick);
  state.visualStats.eventBands = timelineEntries.filter((entry) => (entry.entryCount ?? 1) > 1).length;
  const cluster = selectedEventCluster(timelineEntries, currentTick);
  state.activeEventCluster = cluster ? eventClusterDetailModel(cluster, state.significantEventTicks) : null;
  elements.eventStrip.innerHTML = [
    `<span class="event-strip-current" style="left:${(currentTick / maxTick) * 100}%;"></span>`,
    ...timelineEntries.map((entry) => {
      const clustered = (entry.entryCount ?? 1) > 1;
      if (clustered) {
        const kind = eventBandKind(entry);
        const current = currentTick >= entry.startTick && currentTick <= entry.endTick;
        const past = entry.endTick < currentTick;
        const classes = ["event-band", kind, current ? "current" : "", past ? "past" : ""]
          .filter(Boolean)
          .join(" ");
        return `
          <button
            type="button"
            class="${classes}"
            data-event-band="${kind}"
            data-event-tick="${entry.tick}"
            data-event-count="${entry.entryCount}"
            data-event-range="${entry.startTick}-${entry.endTick}"
            style="left:${entry.startPercent}%;width:max(28px, ${eventBandWidth(entry)}%);"
            title="${escapeHtml(eventStripMarkerTitle(entry))}"
          >
            <span class="event-band-label">${escapeHtml(eventBandLabel(entry))}</span>
            <small>${escapeHtml(String(entry.entryCount))}</small>
          </button>
        `;
      }
      const classes = eventStripMarkerClass(entry, currentTick, clustered);
      return `
        <button
          type="button"
          class="${classes}"
          data-event-tick="${entry.tick}"
          ${clustered ? `data-event-count="${entry.entryCount}" data-event-range="${entry.startTick}-${entry.endTick}"` : ""}
          style="left:${entry.percent}%;"
          title="${escapeHtml(eventStripMarkerTitle(entry))}"
        ></button>
      `;
    }),
  ].join("");
  renderEventClusterDrilldown();
}

function selectedEventCluster(timelineEntries, currentTick) {
  const clusters = timelineEntries.filter((entry) => (entry.entryCount ?? 1) > 1);
  if (clusters.length === 0) return null;
  const active = state.activeEventCluster;
  if (active) {
    const matching = clusters.find(
      (entry) =>
        Number(entry.startTick) === Number(active.startTick) &&
        Number(entry.endTick) === Number(active.endTick),
    );
    if (matching) return matching;
  }
  return null;
}

function renderEventClusterDrilldown() {
  if (!elements.eventClusterDrilldown) return;
  const cluster = state.activeEventCluster;
  if (!cluster?.tickCount) {
    elements.eventClusterDrilldown.innerHTML = "";
    return;
  }
  elements.eventClusterDrilldown.innerHTML = `
    <div class="event-cluster-heading">
      <span>Event Cluster</span>
      <strong>${escapeHtml(cluster.rangeLabel)}</strong>
      <small>${escapeHtml(cluster.summary)}</small>
    </div>
    <div class="event-cluster-ticks">
      ${cluster.ticks
        .map(
          (entry) => `
            <button type="button" data-cluster-tick="${escapeHtml(String(entry.tick))}">
              <span>Tick ${escapeHtml(String(entry.tick))}</span>
              <strong>${escapeHtml(entry.label)}</strong>
            </button>
          `,
        )
        .join("")}
    </div>
  `;
}

function eventClusterFromRange(range) {
  const [start, end] = String(range ?? "")
    .split("-")
    .map((part) => Number(part));
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  return eventClusterDetailModel({ startTick: start, endTick: end }, state.significantEventTicks);
}

function renderEpisodeInspector(frame) {
  if (!elements.episodeInspector || !state.payload || !state.replayModel) return;
  const inspectedTick = state.episodePlayback?.episodeTick ?? Number(frame.tick);
  const context = episodeContextForTick(inspectedTick);
  if (!context?.episode) {
    elements.episodeInspector.innerHTML =
      '<div class="muted">No major birth, death, combat, or carcass episodes are present in this replay.</div>';
    return;
  }

  const { episode, relation, distance } = context;
  const episodeTick = Number(episode.tick);
  const episodeFrameIndex = frameIndexForTick(episodeTick) ?? state.currentFrameIndex;
  const episodeFrame = state.payload.viewer.frames[episodeFrameIndex] ?? frame;
  const previousFrame = state.payload.viewer.frames[Math.max(0, episodeFrameIndex - 1)] ?? episodeFrame;
  const events = state.eventsByTick.get(episodeTick) ?? [];
  const counts = eventCategoryCounts(events);
  const deltas = episodeDeltas(episodeFrame, previousFrame, counts);
  const causeEffect = episodeCauseEffect(episode, counts, deltas);
  const agents = episodeInvolvedAgents(events, episodeTick);
  const ledger = episodeLedger(events).slice(0, 7);
  const lensModes = episodeLensModes(counts);
  const playbackActive = state.episodePlayback?.episodeTick === episodeTick;
  const playbackProgress = playbackActive
    ? `${Math.max(1, state.currentFrameIndex - state.episodePlayback.startIndex + 1)}/${Math.max(
        1,
        state.episodePlayback.endIndex - state.episodePlayback.startIndex + 1,
      )}`
    : null;
  const relationLabel =
    relation === "current"
      ? "Current episode"
      : relation === "future"
        ? `Upcoming in ${distance} tick${distance === 1 ? "" : "s"}`
        : `${distance} tick${distance === 1 ? "" : "s"} ago`;

  elements.episodeInspector.innerHTML = `
    <section class="episode-causal-card ${escapeHtml(episode.kind)}">
      <div class="episode-inspector-header">
        <div>
          <span>${escapeHtml(relationLabel)}</span>
          <strong>Tick ${escapeHtml(String(episodeTick))}: ${escapeHtml(eventTickSummary(episode))}</strong>
        </div>
        <div class="episode-header-actions">
          <button type="button" data-episode-play="${escapeHtml(String(episodeTick))}">
            ${playbackActive ? "Stop Causal Playback" : "Play Causal Window"}
          </button>
          ${
            relation === "current"
              ? ""
              : `<button type="button" data-episode-jump="${escapeHtml(String(episodeTick))}">Jump To Episode</button>`
          }
        </div>
      </div>
      ${playbackProgress ? `<div class="episode-playback-status">Playing causal window ${escapeHtml(playbackProgress)}</div>` : ""}
      <p>${escapeHtml(episodeCausalitySentence(counts, deltas))}</p>
      <div class="episode-cause-effect" data-episode-cause-effect>
        <div>
          <span>Cause</span>
          <strong>${escapeHtml(causeEffect.cause)}</strong>
        </div>
        <div>
          <span>Effect</span>
          <strong>${escapeHtml(causeEffect.effect)}</strong>
        </div>
      </div>
      <div class="episode-kpi-grid">
        ${[
          ["Alive Delta", signedValue(deltas.aliveDelta)],
          ["Births", counts.birth],
          ["Deaths", counts.death],
          ["Attacks", counts.combat],
          ["Carrion Meals", counts.meatMeals],
          ["Tile Changes", deltas.tileChanges],
        ]
          .map(
            ([label, value]) => `
              <div>
                <span>${escapeHtml(String(label))}</span>
                <strong>${escapeHtml(String(formatValue(value)))}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
      <div class="episode-lens-actions" aria-label="Episode map lenses">
        ${lensModes
          .map(
            (mode) => `
              <button
                type="button"
                class="${state.eventLensMode === mode ? "selected" : ""}"
                data-episode-lens="${escapeHtml(mode)}"
              >
                ${escapeHtml(eventLensLabel(mode))}
              </button>
            `,
          )
          .join("")}
      </div>
      <div class="episode-subsection">
        <strong>Involved Agents</strong>
        <div class="episode-agent-grid">
          ${
            agents.length
              ? agents
                  .map(
                    (agent) => `
                      <button type="button" class="episode-agent-pill" data-episode-agent="${escapeHtml(String(agent.agentId))}">
                        <span>${escapeHtml(agent.roleLabel)}</span>
                        <strong>Agent ${escapeHtml(String(agent.agentId))}</strong>
                        <small>${escapeHtml(agent.summary)}</small>
                      </button>
                    `,
                  )
                  .join("")
              : '<div class="muted">No agent-level ids were recorded for this episode.</div>'
          }
        </div>
      </div>
      <div class="episode-subsection">
        <strong>Event Ledger</strong>
        <div class="episode-ledger">
          ${ledger
            .map(
              (item) => `
                <div class="episode-ledger-item ${escapeHtml(item.category)}">
                  <span>${escapeHtml(item.title)}</span>
                  <strong>${escapeHtml(item.detail)}</strong>
                </div>
              `,
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function episodeContextForTick(tick) {
  return modelEpisodeContextForTick(state.replayModel?.episodes ?? [], tick);
}

function episodeDeltas(frame, previousFrame, counts) {
  const tileChanges = countChangedTileDeltas(frame, previousFrame);
  return {
    aliveDelta: Number(frame.alive_agents ?? 0) - Number(previousFrame?.alive_agents ?? frame.alive_agents ?? 0),
    speciesDelta:
      Number(frame.species_counts?.length ?? 0) - Number(previousFrame?.species_counts?.length ?? frame.species_counts?.length ?? 0),
    birthDelta: Math.max(0, Number(frame.births ?? 0) - Number(previousFrame?.births ?? frame.births ?? 0)),
    deathDelta: Math.max(0, Number(frame.deaths ?? 0) - Number(previousFrame?.deaths ?? frame.deaths ?? 0)),
    carcassTileDelta:
      Number(frame.carcass_stats?.carcass_tiles ?? 0) -
      Number(previousFrame?.carcass_stats?.carcass_tiles ?? frame.carcass_stats?.carcass_tiles ?? 0),
    tileChanges,
    moves: counts.moves,
  };
}

function episodeCauseEffect(episode, counts, deltas) {
  return modelEpisodeCauseEffect(episode, counts, deltas, {
    formatValue,
    signedValue,
    episodeLabel,
  });
}

function eventLensLabel(mode) {
  const labels = {
    current: "Current Tick",
    causal: "Causal Trail",
    births: "Births",
    deaths: "Deaths",
    combat: "Combat",
    carcass: "Carcass Flow",
    delta: "Tick Delta",
  };
  return labels[mode] ?? titleCase(mode);
}

function episodeInvolvedAgents(events, tick) {
  return episodeAgentRoleEntries(events)
    .map((entry) => {
      const snapshot = agentSnapshotAtTick(entry.agentId, tick);
      const catalog = state.payload?.viewer?.agent_catalog?.[String(entry.agentId)] ?? {};
      const speciesId = snapshot?.speciesId ?? catalog.lineage_id ?? null;
      const position = snapshot ? `${snapshot.x},${snapshot.y}` : "last position unknown";
      const role = snapshot?.trophicRole ? trophicRoleLabel(snapshot.trophicRole) : "historical agent";
      return {
        agentId: entry.agentId,
        roleLabel: entry.roleLabel,
        summary: `${role} · ${speciesId != null ? speciesLabel(speciesId) : UNKNOWN_LABEL} · ${position}`,
      };
    });
}

function episodeLedger(events) {
  return modelEpisodeLedger(events, {
    roundValue,
    reproductionModeLabel,
    deathCauseLabel,
    foodSourceLabel,
    emptyLabel: VIEWER_EMPTY_LABELS.not_applicable,
  });
}

function signedValue(value) {
  const numeric = Number(value) || 0;
  return numeric > 0 ? `+${formatValue(numeric)}` : formatValue(numeric);
}

function setEventLensMode(mode) {
  const nextMode = String(mode ?? "current");
  if (!["current", "causal", "births", "deaths", "combat", "carcass", "delta"].includes(nextMode)) {
    return;
  }
  state.eventLensMode = nextMode;
  if (elements.eventLensMode) elements.eventLensMode.value = nextMode;
  persistViewerPreferences();
  renderFrame(state.currentFrameIndex);
}

function focusEpisodeAgent(agentId) {
  if (!Number.isFinite(agentId)) return;
  const snapshot = agentSnapshotAtTick(agentId, state.payload?.viewer?.frames?.[state.currentFrameIndex]?.tick ?? 0);
  state.selectedAgentId = agentId;
  state.selectedSpeciesId = snapshot?.speciesId ?? state.selectedSpeciesId;
  setActiveDetailView("agent");
  renderFrame(state.currentFrameIndex);
}

function toggleEpisodePlayback(episodeTick) {
  if (!Number.isFinite(episodeTick) || !state.payload) return;
  if (state.episodePlayback?.episodeTick === episodeTick) {
    stopEpisodePlayback(true);
    return;
  }
  startEpisodePlayback(episodeTick);
}

function startEpisodePlayback(episodeTick) {
  if (!state.payload) return;
  stopPlayback({ stopEpisode: false });
  stopEpisodePlayback(false);
  state.eventLensMode = "causal";
  if (elements.eventLensMode) elements.eventLensMode.value = "causal";

  const frames = state.payload.viewer.frames;
  const maxTick = Number(frames[frames.length - 1]?.tick ?? episodeTick);
  const startTick = Math.max(0, episodeTick - EPISODE_PLAYBACK_WINDOW);
  const endTick = Math.min(maxTick, episodeTick + EPISODE_PLAYBACK_WINDOW);
  const startIndex = frameIndexForTick(startTick) ?? 0;
  const endIndex = frameIndexForTick(endTick) ?? frames.length - 1;
  state.episodePlayback = {
    episodeTick,
    startTick,
    endTick,
    startIndex,
    endIndex,
    scope: episodePlaybackScope(episodeTick),
  };
  renderFrame(startIndex);
  scheduleEpisodePlaybackStep();
}

function scheduleEpisodePlaybackStep() {
  if (!state.episodePlayback || !state.payload) return;
  if (state.episodeTimerId != null) {
    window.clearTimeout(state.episodeTimerId);
  }
  state.episodeTimerId = window.setTimeout(() => {
    if (!state.episodePlayback || !state.payload) return;
    if (state.currentFrameIndex >= state.episodePlayback.endIndex) {
      stopEpisodePlayback(true);
      return;
    }
    renderFrame(state.currentFrameIndex + 1);
    scheduleEpisodePlaybackStep();
  }, EPISODE_PLAYBACK_INTERVAL_MS);
}

function stopEpisodePlayback(render = false) {
  const hadPlayback = Boolean(state.episodePlayback);
  state.episodePlayback = null;
  if (state.episodeTimerId != null) {
    window.clearTimeout(state.episodeTimerId);
    state.episodeTimerId = null;
  }
  if (render && hadPlayback && state.payload) {
    renderFrame(state.currentFrameIndex);
  }
}

function renderEventEpisodes(frame) {
  if (!elements.episodeList || !state.replayModel) return;
  const episodes = state.replayModel.episodes;
  if (episodes.length === 0) {
    elements.episodeList.innerHTML =
      '<div class="muted">No birth, death, or carcass episodes are present in this replay.</div>';
    if (elements.episodeFilterSummary) elements.episodeFilterSummary.textContent = "";
    return;
  }

  const currentTick = Number(frame?.tick ?? 0);
  const filter = episodeFilterModel(episodes, {
    currentTick,
    query: state.episodeFilters.query,
    kind: state.episodeFilters.kind,
    agentId: state.episodeFilters.selectedAgentOnly ? state.selectedAgentId : null,
    eventAgentIdsByTick: eventAgentIdsByTick(),
    limit: 10,
  });
  if (elements.episodeFilterSummary) {
    elements.episodeFilterSummary.textContent = filter.summary;
  }
  if (filter.visibleEpisodes.length === 0) {
    elements.episodeList.innerHTML = '<div class="muted">No episodes match the current filters.</div>';
    return;
  }
  elements.episodeList.innerHTML = filter.visibleEpisodes
    .map((episode) => {
      const selected = episode.tick === currentTick;
      const relative = episode.relative;
      const frameIndex = frameIndexForTick(episode.tick) ?? state.currentFrameIndex;
      const episodeFrame = state.payload.viewer.frames[frameIndex] ?? frame;
      const previousFrame = state.payload.viewer.frames[Math.max(0, frameIndex - 1)] ?? episodeFrame;
      const events = state.eventsByTick.get(Number(episode.tick)) ?? [];
      const counts = eventCategoryCounts(events);
      const causeEffect = episodeCauseEffect(episode, counts, episodeDeltas(episodeFrame, previousFrame, counts));
      return `
        <button
          type="button"
          class="episode-item ${escapeHtml(episode.kind)} ${escapeHtml(relative)}${selected ? " selected" : ""}"
          data-episode-tick="${episode.tick}"
          data-episode-storyline="${escapeHtml(episode.kind)}"
        >
          <span>Tick ${escapeHtml(String(episode.tick))}</span>
          <strong>${escapeHtml(eventTickSummary(episode))}</strong>
          <small>${escapeHtml(episodeLabel(episode))}</small>
          <em class="episode-storyline">
            <b>Cause</b> ${escapeHtml(causeEffect.cause)}
            <b>Effect</b> ${escapeHtml(causeEffect.effect)}
          </em>
        </button>
      `;
    })
    .join("");
}

function eventAgentIdsByTick() {
  const byTick = new Map();
  for (const [tick, events] of state.eventsByTick.entries()) {
    const ids = new Set();
    for (const entry of episodeAgentRoleEntries(events, { limit: 1000 })) {
      ids.add(entry.agentId);
    }
    byTick.set(Number(tick), ids);
  }
  return byTick;
}

function renderEventBookmarks(frame) {
  if (!elements.bookmarkList || !elements.bookmarkCurrentEvent || !state.payload) return;
  const currentTick = Number(frame.tick);
  const bookmarked = state.eventBookmarks.some((entry) => entry.tick === currentTick);
  if (elements.exportStoryboard) {
    const hasEntries = state.eventBookmarks.length > 0 || (state.replayModel?.episodes?.length ?? 0) > 0;
    elements.exportStoryboard.disabled = !hasEntries;
  }
  elements.bookmarkCurrentEvent.textContent = bookmarked
    ? "Remove Tick Bookmark"
    : "Bookmark Current Tick";
  elements.bookmarkCurrentEvent.classList.toggle("selected", bookmarked);

  if (state.eventBookmarks.length === 0) {
    elements.bookmarkList.innerHTML = '<div class="muted">Bookmarked episodes will appear here.</div>';
    renderStoryboardPreview(frame);
    return;
  }

  elements.bookmarkList.innerHTML = state.eventBookmarks
    .map((entry) => {
      const selected = entry.tick === currentTick;
      return `
        <div class="bookmark-item${selected ? " selected" : ""}">
          <button type="button" data-bookmark-tick="${escapeHtml(String(entry.tick))}">
            <span>Tick ${escapeHtml(String(entry.tick))}</span>
            <strong>${escapeHtml(entry.summary)}</strong>
          </button>
          <button
            type="button"
            class="bookmark-remove"
            data-bookmark-remove="${escapeHtml(String(entry.tick))}"
            aria-label="Remove bookmark for tick ${escapeHtml(String(entry.tick))}"
          >Remove</button>
        </div>
      `;
    })
    .join("");
  renderStoryboardPreview(frame);
}

function renderStoryboardPreview(frame) {
  if (!elements.storyboardPreview || !state.payload) return;
  const entries = storyboardPreviewEntries(Number(frame.tick));
  if (entries.length === 0) {
    elements.storyboardPreview.innerHTML =
      '<div class="muted">Storyboard previews will appear when event episodes load.</div>';
    return;
  }
  elements.storyboardPreview.innerHTML = `
    <div class="storyboard-preview-header">
      <span>Storyboard Preview</span>
      <strong>${escapeHtml(String(entries.length))} frame${entries.length === 1 ? "" : "s"} ready</strong>
    </div>
    <div class="storyboard-preview-grid">
      ${entries
        .map(
          (entry) => `
            <button type="button" data-storyboard-preview-tick="${escapeHtml(String(entry.tick))}">
              <img alt="Storyboard map tick ${escapeHtml(String(entry.tick))}" src="${entry.image}" />
              <span>Tick ${escapeHtml(String(entry.tick))}</span>
              <strong>${escapeHtml(entry.summary)}</strong>
            </button>
          `,
        )
        .join("")}
    </div>
  `;
}

function storyboardPreviewEntries(currentTick) {
  const sourceEntries =
    state.eventBookmarks.length > 0
      ? state.eventBookmarks.map((entry) => ({ tick: entry.tick, summary: entry.summary }))
      : nearestStoryboardEpisodes(currentTick);
  return sourceEntries
    .map((entry) => {
      const frameIndex = frameIndexForTick(entry.tick) ?? 0;
      const image = mapSnapshotDataUrl(frameIndex, { maxPixels: 160 });
      if (!image) return null;
      return {
        tick: Number(entry.tick),
        summary: entry.summary ?? eventBookmarkSummary(entry.tick),
        image,
      };
    })
    .filter(Boolean)
    .slice(0, 4);
}

function nearestStoryboardEpisodes(currentTick) {
  const episodes = state.replayModel?.episodes ?? [];
  if (episodes.length === 0) return [];
  const ordered = [...episodes].sort(
    (left, right) => Math.abs(Number(left.tick) - currentTick) - Math.abs(Number(right.tick) - currentTick),
  );
  return ordered.slice(0, 4).map((episode) => ({
    tick: Number(episode.tick),
    summary: eventTickSummary(episode),
  }));
}

function toggleCurrentEventBookmark() {
  if (!state.payload) return;
  const tick = Number(state.payload.viewer.frames[state.currentFrameIndex]?.tick ?? 0);
  if (state.eventBookmarks.some((entry) => entry.tick === tick)) {
    removeEventBookmark(tick);
    return;
  }
  const entry = {
    tick,
    summary: eventBookmarkSummary(tick),
    createdAt: new Date().toISOString(),
  };
  state.eventBookmarks = [...state.eventBookmarks, entry].sort((left, right) => left.tick - right.tick);
  saveEventBookmarks();
  renderFrame(state.currentFrameIndex);
}

function removeEventBookmark(tick) {
  state.eventBookmarks = state.eventBookmarks.filter((entry) => entry.tick !== tick);
  saveEventBookmarks();
  renderFrame(state.currentFrameIndex);
}

function exportStoryboard() {
  if (!state.payload) return;
  const payload = buildStoryboardExport();
  const text = JSON.stringify(payload, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${safeFilePart(payload.run_id)}-storyboard.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  state.lastStoryboardExport = {
    source: payload.source,
    entryCount: payload.entries.length,
    pngFrameCount: payload.entries.filter((entry) => typeof entry.map_png_data_url === "string").length,
    previewCount: elements.storyboardPreview?.querySelectorAll("[data-storyboard-preview-tick] img").length ?? 0,
    runId: payload.run_id,
  };
  if (elements.storyboardExportStatus) {
    elements.storyboardExportStatus.textContent = `Exported ${payload.entries.length} storyboard ${
      payload.entries.length === 1 ? "entry" : "entries"
    } with ${state.lastStoryboardExport.pngFrameCount} PNG map frame${
      state.lastStoryboardExport.pngFrameCount === 1 ? "" : "s"
    }.`;
  }
  syncDebugState();
}

function buildStoryboardExport() {
  const runId = String(state.payload?.summary?.run_id ?? state.payload?.run_id ?? state.replaySource ?? "unknown-run");
  return buildStoryboardExportModel({
    runId,
    replaySource: state.replaySource,
    eventBookmarks: state.eventBookmarks,
    episodes: state.replayModel?.episodes ?? [],
    entryForTick: (entry) => storyboardEntryForTick(entry.tick, entry),
  });
}

function storyboardEntryForTick(tick, sourceEntry) {
  const numericTick = Number(tick);
  const frameIndex = frameIndexForTick(numericTick) ?? 0;
  const frame = state.payload.viewer.frames[frameIndex];
  const previousFrame = state.payload.viewer.frames[Math.max(0, frameIndex - 1)] ?? frame;
  const episode = (state.replayModel?.episodes ?? []).find((candidate) => Number(candidate.tick) === numericTick) ?? null;
  const events = state.eventsByTick.get(numericTick) ?? [];
  const counts = eventCategoryCounts(events);
  const deltas = episodeDeltas(frame, previousFrame, counts);
  const agents = episodeInvolvedAgents(events, numericTick).map((agent) => ({
    agent_id: agent.agentId,
    role: agent.roleLabel,
    summary: agent.summary,
  }));
  return {
    tick: numericTick,
    frame_index: frameIndex,
    note: sourceEntry.note,
    bookmarked_at: sourceEntry.bookmarked_at,
    episode_kind: episode?.kind ?? "tick",
    summary: episode ? eventTickSummary(episode) : eventBookmarkSummary(numericTick),
    causal_sentence: episodeCausalitySentence(counts, deltas),
    counts,
    deltas,
    involved_agents: agents,
    event_ledger: episodeLedger(events).slice(0, 10),
    map_png_data_url: storyboardMapPngDataUrl(frameIndex),
  };
}

function storyboardMapPngDataUrl(frameIndex) {
  return mapSnapshotDataUrl(frameIndex, { maxPixels: 300 });
}

function mapSnapshotDataUrl(frameIndex, options = {}) {
  const payload = state.payload;
  const frame = payload?.viewer?.frames?.[frameIndex];
  const map = payload?.viewer?.map;
  if (!payload || !frame || !map) return null;
  const maxPixels = Number(options.maxPixels ?? 300);
  const baselineIndex = options.baselineIndex == null ? null : Number(options.baselineIndex);
  const delta = Boolean(options.delta && baselineIndex != null);
  const key = snapshotCacheKey({
    frameIndex,
    baselineIndex,
    overlayMode: state.overlayMode,
    overlayOpacity: state.overlayOpacity,
    blendTerrain: state.blendTerrain,
    maxPixels,
    delta,
  });
  const cached = state.snapshotCache.get(key);
  if (cached) {
    state.snapshotStats.hits += 1;
    state.snapshotStats.lastBudget = buildSnapshotBudgetReport({
      durationMs: 0,
      budgetMs: DEFAULT_SNAPSHOT_BUDGET_MS,
      cacheHit: true,
    });
    return cached;
  }
  const startedAt = performance.now();
  const scale = snapshotScaleForMap(map, maxPixels);
  const canvas = document.createElement("canvas");
  canvas.width = map.width * scale;
  canvas.height = map.height * scale;
  const context = canvas.getContext("2d");
  if (!context) return null;

  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const code = map.terrain_codes?.[y]?.[x] ?? 0;
      const overlay = state.overlayMode === "terrain" ? terrainColor(code) : overlayColorForTile(payload, frame, x, y, code);
      context.fillStyle = `#${overlay.toString(16).padStart(6, "0")}`;
      context.fillRect(x * scale, y * scale, scale, scale);
    }
  }

  const agents = state.replayModel?.decodedFrames?.[frameIndex]?.agents ?? [];
  for (const agent of agents) {
    context.fillStyle = `#${colorForSpecies(agent.speciesId).toString(16).padStart(6, "0")}`;
    context.beginPath();
    context.arc((agent.x + 0.5) * scale, (agent.y + 0.5) * scale, Math.max(1.2, scale * 0.42), 0, Math.PI * 2);
    context.fill();
  }

  if (delta) {
    const baseline = payload.viewer.frames[baselineIndex];
    const changes = changedTileCoordinates({
      frame,
      baseline,
      map,
      maxChanges: Math.max(128, Math.floor((map.width * map.height) / 2)),
    });
    context.lineWidth = Math.max(1, scale * 0.18);
    for (const change of changes) {
      context.fillStyle = "rgba(255, 214, 92, 0.5)";
      context.fillRect(change.x * scale, change.y * scale, scale, scale);
      context.strokeStyle = "rgba(255, 255, 255, 0.82)";
      context.strokeRect(change.x * scale + 0.5, change.y * scale + 0.5, scale - 1, scale - 1);
    }
  }

  const dataUrl = canvas.toDataURL("image/png");
  state.snapshotStats.renders += 1;
  state.snapshotStats.lastBudget = buildSnapshotBudgetReport({
    durationMs: performance.now() - startedAt,
    budgetMs: DEFAULT_SNAPSHOT_BUDGET_MS,
    cacheHit: false,
  });
  state.snapshotCache.set(key, dataUrl);
  trimSnapshotCache(state.snapshotCache, DEFAULT_SNAPSHOT_CACHE_LIMIT);
  return dataUrl;
}

function safeFilePart(value) {
  return String(value ?? "storyboard")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "storyboard";
}

function eventBookmarkSummary(tick) {
  return modelEventBookmarkSummary(
    tick,
    state.significantEventTicks,
    state.eventsByTick.get(Number(tick)) ?? [],
  );
}

function loadEventBookmarks() {
  try {
    const raw = window.localStorage?.getItem(eventBookmarkStorageKey());
    const parsed = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((entry) => ({
        tick: Number(entry.tick),
        summary: String(entry.summary ?? eventBookmarkSummary(Number(entry.tick))),
        createdAt: String(entry.createdAt ?? ""),
      }))
      .filter((entry) => Number.isFinite(entry.tick))
      .sort((left, right) => left.tick - right.tick);
  } catch {
    return [];
  }
}

function saveEventBookmarks() {
  try {
    window.localStorage?.setItem(eventBookmarkStorageKey(), JSON.stringify(state.eventBookmarks));
  } catch {
    // Bookmarks are a local viewer affordance; storage failures should not block replay inspection.
  }
}

function eventBookmarkStorageKey() {
  const runId = state.payload?.summary?.run_id ?? state.payload?.run_id ?? state.replaySource ?? "unknown";
  return `evolution-sim:event-bookmarks:${String(runId)}`;
}

function jumpToAdjacentEvent(direction) {
  if (!state.payload || state.significantEventTicks.length === 0) return;
  const currentTick = state.payload.viewer.frames[state.currentFrameIndex]?.tick ?? 0;
  const target =
    direction < 0
      ? [...state.significantEventTicks].reverse().find((entry) => entry.tick < currentTick)
      : state.significantEventTicks.find((entry) => entry.tick > currentTick);
  if (!target) return;
  jumpToTick(target.tick);
}

function jumpToTick(tick) {
  const index = frameIndexForTick(tick);
  if (index == null) return;
  stopPlayback();
  renderFrame(index);
}

function frameIndexForTick(tick) {
  const frames = state.payload?.viewer?.frames ?? [];
  if (frames.length === 0) return null;
  const exactIndex = state.replayModel?.frameIndexByTick?.get(Number(tick));
  if (exactIndex != null) return exactIndex;
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  frames.forEach((frame, index) => {
    const distance = Math.abs(Number(frame.tick) - tick);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function countAgentRoles(decodedAgents) {
  const counts = { herbivore: 0, omnivore: 0, carnivore: 0 };
  for (const agent of decodedAgents) {
    counts[agent.trophicRole] = (counts[agent.trophicRole] ?? 0) + 1;
  }
  return counts;
}

function dominantCountEntry(counts) {
  return Object.entries(counts ?? {}).reduce(
    (best, [key, value]) => {
      const count = Number(value) || 0;
      if (!best || count > best.value) {
        return { key, value: count };
      }
      return best;
    },
    null,
  );
}

function describePrimaryPressure(frame, hazardShare, carcassTiles) {
  if (hazardShare >= 0.45) {
    return "The clearest pressure is environmental hazard.";
  }
  if ((frame.ecology_state_counts?.depleted ?? 0) > 0) {
    return "The clearest pressure is depleted vegetation.";
  }
  if (carcassTiles > 0) {
    return "Carrion is present, so scavengers have visible food opportunities.";
  }
  if ((frame.reproduction_stats?.ready_agents ?? 0) > 0) {
    return "Some agents are ready to reproduce.";
  }
  return "No single pressure dominates this frame.";
}

function describeWorld(frame, hazardShare, hardWaterTiles) {
  const hazardText =
    hazardShare > 0
      ? `${formatPercent(hazardShare)} of the map is hazardous`
      : "hazards are quiet";
  const habitat = frame.habitat_state_counts ?? {};
  const habitatText = [
    habitat.bloom ? `${habitat.bloom} bloom tiles` : null,
    habitat.flooded ? `${habitat.flooded} flooded tiles` : null,
    habitat.parched ? `${habitat.parched} parched tiles` : null,
  ]
    .filter(Boolean)
    .join(", ");
  return `${titleCase(frame.season ?? "unknown")} season, ${
    frame.field_state?.disturbance_type ?? "no disturbance"
  }. ${hazardText}; ${hardWaterTiles} tiles have direct water access${
    habitatText ? `; ${habitatText}` : ""
  }.`;
}

function describePopulation(frame, birthDelta, deathDelta, topSpecies) {
  const movement =
    birthDelta || deathDelta
      ? `${birthDelta} births and ${deathDelta} deaths happened since the previous tick`
      : "no births or deaths happened since the previous tick";
  const topSpeciesText = topSpecies
    ? ` The largest live species is ${speciesLabel(topSpecies[0])} with ${topSpecies[1]} agents.`
    : "";
  return `${movement}. ${frame.alive_agents} agents are alive now.${topSpeciesText}`;
}

function describeFoodWeb(roleCounts, carcassTiles) {
  const herbivore = roleCounts.herbivore ?? 0;
  const omnivore = roleCounts.omnivore ?? 0;
  const carnivore = roleCounts.carnivore ?? 0;
  const carcassText =
    carcassTiles > 0
      ? `${carcassTiles} carcass tiles are available`
      : "there are no visible carcass tiles";
  return `${herbivore} herbivores, ${omnivore} omnivores, and ${carnivore} carnivores are alive; ${carcassText}.`;
}

function describeTickEvents(frame, birthDelta, deathDelta, eventCounts = {}) {
  const signalFlow = frame.signal_flow ?? {};
  const carcassFlow = frame.carcass_flow ?? {};
  const events = [];
  if (birthDelta > 0) events.push(`${birthDelta} new birth${birthDelta === 1 ? "" : "s"}`);
  if (deathDelta > 0) events.push(`${deathDelta} death${deathDelta === 1 ? "" : "s"}`);
  if ((eventCounts.moves ?? 0) > 0) {
    events.push(`${eventCounts.moves} movement${eventCounts.moves === 1 ? "" : "s"}`);
  }
  if ((eventCounts.attacks ?? 0) > 0) {
    events.push(`${eventCounts.attacks} attack${eventCounts.attacks === 1 ? "" : "s"}`);
  }
  if ((eventCounts.meatMeals ?? 0) > 0) {
    events.push(`${eventCounts.meatMeals} meat meal${eventCounts.meatMeals === 1 ? "" : "s"}`);
  }
  if ((signalFlow.reproductive_emissions ?? 0) > 0) {
    events.push(`${signalFlow.reproductive_emissions} reproductive signal emissions`);
  }
  if ((carcassFlow.deposition_events ?? 0) > 0) {
    events.push(`${carcassFlow.deposition_events} carcass deposits`);
  }
  if ((carcassFlow.consumption_events ?? 0) > 0) {
    events.push(`${carcassFlow.consumption_events} carrion meals`);
  }
  return events.length ? `${titleCaseList(events)}.` : "Nothing dramatic changed on this tick.";
}

function eventCountsForTick(tick) {
  const counts = {
    moves: 0,
    attacks: 0,
    meatMeals: 0,
  };
  for (const event of state.eventsByTick.get(Number(tick)) ?? []) {
    if (event.type === "agent_moved") {
      counts.moves += 1;
    } else if (event.type === "agent_attacked") {
      counts.attacks += 1;
    } else if (
      event.type === "agent_ate" &&
      ["carcass", "fresh_kill"].includes(String(event.data?.food_source ?? ""))
    ) {
      counts.meatMeals += 1;
    }
  }
  return counts;
}

function describeSelectedFocus(frame, decodedAgents) {
  if (state.selectedAgentId != null) {
    const agent = decodedAgents.find((candidate) => candidate.agentId === state.selectedAgentId);
    if (!agent) {
      return {
        kind: "focus",
        title: "Selected Agent",
        text: `Agent ${state.selectedAgentId} is not alive on this tick.`,
      };
    }
    return {
      kind: "focus",
      title: "Selected Agent",
      text: `Agent ${agent.agentId} is a ${agent.trophicRole} in ${speciesLabel(
        agent.speciesId,
      )}, with ${formatPercent(agent.energyRatio)} energy and ${formatPercent(
        agent.hydrationRatio,
      )} hydration.`,
    };
  }
  if (state.selectedSpeciesId != null) {
    const count =
      frame.species_counts.find(([speciesId]) => speciesId === state.selectedSpeciesId)?.[1] ?? 0;
    return {
      kind: "focus",
      title: "Selected Species",
      text: `${speciesLabel(state.selectedSpeciesId)} has ${count} live agents on this tick.`,
    };
  }
  return null;
}

function renderFocusCards(frame, decodedAgents, speciesCounts) {
  if (!elements.focusCards) return;
  const topSpeciesCards = speciesCounts.slice(0, 4).map(([speciesId, count]) => {
    const color = `#${colorForSpecies(speciesId).toString(16).padStart(6, "0")}`;
    return `
      <button
        type="button"
        class="focus-card${state.selectedSpeciesId === speciesId ? " selected" : ""}"
        data-story-species-id="${speciesId}"
      >
        <i style="background:${color};"></i>
        <span>${escapeHtml(speciesLabel(speciesId))}</span>
        <strong>${count}</strong>
      </button>
    `;
  });
  const selectedAgent = decodedAgents.find(
    (agent) => agent.agentId === state.selectedAgentId,
  );
  const selectedCard = selectedAgent
    ? `
      <div class="focus-note">
        Following agent ${selectedAgent.agentId} at ${selectedAgent.x},${selectedAgent.y}.
      </div>
    `
    : `<div class="focus-note">Click an agent on the map or a species below to follow it.</div>`;

  elements.focusCards.innerHTML = `
    ${selectedCard}
    <div class="focus-card-grid">${topSpeciesCards.join("")}</div>
  `;
  [...elements.focusCards.querySelectorAll("[data-story-species-id]")].forEach((button) => {
    button.addEventListener("click", () => {
      const speciesId = Number(button.dataset.storySpeciesId);
      const match = decodedAgents.find((agent) => agent.speciesId === speciesId);
      state.selectedSpeciesId = speciesId;
      state.selectedAgentId = match?.agentId ?? null;
      setActiveDetailView(match ? "agent" : "species");
      renderFrame(state.currentFrameIndex);
    });
  });
}

function renderLineageTree(frame) {
  if (!elements.lineageTree || !state.payload) return;
  state.visualStats.lineageBranches = 0;
  const catalog = state.payload.viewer.species_catalog ?? {};
  const taxonomy = state.payload.viewer.taxonomy ?? {};
  const records = lineageDisplayRecords(
    Object.values(catalog).map((record) => ({
      ...record,
      species_id: Number(record.species_id),
      aliveNow:
        frame.species_counts?.find(
          ([speciesId]) => speciesId === Number(record.species_id),
        )?.[1] ?? 0,
    })),
  );
  if (records.length === 0) {
    elements.lineageTree.innerHTML = '<div class="muted">Replay taxonomy is not available for this run.</div>';
    return;
  }
  const statusCounts = taxonomy.species_status_counts ?? {};
  const extant = statusCounts.extant ?? records.filter((record) => record.status === "extant").length;
  const extinct = statusCounts.extinct ?? records.filter((record) => record.status === "extinct").length;
  elements.lineageTree.innerHTML = `
    <div class="lineage-summary">
      <div>
        <strong>Lineage Tree</strong>
        <span>Replay Taxonomy · ${escapeHtml(taxonomy.version ?? UNKNOWN_LABEL)}</span>
      </div>
      <dl>
        <div><dt>Extant</dt><dd>${escapeHtml(formatValue(extant))}</dd></div>
        <div><dt>Extinct</dt><dd>${escapeHtml(formatValue(extinct))}</dd></div>
      </dl>
    </div>
    <div class="lineage-list">
      ${records
        .map((record) => {
          const selected = state.selectedSpeciesId === record.species_id;
          const parent = record.parent_species_id != null ? speciesCode(record.parent_species_id) : "Founder";
          const children = Array.isArray(record.child_species_ids) ? record.child_species_ids.length : 0;
          const color = `#${colorForSpecies(record.species_id).toString(16).padStart(6, "0")}`;
          return `
            <button
              type="button"
              class="lineage-item${selected ? " selected" : ""}${record.depth === 0 ? " founder" : ""}"
              data-lineage-species-id="${escapeHtml(String(record.species_id))}"
              data-lineage-depth="${escapeHtml(String(record.depth))}"
              ${record.parent_species_id != null ? `data-lineage-parent-id="${escapeHtml(String(record.parent_species_id))}"` : ""}
              style="--lineage-depth:${escapeHtml(String(Math.min(record.depth, 8)))};"
            >
              <i class="lineage-swatch" style="background:${color};"></i>
              <span>
                <strong>${escapeHtml(speciesLabel(record.species_id))}</strong>
                <small>${escapeHtml(speciesCode(record.species_id))} · ${escapeHtml(parent)} · ${children} children</small>
              </span>
              <em class="lineage-status ${escapeHtml(record.status ?? "unknown")}">
                ${escapeHtml(titleCase(record.status ?? "unknown"))}
              </em>
              <b>${escapeHtml(formatValue(record.aliveNow))} now</b>
              <small>Peak ${escapeHtml(formatValue(record.peak_members))} · observed ${escapeHtml(formatValue(record.observed_ticks))}</small>
            </button>
          `;
        })
        .join("")}
    </div>
  `;
}

function lineageDisplayRecords(records) {
  const recordById = new Map(records.map((record) => [record.species_id, record]));
  const childrenByParent = new Map();
  const roots = [];
  for (const record of records) {
    const parentId = record.parent_species_id == null ? null : Number(record.parent_species_id);
    if (parentId == null || !recordById.has(parentId)) {
      roots.push(record);
    } else {
      if (!childrenByParent.has(parentId)) {
        childrenByParent.set(parentId, []);
      }
      childrenByParent.get(parentId).push(record);
    }
  }
  const sortRecords = (left, right) => {
    if ((left.status === "extant") !== (right.status === "extant")) {
      return left.status === "extant" ? -1 : 1;
    }
    return (left.start_tick ?? 0) - (right.start_tick ?? 0) || left.species_id - right.species_id;
  };
  roots.sort(sortRecords);
  for (const childRecords of childrenByParent.values()) {
    childRecords.sort(sortRecords);
  }

  const output = [];
  const visited = new Set();
  const visit = (record, depth) => {
    if (visited.has(record.species_id)) return;
    visited.add(record.species_id);
    output.push({ ...record, depth });
    for (const child of childrenByParent.get(record.species_id) ?? []) {
      visit(child, depth + 1);
    }
  };
  roots.forEach((record) => visit(record, 0));
  records
    .filter((record) => !visited.has(record.species_id))
    .sort(sortRecords)
    .forEach((record) => visit(record, 0));
  state.visualStats.lineageBranches = output.filter((record) => record.depth > 0).length;
  return output;
}

function speciesLabel(speciesId) {
  return speciesDisplayName(speciesId);
}

function speciesCode(speciesId) {
  const record = state.payload?.viewer?.species_catalog?.[String(speciesId)];
  const raw = record?.label;
  if (typeof raw === "string" && raw.trim()) return raw;
  const numeric = Number(speciesId);
  return Number.isFinite(numeric) ? `S${String(numeric).padStart(3, "0")}` : `S${speciesId}`;
}

function speciesDisplayName(speciesId) {
  const numeric = Math.abs(Number(speciesId) || 0);
  const profile = speciesNameProfile(speciesId);
  const habitatParts = SPECIES_NAME_PARTS[profile.habitat] ?? SPECIES_NAME_PARTS.plain;
  const adjectivePool = habitatParts.adjectives;
  const nounPool = habitatParts[profile.strategy] ?? habitatParts.forager;
  const adjective = adjectivePool[numeric % adjectivePool.length];
  const noun = nounPool[Math.floor(numeric / adjectivePool.length) % nounPool.length];
  return `${adjective} ${noun}`;
}

function speciesNameProfile(speciesId) {
  const record = state.payload?.viewer?.species_catalog?.[String(speciesId)];
  const centroid = record?.normalized_centroid ?? record?.centroid ?? {};
  return {
    habitat: dominantSpeciesHabitat(centroid, speciesId),
    strategy: dominantSpeciesStrategy(centroid),
  };
}

function dominantSpeciesHabitat(centroid, speciesId) {
  const scores = {
    plain: Number(centroid.plain_affinity),
    forest: Number(centroid.forest_affinity),
    wetland: Number(centroid.wetland_affinity),
    rocky: Number(centroid.rocky_affinity),
  };
  const ranked = Object.entries(scores).filter(([, value]) => Number.isFinite(value));
  if (ranked.length > 0) {
    ranked.sort((left, right) => right[1] - left[1]);
    return ranked[0][0];
  }
  const habitats = Object.keys(SPECIES_NAME_PARTS);
  return habitats[Math.abs(Number(speciesId) || 0) % habitats.length];
}

function dominantSpeciesStrategy(centroid) {
  const plant = Number(centroid.plant_bias ?? 0);
  const carrion = Number(centroid.carrion_bias ?? 0);
  const prey = Number(centroid.live_prey_bias ?? 0);
  const meat = Number(centroid.meat_efficiency ?? 0);
  const attack = Number(centroid.attack_power ?? 0);
  if (prey + attack + meat > plant + carrion + 0.32) return "predator";
  if (carrion + meat > plant * 0.72 && carrion > 0.12) return "scavenger";
  if (plant >= prey + carrion + meat) return "grazer";
  return "forager";
}

function titleCaseList(items) {
  if (items.length === 0) return "";
  return `${items[0][0].toUpperCase()}${items[0].slice(1)}${items
    .slice(1)
    .map((item) => `, ${item}`)
    .join("")}`;
}

function renderClimateState(frame) {
  const fieldState = frame.field_state ?? {};
  [...elements.climateGrid.querySelectorAll("[data-climate-field]")].forEach((node) => {
    const field = node.dataset.climateField;
    node.textContent = formatClimateField(fieldState[field]);
  });
}

function renderHydrologyState(frame) {
  const hydrologyPrimaryCounts = frame.hydrology_primary_counts ?? {};
  const hydrologyPrimaryStats = frame.hydrology_primary_stats ?? {};
  const hydrologySupportCounts = frame.hydrology_support_counts ?? {};
  const refugeCounts = frame.refuge_counts ?? {};
  const refugeStats = frame.refuge_stats ?? {};
  [...elements.hydrologyPrimaryGrid.querySelectorAll("[data-hydrology-primary-field]")].forEach(
    (node) => {
      const field = node.dataset.hydrologyPrimaryField;
      const value =
        field === "hard_access_tiles"
          ? hydrologyPrimaryStats[field]
          : hydrologyPrimaryCounts[field] ?? 0;
      node.textContent = formatValue(value, typeof value === "number");
    },
  );
  [...elements.hydrologySupportGrid.querySelectorAll("[data-hydrology-support-field]")].forEach(
    (node) => {
      const field = node.dataset.hydrologySupportField;
      let value = hydrologySupportCounts[field] ?? 0;
      if (field === "canopy_refuge_tiles") {
        value = refugeCounts.canopy_refuge ?? 0;
      } else if (field === "avg_refuge_score_forest_tiles") {
        value = refugeStats[field];
      }
      node.textContent = formatValue(value, typeof value === "number");
    },
  );
}

function renderHazardState(frame) {
  const hazardCounts = frame.hazard_counts ?? {};
  const hazardStats = frame.hazard_stats ?? {};
  [...elements.hazardGrid.querySelectorAll("[data-hazard-field]")].forEach((node) => {
    const field = node.dataset.hazardField;
    const value =
      field === "hazardous_tiles" || field === "avg_hazard_level"
        ? hazardStats[field]
        : hazardCounts[field] ?? 0;
    node.textContent = formatValue(value, typeof value === "number");
  });
}

function renderCarcassState(frame) {
  const carcassStats = frame.carcass_stats ?? {};
  const carcassFlow = frame.carcass_flow ?? {};
  [...elements.carcassGrid.querySelectorAll("[data-carcass-field]")].forEach((node) => {
    const field = node.dataset.carcassField;
    const value =
      Object.prototype.hasOwnProperty.call(carcassFlow, field)
        ? carcassFlow[field]
        : carcassStats[field] ?? 0;
    node.textContent = formatValue(value, typeof value === "number");
  });
}

function renderTrophicState(decodedAgents) {
  const counts = { herbivore: 0, omnivore: 0, carnivore: 0 };
  for (const agent of decodedAgents) {
    counts[agent.trophicRole] = (counts[agent.trophicRole] ?? 0) + 1;
  }
  [...elements.trophicGrid.querySelectorAll("[data-trophic-field]")].forEach((node) => {
    const field = node.dataset.trophicField;
    node.textContent = formatValue(counts[field] ?? 0);
  });
}

function renderHabitatState(frame) {
  const habitatState = frame.habitat_state_counts ?? {};
  [...elements.habitatGrid.querySelectorAll("[data-habitat-field]")].forEach((node) => {
    const field = node.dataset.habitatField;
    node.textContent = formatValue(habitatState[field] ?? 0);
  });
}

function renderEcologyState(frame) {
  const ecologyState = frame.ecology_state_counts ?? {};
  const ecologyStats = frame.ecology_stats ?? {};
  [...elements.ecologyGrid.querySelectorAll("[data-ecology-field]")].forEach((node) => {
    const field = node.dataset.ecologyField;
    const value =
      field === "avg_vegetation" || field === "avg_recovery_debt"
        ? ecologyStats[field]
        : ecologyState[field] ?? 0;
    node.textContent = formatValue(value, typeof value === "number");
  });
}

function renderSpeciesList(frame, decodedAgents) {
  const speciesCounts = [...frame.species_counts].sort((left, right) => right[1] - left[1]);
  if (speciesCounts.length === 0) {
    elements.speciesEmpty.textContent = "No live species at this frame.";
    elements.speciesEmpty.classList.remove("hidden");
    elements.speciesList.classList.add("hidden");
    elements.speciesList.innerHTML = "";
    return;
  }

  const maxCount = speciesCounts[0][1];
  elements.speciesEmpty.classList.add("hidden");
  elements.speciesList.classList.remove("hidden");
  elements.speciesList.innerHTML = speciesCounts
    .map(([speciesId, count]) => {
      const record = state.payload.viewer.species_catalog[String(speciesId)];
      const liveAgent = decodedAgents.find((agent) => agent.speciesId === speciesId);
      const frameMetrics = frame.species_metrics?.[String(speciesId)] ?? null;
      const isSelected = state.selectedSpeciesId === speciesId;
      const color = `#${colorForSpecies(speciesId).toString(16).padStart(6, "0")}`;
      const share = maxCount === 0 ? 0 : count / maxCount;
      const lineages = record?.lineages?.length ?? 0;
      const title = speciesLabel(speciesId);
      const code = speciesCode(speciesId);
      return `
        <button
          type="button"
          class="species-item${isSelected ? " selected" : ""}"
          data-species-id="${speciesId}"
          style="--species-share:${share};"
          ${liveAgent ? `data-agent-id="${liveAgent.agentId}"` : ""}
        >
          <div class="species-item-header">
            <span class="species-name">
              <i class="species-swatch" style="background:${color};"></i>
              <span class="species-display-name">${escapeHtml(title)}</span>
              <span class="species-code">${escapeHtml(code)}</span>
            </span>
            <span class="species-count">${count} alive</span>
          </div>
          <div class="species-item-meta">
            <span>Peak ${record?.peak_members ?? "-"}</span>
            <span>${lineages} lineages</span>
            <span>Hydration stress ${formatPercent(frameMetrics?.hydration_stress_rate ?? 0)}</span>
          </div>
        </button>
      `;
    })
    .join("");
}

function updateInspector() {
  const payload = state.payload;
  if (!payload || state.selectedAgentId == null) {
    elements.inspectorEmpty.classList.remove("hidden");
    elements.inspectorGrid.classList.add("hidden");
    elements.inspectorGrid.innerHTML = "";
    return;
  }

  const catalog = payload.viewer.agent_catalog[String(state.selectedAgentId)];
  const frame = payload.viewer.frames[state.currentFrameIndex];
  const current = decodedAgentsForFrame(state.currentFrameIndex).find(
    (agent) => agent.agentId === state.selectedAgentId,
  ) ?? null;
  const latest = current ?? selectedAgentSnapshotAtOrBefore(state.selectedAgentId, state.currentFrameIndex);
  const snapshot = current ?? latest;
  const snapshotFrameIndex = current ? state.currentFrameIndex : latest?.frameIndex ?? state.currentFrameIndex;
  const contextFrame = payload.viewer.frames[snapshotFrameIndex] ?? frame;
  const tileContext = tileContextForAgent(snapshot, snapshotFrameIndex);
  const isAliveNow = Boolean(current);
  const hasKnownSnapshot = Boolean(snapshot);
  const speciesId = current?.speciesId ?? latest?.speciesId ?? state.selectedSpeciesId;
  const speciesRecord =
    speciesId != null ? payload.viewer.species_catalog[String(speciesId)] : null;
  const ecotypeRecord =
    hasKnownSnapshot && snapshot?.ecotypeId != null
      ? payload.viewer.ecotype_catalog?.[String(snapshot.ecotypeId)] ?? null
      : null;
  const liveSpeciesCount =
    speciesId == null
      ? "-"
      : frame.species_counts.find(([candidate]) => candidate === speciesId)?.[1] ?? 0;
  const genome = catalog.genome;
  const currentTerrainName = tileContext.terrainName;
  const currentHabitatState = tileContext.habitatState;
  const currentEcologyState = tileContext.ecologyState;
  const currentHazardType = tileContext.hazardType;
  const currentHazardLevel = tileContext.hazardLevel;
  const currentCarcassEnergy = tileContext.carcassEnergy;
  const currentReproductiveSignal = tileContext.reproductiveSignal;
  const currentCommunicationSignal = tileContext.communicationSignal;
  const currentCarcassPatch = tileContext.carcassPatch;
  const waterAccessReason = hasKnownSnapshot ? snapshot.waterAccessReason ?? "none" : null;
  const softRefugeReason = hasKnownSnapshot ? snapshot.softRefugeReason ?? "none" : null;
  const hydrologySupport = hydrologySupportFromCode(snapshot?.hydrologySupportCode ?? 0);
  const carcassDominantSource =
    currentCarcassPatch?.dominant_source_species != null
      ? speciesLabel(currentCarcassPatch.dominant_source_species)
      : currentCarcassPatch?.mixed_sources
        ? "Mixed"
        : MISSING_LABEL;
  const carcassSourceMix =
    currentCarcassPatch?.source_breakdown?.length
      ? currentCarcassPatch.source_breakdown
          .map((entry) => {
            const speciesId = entry.source_species;
            const label =
              speciesId != null
                ? speciesLabel(speciesId)
                : UNKNOWN_LABEL;
            return `${label}:${roundValue(entry.energy)}`;
          })
          .join(", ")
      : MISSING_LABEL;
  const parentIds = Array.isArray(catalog.parent_ids) ? catalog.parent_ids : [];

  const entries = [
    ["Agent", state.selectedAgentId],
    ["Species", speciesId != null ? `${speciesLabel(speciesId)} (${speciesCode(speciesId)})` : MISSING_LABEL],
    ["Species Status", speciesRecord?.status ? titleCase(String(speciesRecord.status).replaceAll("_", " ")) : UNKNOWN_LABEL],
    ["Ecotype", hasKnownSnapshot ? ecotypeRecord?.label ?? snapshot?.ecotypeId ?? UNKNOWN_LABEL : UNKNOWN_LABEL],
    ["Species Members", liveSpeciesCount],
    ["Species Peak", speciesRecord?.peak_members ?? "-"],
    [
      "Species Lineages",
      speciesRecord?.lineages?.length ? speciesRecord.lineages.join(", ") : MISSING_LABEL,
    ],
    ["Parent Species", speciesRecord?.parent_species_id ?? MISSING_LABEL],
    ["Split Tick", speciesRecord?.split_tick ?? MISSING_LABEL],
    ["Lineage", catalog.lineage_id],
    ["Parent", catalog.parent_id ?? "root"],
    ["Parents", parentIds.length ? parentIds.join(", ") : "root"],
    ["Reproductive Group", catalog.reproductive_group_id ?? MISSING_LABEL],
    [
      "Reproductive Stage",
      catalog.reproductive_stage ? titleCase(catalog.reproductive_stage) : MISSING_LABEL,
    ],
    [
      "Reproductive Expression",
      catalog.reproductive_expression
        ? titleCase(catalog.reproductive_expression)
        : MISSING_LABEL,
    ],
    ["Alive Now", yesNoLabel(isAliveNow)],
    ["Position", hasKnownSnapshot ? `${snapshot.x}, ${snapshot.y}${isAliveNow ? "" : " last seen"}` : UNKNOWN_LABEL],
    ["Terrain Here", terrainLabel(currentTerrainName)],
    ["Has Water Access", hasKnownSnapshot ? yesNoLabel(waterAccessReason !== "none") : UNKNOWN_LABEL],
    ["Water Access Reason", waterAccessLabel(waterAccessReason)],
    ["Adjacent To Water", hasKnownSnapshot ? yesNoLabel(hydrologySupport.adjacentToWater) : UNKNOWN_LABEL],
    ["Wetland Substrate", hasKnownSnapshot ? yesNoLabel(hydrologySupport.wetland) : UNKNOWN_LABEL],
    ["Flooded Support", hasKnownSnapshot ? yesNoLabel(hydrologySupport.flooded) : UNKNOWN_LABEL],
    ["Soft Refuge", softRefugeLabel(softRefugeReason)],
    ["Refuge Score", hasKnownSnapshot ? formatPercent(snapshot.refugeScore ?? 0) : UNKNOWN_LABEL],
    ["Trophic Role", hasKnownSnapshot ? trophicRoleLabel(snapshot.trophicRole) : UNKNOWN_LABEL],
    ["Health", hasKnownSnapshot ? roundValue(snapshot.health) : UNKNOWN_LABEL],
    ["Health Ratio", hasKnownSnapshot ? formatPercent(snapshot.healthRatio ?? 0) : UNKNOWN_LABEL],
    ["Injury Load", hasKnownSnapshot ? formatPercent(snapshot.injuryLoad ?? 0) : UNKNOWN_LABEL],
    ["Last Damage Source", hasKnownSnapshot ? damageSourceLabel(snapshot.lastDamageSource) : UNKNOWN_LABEL],
    ["Hazard Here", hazardLabel(currentHazardType)],
    ["Hazard Level", hasKnownSnapshot ? formatPercent(currentHazardLevel ?? 0) : UNKNOWN_LABEL],
    ["Reproductive Signal Here", hasKnownSnapshot ? roundValue(currentReproductiveSignal ?? 0) : UNKNOWN_LABEL],
    ["Communication Signal Here", hasKnownSnapshot ? roundValue(currentCommunicationSignal ?? 0) : UNKNOWN_LABEL],
    ["Carcass Here", hasKnownSnapshot ? roundValue(currentCarcassEnergy ?? 0) : UNKNOWN_LABEL],
    ["Carcass Freshness", currentCarcassPatch ? formatPercent(currentCarcassPatch.avg_freshness ?? 0) : MISSING_LABEL],
    ["Carcass Deposits", currentCarcassPatch?.deposit_count ?? MISSING_LABEL],
    ["Carcass Mixed Sources", currentCarcassPatch ? yesNoLabel(currentCarcassPatch.mixed_sources) : MISSING_LABEL],
    ["Carcass Dominant Source", carcassDominantSource],
    ["Carcass Source Mix", carcassSourceMix],
    ["Habitat State", habitatLabel(currentHabitatState)],
    ["Ecology State", ecologyLabel(currentEcologyState)],
    ["Energy Modifier", hasKnownSnapshot ? roundValue(snapshot.energyModifier) : UNKNOWN_LABEL],
    ["Hydration Modifier", hasKnownSnapshot ? roundValue(snapshot.hydrationModifier) : UNKNOWN_LABEL],
    ["Energy", hasKnownSnapshot ? roundValue(snapshot.energy) : UNKNOWN_LABEL],
    ["Hydration", hasKnownSnapshot ? roundValue(snapshot.hydration) : UNKNOWN_LABEL],
    ["Age", hasKnownSnapshot ? snapshot.age : UNKNOWN_LABEL],
    [
      "Fertility Here",
      hasKnownSnapshot
        ? roundValue(effectiveFieldValue(payload, contextFrame, "fertility", snapshot.x, snapshot.y))
        : UNKNOWN_LABEL,
    ],
    [
      "Moisture Here",
      hasKnownSnapshot
        ? roundValue(effectiveFieldValue(payload, contextFrame, "moisture", snapshot.x, snapshot.y))
        : UNKNOWN_LABEL,
    ],
    [
      "Heat Here",
      hasKnownSnapshot
        ? roundValue(effectiveFieldValue(payload, contextFrame, "heat", snapshot.x, snapshot.y))
        : UNKNOWN_LABEL,
    ],
    ["Max Energy", roundValue(genome.max_energy)],
    ["Max Hydration", roundValue(genome.max_hydration)],
    ["Max Health", roundValue(genome.max_health)],
    ["Move Cost", roundValue(genome.move_cost)],
    ["Food Eff.", roundValue(genome.food_efficiency)],
    ["Water Eff.", roundValue(genome.water_efficiency)],
    ["Attack Power", roundValue(genome.attack_power)],
    ["Attack Cost Mult.", roundValue(genome.attack_cost_multiplier)],
    ["Defense Rating", roundValue(genome.defense_rating)],
    ["Meat Eff.", roundValue(genome.meat_efficiency)],
    ["Healing Eff.", roundValue(genome.healing_efficiency)],
    ["Plant Bias", roundValue(genome.plant_bias)],
    ["Carrion Bias", roundValue(genome.carrion_bias)],
    ["Live Prey Bias", roundValue(genome.live_prey_bias)],
    ["Forest Aff.", roundValue(genome.forest_affinity)],
    ["Plain Aff.", roundValue(genome.plain_affinity)],
    ["Wetland Aff.", roundValue(genome.wetland_affinity)],
    ["Rocky Aff.", roundValue(genome.rocky_affinity)],
    ["Heat Tol.", roundValue(genome.heat_tolerance)],
    ["Repro Threshold", roundValue(genome.reproduction_threshold)],
    ["Mutation Scale", roundValue(genome.mutation_scale)],
  ];

  elements.inspectorGrid.innerHTML = entries
    .map(
      ([label, value]) =>
        `<div><dt>${escapeHtml(String(label))}</dt><dd>${escapeHtml(String(value))}</dd></div>`,
    )
    .join("");
  elements.inspectorEmpty.classList.add("hidden");
  elements.inspectorGrid.classList.remove("hidden");
}

function renderAnalyticsPanels(frame) {
  const analytics = state.payload?.viewer?.analytics ?? null;
  if (!analytics) {
    renderChartUnavailable(elements.populationChart);
    renderChartUnavailable(elements.speciesChart);
    renderChartUnavailable(elements.turnoverChart);
    renderChartUnavailable(elements.traitChart);
    renderChartUnavailable(elements.habitatChart);
    renderChartUnavailable(elements.hydrologyPrimaryChart);
    renderChartUnavailable(elements.hydrologySupportChart);
    renderChartUnavailable(elements.hazardChart);
    renderChartUnavailable(elements.carcassChart);
    renderChartUnavailable(elements.combatChart);
    renderChartUnavailable(elements.ecologyChart);
    renderSpeciesEcology(frame, null);
    renderCollapseEvents(frame, null);
    return;
  }

  renderMetricChart(elements.populationChart, {
    series: [
      {
        label: "Alive agents",
        values: analytics.population.alive_agents,
        color: "#7dd3fc",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.speciesChart, {
    series: [
      {
        label: "Live species",
        values: analytics.population.species_count,
        color: "#c084fc",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.turnoverChart, {
    series: [
      {
        label: "Births",
        values: analytics.population.births,
        color: "#4ade80",
      },
      {
        label: "Deaths",
        values: analytics.population.deaths,
        color: "#f87171",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.traitChart, {
    series: [
      {
        label: "Max energy",
        values: analytics.traits.avg_max_energy,
        color: "#f59e0b",
      },
      {
        label: "Max health",
        values: analytics.traits.avg_max_health,
        color: "#f8fafc",
      },
      {
        label: "Attack power",
        values: analytics.traits.avg_attack_power,
        color: "#fb7185",
      },
      {
        label: "Meat efficiency",
        values: analytics.traits.avg_meat_efficiency,
        color: "#38bdf8",
      },
      {
        label: "Carrion bias",
        values: analytics.traits.avg_carrion_bias,
        color: "#4ade80",
      },
      {
        label: "Live prey bias",
        values: analytics.traits.avg_live_prey_bias,
        color: "#a78bfa",
      },
    ],
    currentIndex: state.currentFrameIndex,
  });

  renderMetricChart(elements.habitatChart, {
    series: [
      {
        label: "Bloom",
        values: analytics.habitat?.bloom ?? [],
        color: "#4ade80",
      },
      {
        label: "Flooded",
        values: analytics.habitat?.flooded ?? [],
        color: "#38bdf8",
      },
      {
        label: "Parched",
        values: analytics.habitat?.parched ?? [],
        color: "#f97316",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.hydrologyPrimaryChart, {
    series: [
      {
        label: "Hard Access",
        values: analytics.hydrology_primary?.hard_access_tiles ?? [],
        color: "#f8fafc",
      },
      {
        label: "Primary Adjacent Water",
        values: analytics.hydrology_primary?.adjacent_water ?? [],
        color: "#38bdf8",
      },
      {
        label: "Primary Wetland",
        values: analytics.hydrology_primary?.wetland ?? [],
        color: "#14b8a6",
      },
      {
        label: "Primary Flooded",
        values: analytics.hydrology_primary?.flooded ?? [],
        color: "#7dd3fc",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.hydrologySupportChart, {
    series: [
      {
        label: "Shoreline Support",
        values: analytics.hydrology_support?.shoreline_support ?? [],
        color: "#fbbf24",
      },
      {
        label: "Wetland Support",
        values: analytics.hydrology_support?.wetland_support ?? [],
        color: "#14b8a6",
      },
      {
        label: "Flooded Support",
        values: analytics.hydrology_support?.flooded_support ?? [],
        color: "#7dd3fc",
      },
      {
        label: "Canopy Refuge",
        values: analytics.refuge?.canopy_refuge_tiles ?? [],
        color: "#34d399",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.hazardChart, {
    series: [
      {
        label: "Exposure",
        values: analytics.hazards?.exposure ?? [],
        color: "#fb7185",
      },
      {
        label: "Instability",
        values: analytics.hazards?.instability ?? [],
        color: "#f59e0b",
      },
      {
        label: "Hazardous tiles",
        values: analytics.hazards?.hazardous_tiles ?? [],
        color: "#f8fafc",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.carcassChart, {
    series: [
      {
        label: "Carcass tiles",
        values: analytics.carcasses?.carcass_tiles ?? [],
        color: "#fbbf24",
      },
      {
        label: "Carcass stock",
        values: analytics.carcasses?.total_carcass_energy ?? [],
        color: "#f97316",
      },
      {
        label: "Deposited",
        values: analytics.carcasses?.carcass_energy_deposited ?? [],
        color: "#fb7185",
      },
      {
        label: "Consumed",
        values: analytics.carcasses?.carcass_energy_consumed ?? [],
        color: "#4ade80",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.combatChart, {
    series: [
      {
        label: "Attacks",
        values: analytics.combat?.attack_attempts ?? [],
        color: "#fb7185",
      },
      {
        label: "Kills",
        values: analytics.combat?.kills ?? [],
        color: "#f97316",
      },
      {
        label: "Hazard damage",
        values: analytics.combat?.hazard_damage_taken ?? [],
        color: "#f8fafc",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderMetricChart(elements.ecologyChart, {
    series: [
      {
        label: "Lush",
        values: analytics.ecology?.lush ?? [],
        color: "#4ade80",
      },
      {
        label: "Recovering",
        values: analytics.ecology?.recovering ?? [],
        color: "#facc15",
      },
      {
        label: "Depleted",
        values: analytics.ecology?.depleted ?? [],
        color: "#fb7185",
      },
    ],
    currentIndex: state.currentFrameIndex,
    baselineZero: true,
  });

  renderSpeciesEcology(frame, analytics);
  renderCollapseEvents(frame, analytics);
}

function renderSpeciesEcology(frame, analytics) {
  const speciesId = state.selectedSpeciesId;
  if (speciesId == null || !state.payload) {
    elements.speciesEcologyEmpty.classList.remove("hidden");
    elements.speciesEcologyGrid.classList.add("hidden");
    elements.speciesEcologyGrid.innerHTML = "";
    elements.speciesOccupancyBars.classList.add("hidden");
    elements.speciesOccupancyBars.innerHTML = "";
    elements.speciesTrendChart.classList.add("hidden");
    elements.speciesTrendChart.innerHTML = "";
    return;
  }

  const speciesKey = String(speciesId);
  const record = state.payload.viewer.species_catalog[speciesKey];
  const metrics = frame.species_metrics?.[speciesKey] ?? {
    alive_count: analytics?.species_population?.[speciesKey]?.[state.currentFrameIndex] ?? 0,
    births: 0,
    deaths: 0,
    reproduction_success: 0,
    avg_energy_ratio: 0,
    avg_hydration_ratio: 0,
    avg_health_ratio: 0,
    avg_age: 0,
    avg_tile_vegetation: 0,
    avg_recovery_debt: 0,
    avg_refuge_score_occupied_tiles: 0,
    refuge_exposure_rate: 0,
    injury_rate: 0,
    hazard_exposure_rate: 0,
    energy_stress_rate: 0,
    hydration_stress_rate: 0,
    reproduction_ready_rate: 0,
    attack_attempts: 0,
    successful_attacks: 0,
    kills: 0,
    damage_dealt: 0,
    damage_taken: 0,
    hazard_damage_taken: 0,
    carcass_deposition: 0,
    carcass_energy_deposited: 0,
    carcass_consumption: 0,
    carcass_energy_consumed: 0,
    carcass_gained_energy: 0,
    terrain_occupancy: emptyTerrainOccupancy(state.payload.viewer.map),
    hydrology_exposure_counts: emptyHydrologyExposureCounts(),
    habitat_occupancy: { stable: 0, bloom: 0, flooded: 0, parched: 0 },
    ecology_occupancy: { stable: 0, lush: 0, recovering: 0, depleted: 0 },
    hazard_occupancy: { none: 0, exposure: 0, instability: 0 },
    trophic_role_occupancy: { herbivore: 0, omnivore: 0, carnivore: 0 },
  };
  const hydrologyExposure = metrics.hydrology_exposure_counts ?? emptyHydrologyExposureCounts();
  const habitatOccupancy = metrics.habitat_occupancy ?? {
    stable: 0,
    bloom: 0,
    flooded: 0,
    parched: 0,
  };
  const ecologyOccupancy = metrics.ecology_occupancy ?? {
    stable: 0,
    lush: 0,
    recovering: 0,
    depleted: 0,
  };
  const hazardOccupancy = metrics.hazard_occupancy ?? { none: 0, exposure: 0, instability: 0 };
  const trophicOccupancy = metrics.trophic_role_occupancy ?? {
    herbivore: 0,
    omnivore: 0,
    carnivore: 0,
  };

  const entries = [
    ["Species", `${speciesLabel(speciesId)} (${speciesCode(speciesId)})`],
    ["Alive Now", metrics.alive_count],
    ["Births This Tick", metrics.births],
    ["Deaths This Tick", metrics.deaths],
    ["Repro Success", metrics.reproduction_success],
    ["Avg Energy", formatPercent(metrics.avg_energy_ratio)],
    ["Avg Hydration", formatPercent(metrics.avg_hydration_ratio)],
    ["Avg Health", formatPercent(metrics.avg_health_ratio)],
    ["Avg Age", metrics.avg_age],
    ["Avg Vegetation", formatPercent(metrics.avg_tile_vegetation)],
    ["Avg Recovery Debt", formatPercent(metrics.avg_recovery_debt)],
    ["Avg Refuge Score (Occupied Tiles)", formatPercent(metrics.avg_refuge_score_occupied_tiles)],
    ["Refuge Exposure Rate", formatPercent(metrics.refuge_exposure_rate)],
    ["Injury Rate", formatPercent(metrics.injury_rate)],
    ["Hazard Exposure Rate", formatPercent(metrics.hazard_exposure_rate)],
    ["Energy Stress", formatPercent(metrics.energy_stress_rate)],
    ["Hydration Stress", formatPercent(metrics.hydration_stress_rate)],
    ["Ready To Reproduce", formatPercent(metrics.reproduction_ready_rate)],
    ["Attack Attempts", metrics.attack_attempts],
    ["Successful Attacks", metrics.successful_attacks],
    ["Kills", metrics.kills],
    ["Damage Dealt", roundValue(metrics.damage_dealt)],
    ["Damage Taken", roundValue(metrics.damage_taken)],
    ["Hazard Damage", roundValue(metrics.hazard_damage_taken)],
    ["Carcass Deposition", metrics.carcass_deposition],
    ["Carcass Energy Deposited", roundValue(metrics.carcass_energy_deposited)],
    ["Carcass Consumption", metrics.carcass_consumption],
    ["Carcass Energy Consumed", roundValue(metrics.carcass_energy_consumed)],
    ["Carcass Gained Energy", roundValue(metrics.carcass_gained_energy)],
    ["Hard Water Exposure", `${metrics.terrain_occupancy?.water_access ?? 0} (${formatPercent((metrics.terrain_occupancy?.water_access ?? 0) / Math.max(metrics.alive_count, 1))})`],
    ["Shoreline Support Exposure", `${hydrologyExposure.shoreline_support} (${formatPercent(hydrologyExposure.shoreline_support / Math.max(metrics.alive_count, 1))})`],
    ["Primary Adjacent Water Exposure", `${hydrologyExposure.primary_adjacent_water} (${formatPercent(hydrologyExposure.primary_adjacent_water / Math.max(metrics.alive_count, 1))})`],
    ["Primary Wetland Exposure", `${hydrologyExposure.primary_wetland} (${formatPercent(hydrologyExposure.primary_wetland / Math.max(metrics.alive_count, 1))})`],
    ["Primary Flooded Exposure", `${hydrologyExposure.primary_flooded} (${formatPercent(hydrologyExposure.primary_flooded / Math.max(metrics.alive_count, 1))})`],
    ["Refuge Exposure", `${hydrologyExposure.refuge_exposed} (${formatPercent(hydrologyExposure.refuge_exposed / Math.max(metrics.alive_count, 1))})`],
    ["Exposure Hazard", `${hazardOccupancy.exposure} (${formatPercent(hazardOccupancy.exposure / Math.max(metrics.alive_count, 1))})`],
    ["Instability Hazard", `${hazardOccupancy.instability} (${formatPercent(hazardOccupancy.instability / Math.max(metrics.alive_count, 1))})`],
    ["Herbivore Mix", `${trophicOccupancy.herbivore} (${formatPercent(trophicOccupancy.herbivore / Math.max(metrics.alive_count, 1))})`],
    ["Omnivore Mix", `${trophicOccupancy.omnivore} (${formatPercent(trophicOccupancy.omnivore / Math.max(metrics.alive_count, 1))})`],
    ["Carnivore Mix", `${trophicOccupancy.carnivore} (${formatPercent(trophicOccupancy.carnivore / Math.max(metrics.alive_count, 1))})`],
    ["Bloom Occupancy", `${habitatOccupancy.bloom} (${formatPercent(habitatOccupancy.bloom / Math.max(metrics.alive_count, 1))})`],
    ["Flood Exposure", `${habitatOccupancy.flooded} (${formatPercent(habitatOccupancy.flooded / Math.max(metrics.alive_count, 1))})`],
    ["Parched Exposure", `${habitatOccupancy.parched} (${formatPercent(habitatOccupancy.parched / Math.max(metrics.alive_count, 1))})`],
    ["Lush Exposure", `${ecologyOccupancy.lush} (${formatPercent(ecologyOccupancy.lush / Math.max(metrics.alive_count, 1))})`],
    ["Recovery Exposure", `${ecologyOccupancy.recovering} (${formatPercent(ecologyOccupancy.recovering / Math.max(metrics.alive_count, 1))})`],
    ["Depleted Exposure", `${ecologyOccupancy.depleted} (${formatPercent(ecologyOccupancy.depleted / Math.max(metrics.alive_count, 1))})`],
    ["Peak Members", record?.peak_members ?? "-"],
  ];

  elements.speciesEcologyGrid.innerHTML = entries
    .map(
      ([label, value]) =>
        `<div><dt>${escapeHtml(String(label))}</dt><dd>${escapeHtml(String(value))}</dd></div>`,
    )
    .join("");
  elements.speciesEcologyEmpty.classList.add("hidden");
  elements.speciesEcologyGrid.classList.remove("hidden");

  const occupancy = metrics.terrain_occupancy ?? emptyTerrainOccupancy(state.payload.viewer.map);
  const aliveCount = Math.max(metrics.alive_count, 1);
  const occupancySpec = [
    ...terrainEntries(state.payload.viewer.map)
      .filter(({ name }) => name !== "water")
      .map(({ name }) => ({
        label: titleCase(name),
        key: name,
        color: terrainCssColor(terrainCodeByName(name)),
      })),
    { label: "Water Access", key: "water_access", color: "#38bdf8" },
  ];
  elements.speciesOccupancyBars.innerHTML = occupancySpec
    .map(({ label, key, color }) => {
      const count = occupancy[key] ?? 0;
      const ratio = count / aliveCount;
      return `
        <div class="occupancy-row">
          <span>${escapeHtml(label)}</span>
          <div class="occupancy-track">
            <span class="occupancy-fill" style="width:${ratio * 100}%; background:${color};"></span>
          </div>
          <span>${count} (${formatPercent(ratio)})</span>
        </div>
      `;
    })
    .join("");
  elements.speciesOccupancyBars.classList.remove("hidden");

  const populationSeries = analytics?.species_population?.[speciesKey] ?? null;
  if (populationSeries) {
    elements.speciesTrendChart.classList.remove("hidden");
    renderMetricChart(elements.speciesTrendChart, {
      series: [
        {
          label: `${speciesLabel(speciesId)} population`,
          values: populationSeries,
          color: `#${colorForSpecies(speciesId).toString(16).padStart(6, "0")}`,
        },
      ],
      currentIndex: state.currentFrameIndex,
      baselineZero: true,
    });
  } else {
    elements.speciesTrendChart.classList.add("hidden");
    elements.speciesTrendChart.innerHTML = "";
  }
}

function taxonomyEventSpeciesLabel(speciesId) {
  if (speciesId == null) {
    return "Unknown species";
  }
  return speciesLabel(speciesId);
}

function taxonomyEventSelected(event) {
  if (state.selectedSpeciesId == null) {
    return false;
  }
  if (event.type === "speciation") {
    return [
      event.source_species_id,
      event.continuation_species_id,
      event.daughter_species_id,
    ].includes(state.selectedSpeciesId);
  }
  return event.species_id === state.selectedSpeciesId;
}

function renderCollapseEvents(frame, analytics) {
  const events = [
    ...(analytics?.collapse_events ?? []),
    ...(analytics?.speciation_events ?? []),
  ];
  const visibleEvents = events
    .filter((event) => event.tick <= frame.tick)
    .sort((left, right) => {
      if (left.tick !== right.tick) {
        return right.tick - left.tick;
      }
      if (left.type !== right.type) {
        return String(left.type).localeCompare(String(right.type));
      }
      return (
        (right.species_id ?? right.source_species_id ?? 0) -
        (left.species_id ?? left.source_species_id ?? 0)
      );
    })
    .slice(0, 12);
  if (visibleEvents.length === 0) {
    elements.collapseEmpty.classList.remove("hidden");
    elements.collapseList.classList.add("hidden");
    elements.collapseList.innerHTML = "";
    return;
  }

  elements.collapseEmpty.classList.add("hidden");
  elements.collapseList.classList.remove("hidden");
  elements.collapseList.innerHTML = visibleEvents
    .map((event) => {
      const classes = [
        "event-item",
        event.tick === frame.tick ? "current" : "",
        taxonomyEventSelected(event) ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" ");
      if (event.type === "speciation") {
        return `
          <div class="${classes}">
            <div class="event-type">${escapeHtml(event.type)}</div>
            <div>
              ${escapeHtml(taxonomyEventSpeciesLabel(event.source_species_id))}
              -> ${escapeHtml(taxonomyEventSpeciesLabel(event.continuation_species_id))}
              / ${escapeHtml(taxonomyEventSpeciesLabel(event.daughter_species_id))}
            </div>
            <div class="event-meta">
              Tick ${event.tick}
            </div>
          </div>
        `;
      }
      return `
        <div class="${classes}">
          <div class="event-type">${escapeHtml(event.type)}</div>
          <div>${escapeHtml(taxonomyEventSpeciesLabel(event.species_id))}</div>
          <div class="event-meta">
            ${
              event.type === "collapse"
                ? `Tick ${event.tick} - peak ${event.peak} - current ${event.current}`
                : `Tick ${event.tick}`
            }
          </div>
        </div>
      `;
    })
    .join("");
}

function renderMetricChart(
  host,
  {
    series,
    currentIndex,
    baselineZero = false,
  },
) {
  if (!host || !series || series.length === 0 || series[0].values.length === 0) {
    renderChartUnavailable(host);
    return;
  }

  const width = 560;
  const height = 160;
  const padding = { top: 10, right: 10, bottom: 22, left: 24 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const allValues = series.flatMap((item) => item.values);
  const minValue = baselineZero ? 0 : Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = Math.max(maxValue - minValue, 1e-9);
  const steps = Math.max(series[0].values.length - 1, 1);
  const currentX = padding.left + (innerWidth * currentIndex) / steps;

  const gridLines = [0, 0.5, 1].map((ratio) => {
    const y = padding.top + innerHeight * ratio;
    return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="rgba(159,179,209,0.16)" stroke-width="1" />`;
  });

  const paths = series.map((item) => {
    const path = item.values
      .map((value, index) => {
        const x = padding.left + (innerWidth * index) / steps;
        const y =
          padding.top +
          innerHeight -
          ((value - minValue) / range) * innerHeight;
        return `${index === 0 ? "M" : "L"} ${roundValue(x)} ${roundValue(y)}`;
      })
      .join(" ");
    return `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />`;
  });

  const markers = series
    .map((item) => {
      const currentValue = item.values[currentIndex];
      const y =
        padding.top +
        innerHeight -
        ((currentValue - minValue) / range) * innerHeight;
      return `<circle cx="${roundValue(currentX)}" cy="${roundValue(y)}" r="3.5" fill="${item.color}" stroke="#f8fafc" stroke-width="1.1" />`;
    })
    .join("");

  const svg = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Replay analytics chart">
      <rect x="0" y="0" width="${width}" height="${height}" rx="10" fill="transparent" />
      ${gridLines.join("")}
      <line
        x1="${roundValue(currentX)}"
        y1="${padding.top}"
        x2="${roundValue(currentX)}"
        y2="${height - padding.bottom}"
        stroke="rgba(248,250,252,0.24)"
        stroke-width="1"
        stroke-dasharray="4 4"
      />
      ${paths.join("")}
      ${markers}
    </svg>
  `;

  const legend = `
    <div class="chart-legend">
      ${series
        .map((item) => {
          const value = item.values[currentIndex];
          return `
            <span>
              <i class="chart-dot" style="background:${item.color};"></i>
              ${escapeHtml(item.label)}: ${escapeHtml(formatValue(value, true))}
            </span>
          `;
        })
        .join("")}
    </div>
  `;

  host.innerHTML = svg + legend;
}

function renderChartUnavailable(host) {
  if (!host) return;
  host.innerHTML = '<div class="muted">Analytics unavailable for this replay.</div>';
}

function onCanvasPointerDown(event) {
  const payload = state.payload;
  if (!payload) return;
  state.mapPointer = {
    startX: event.global.x,
    startY: event.global.y,
    lastX: event.global.x,
    lastY: event.global.y,
    panX: state.mapView.panX,
    panY: state.mapView.panY,
    moved: false,
  };
  elements.canvasHost?.classList.add("panning");
}

function onCanvasPointerMove(event) {
  if (!state.payload || !state.mapPointer) return;
  const dx = event.global.x - state.mapPointer.startX;
  const dy = event.global.y - state.mapPointer.startY;
  if (Math.abs(dx) + Math.abs(dy) < 4 && !state.mapPointer.moved) return;
  state.mapPointer.moved = true;
  state.mapView.panX = state.mapPointer.panX + dx;
  state.mapView.panY = state.mapPointer.panY + dy;
  clampMapView();
  renderFrame(state.currentFrameIndex);
}

function onCanvasPointerUp(event) {
  if (!state.payload || !state.mapPointer) return;
  const pointer = state.mapPointer;
  state.mapPointer = null;
  elements.canvasHost?.classList.remove("panning");
  if (pointer.moved) {
    renderFrame(state.currentFrameIndex);
    return;
  }

  const tile = tileFromCanvasPoint(event.global.x, event.global.y);
  if (!tile) return;
  state.hoveredTile = tile;
  state.explainedTile = tile;
  selectAgentAtTile(tile.x, tile.y);
}

function tileFromCanvasPoint(localX, localY) {
  const payload = state.payload;
  if (!payload) return null;
  const map = payload.viewer.map;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);
  const gridX = Math.floor((localX - offset.x) / tileSize);
  const gridY = Math.floor((localY - offset.y) / tileSize);
  if (!isPointOnMap(gridX, gridY)) return null;
  return { x: gridX, y: gridY };
}

function selectAgentAtTile(gridX, gridY) {
  const hit = decodedAgentsForFrame(state.currentFrameIndex).find(
    (agent) => agent.x === gridX && agent.y === gridY,
  );
  if (!hit) {
    state.selectedAgentId = null;
    state.selectedSpeciesId = null;
    setActiveDetailView("overview");
    renderFrame(state.currentFrameIndex);
    return;
  }

  state.selectedAgentId = hit.agentId;
  state.selectedSpeciesId = hit.speciesId;
  setActiveDetailView("agent");
  renderFrame(state.currentFrameIndex);
}

function onCanvasKeyDown(event) {
  if (!state.payload) return;
  const handledKeys = new Set([
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "Enter",
    "Escape",
    "+",
    "=",
    "-",
    "_",
    "0",
  ]);
  if (!handledKeys.has(event.key)) return;
  event.preventDefault();

  if (event.key === "+" || event.key === "=") {
    setMapZoom(state.mapView.zoom * MAP_ZOOM_STEP);
    return;
  }
  if (event.key === "-" || event.key === "_") {
    setMapZoom(state.mapView.zoom / MAP_ZOOM_STEP);
    return;
  }
  if (event.key === "0") {
    resetMapView();
    return;
  }
  if (event.key === "Escape") {
    state.selectedAgentId = null;
    state.selectedSpeciesId = null;
    setActiveDetailView("overview");
    renderFrame(state.currentFrameIndex);
    return;
  }
  if (event.key === "Enter") {
    const tile = state.hoveredTile ?? defaultFocusedTile();
    if (tile) {
      state.hoveredTile = tile;
      selectAgentAtTile(tile.x, tile.y);
    }
    return;
  }

  const tile = state.hoveredTile ?? defaultFocusedTile();
  if (!tile) return;
  const next = {
    x: tile.x + (event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0),
    y: tile.y + (event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0),
  };
  const map = state.payload.viewer.map;
  state.hoveredTile = {
    x: clamp(next.x, 0, map.width - 1),
    y: clamp(next.y, 0, map.height - 1),
  };
  renderTileHover();
  renderTileInspector();
  syncDebugState();
}

function defaultFocusedTile() {
  const payload = state.payload;
  if (!payload) return null;
  const selected = decodedAgentsForFrame(state.currentFrameIndex).find(
    (agent) => agent.agentId === state.selectedAgentId,
  );
  if (selected) return { x: selected.x, y: selected.y };
  const map = payload.viewer.map;
  return {
    x: Math.floor(map.width / 2),
    y: Math.floor(map.height / 2),
  };
}

function startPlayback() {
  if (!state.payload || state.playing) return;
  stopEpisodePlayback(false);
  state.playing = true;
  elements.playToggle.textContent = "Pause";

  const tick = () => {
    if (!state.playing || !state.payload) return;
    const speed = Number(elements.playbackSpeed.value);
    const nextIndex =
      state.currentFrameIndex >= state.payload.viewer.frames.length - 1
        ? 0
        : state.currentFrameIndex + 1;
    renderFrame(nextIndex);
    state.timerId = window.setTimeout(tick, Math.max(40, 220 / speed));
  };

  tick();
}

function stopPlayback(options = {}) {
  state.playing = false;
  elements.playToggle.textContent = "Play";
  if (state.timerId != null) {
    window.clearTimeout(state.timerId);
    state.timerId = null;
  }
  if (options.stopEpisode !== false) {
    stopEpisodePlayback(false);
  }
}

function terrainColor(code) {
  const palette = {
    0: 0x293d2d,
    1: 0x0f5132,
    2: 0x496e40,
    3: 0x5d5249,
    4: 0x1d4ed8,
  };
  return palette[code] ?? UNKNOWN_TERRAIN_COLOR;
}

function terrainBaseColor(code) {
  const palette = {
    0: 0x253028,
    1: 0x183626,
    2: 0x314b34,
    3: 0x403732,
    4: 0x143262,
  };
  return palette[code] ?? UNKNOWN_TERRAIN_COLOR;
}

function terrainCssColor(code) {
  return `#${terrainColor(code).toString(16).padStart(6, "0")}`;
}

function fieldColor(mode, value) {
  const clamped = clamp01(value);
  if (mode === "fertility") {
    return blendColor(0x2b1e10, 0x88ff8a, clamped);
  }
  if (mode === "moisture") {
    return blendColor(0x3a2611, 0x45c7ff, clamped);
  }
  return blendColor(0x24476b, 0xff7b3d, clamped);
}

function habitatStateColor(code) {
  const palette = {
    0: 0x6b7280,
    1: 0x4ade80,
    2: 0x38bdf8,
    3: 0xf97316,
  };
  return palette[code] ?? palette[0];
}

function hydrologyReasonColor(code, terrainCode) {
  if (terrainCode === terrainCodeByName("water") || code == null || code < 0) {
    return terrainColor(terrainCodeByName("water"));
  }
  const palette = {
    0: 0x525f75,
    1: 0x38bdf8,
    2: 0x14b8a6,
    3: 0x7dd3fc,
  };
  return palette[code] ?? palette[0];
}

function shorelineColor(code, terrainCode) {
  if (terrainCode === terrainCodeByName("water") || code == null || code < 0) {
    return terrainColor(terrainCodeByName("water"));
  }
  const support = hydrologySupportFromCode(code);
  if (!support.adjacentToWater) {
    return 0x4b5563;
  }
  if (support.flooded) {
    return 0x7dd3fc;
  }
  if (support.wetland) {
    return 0x14b8a6;
  }
  return 0xfbbf24;
}

function refugeColor(reasonCode, scoreCode, terrainCode) {
  if (terrainCode === terrainCodeByName("water") || scoreCode == null || scoreCode < 0) {
    return terrainColor(terrainCodeByName("water"));
  }
  const score = clamp01(scoreCode / 100);
  const base = blendColor(0x374151, 0x86efac, score);
  return reasonCode === 1 ? blendColor(base, 0x34d399, 0.55) : base;
}

function hazardColor(typeCode, levelCode, terrainCode) {
  if (terrainCode === terrainCodeByName("water") || typeCode == null || typeCode < 0) {
    return terrainColor(terrainCodeByName("water"));
  }
  const level = clamp01((levelCode ?? 0) / 100);
  if (typeCode === 1) {
    return blendColor(0x374151, 0xfb7185, level);
  }
  if (typeCode === 2) {
    return blendColor(0x374151, 0xf59e0b, level);
  }
  return blendColor(0x374151, 0x94a3b8, 0.22);
}

function carcassColor(energyCode, freshnessCode, terrainCode) {
  if (terrainCode === terrainCodeByName("water") || energyCode == null || energyCode < 0) {
    return terrainColor(terrainCodeByName("water"));
  }
  const energy = clamp01((energyCode ?? 0) / 100);
  if (energy <= 0) {
    return blendColor(0x1f2937, terrainBaseColor(terrainCode), 0.55);
  }
  const freshness = clamp01((freshnessCode ?? 0) / 100);
  return blendColor(0x5b2c06, 0xfbbf24, energy * 0.55 + freshness * 0.45);
}

function signalColor(value, terrainCode) {
  if (terrainCode === terrainCodeByName("water")) {
    return terrainColor(terrainCodeByName("water"));
  }
  const intensity = clamp01(Number(value ?? 0));
  if (intensity <= 0) {
    return blendColor(0x1f2937, terrainBaseColor(terrainCode), 0.5);
  }
  return blendColor(0x0f172a, 0x2dd4bf, Math.sqrt(intensity));
}

function trophicColor(code, terrainCode) {
  if (terrainCode === terrainCodeByName("water")) {
    return terrainColor(terrainCodeByName("water"));
  }
  const palette = {
    0: 0x334155,
    1: 0x22c55e,
    2: 0xf59e0b,
    3: 0xfb7185,
  };
  return palette[code] ?? palette[0];
}

function ecologyStateColor(code, terrainCode) {
  if (terrainCode === terrainCodeByName("water") || code == null || code < 0) {
    return terrainColor(terrainCodeByName("water"));
  }
  const palette = {
    0: 0x64748b,
    1: 0x22c55e,
    2: 0xfacc15,
    3: 0xfb7185,
  };
  return palette[code] ?? palette[0];
}

function colorForSpecies(speciesId) {
  const hue = (speciesId * 47) % 360;
  const [red, green, blue] = hslToRgb(hue / 360, 0.7, 0.62);
  return (red << 16) + (green << 8) + blue;
}

function getBaseTileSize(width, height) {
  const hostWidth = elements.canvasHost.clientWidth;
  const hostHeight = elements.canvasHost.clientHeight;
  return Math.max(8, Math.floor(Math.min(hostWidth / width, hostHeight / height)));
}

function getTileSize(width, height) {
  return getBaseTileSize(width, height) * clamp(state.mapView.zoom, MAP_MIN_ZOOM, MAP_MAX_ZOOM);
}

function getMapOffset(width, height, tileSize) {
  return {
    x: Math.floor((elements.canvasHost.clientWidth - width * tileSize) / 2 + state.mapView.panX),
    y: Math.floor((elements.canvasHost.clientHeight - height * tileSize) / 2 + state.mapView.panY),
  };
}

function setMapZoom(nextZoom) {
  if (!state.payload) return;
  state.activeCameraPreset = "custom";
  state.mapView.zoom = clamp(Number(nextZoom) || 1, MAP_MIN_ZOOM, MAP_MAX_ZOOM);
  clampMapView();
  persistViewerPreferences();
  renderFrame(state.currentFrameIndex);
}

function resetMapView() {
  state.mapView = { zoom: 1, panX: 0, panY: 0 };
  state.activeCameraPreset = "world";
  persistViewerPreferences();
  renderFrame(state.currentFrameIndex);
}

function fitMapToFocus() {
  state.activeCameraPreset = "selected";
  persistViewerPreferences();
  focusMapOnPoint(mapFocusTarget(), 1.82);
}

function applyMapCameraPreset(preset) {
  if (!state.payload) return;
  const normalized = ["world", "selected", "episode", "pressure"].includes(preset) ? preset : "world";
  state.activeCameraPreset = normalized;
  persistViewerPreferences();
  if (normalized === "world") {
    state.mapView = { zoom: 1, panX: 0, panY: 0 };
    renderFrame(state.currentFrameIndex);
    return;
  }
  if (normalized === "selected") {
    focusMapOnPoint(mapFocusTarget(), 1.82);
    return;
  }
  if (normalized === "episode") {
    focusMapOnPoint(currentEpisodeFocusTarget(), 2.05);
    return;
  }
  focusMapOnPoint(primaryPressureTarget(), 1.9);
}

function focusMapOnPoint(target, zoom = 1.82) {
  if (!state.payload) return;
  if (!target) {
    resetMapView();
    return;
  }
  const map = state.payload.viewer.map;
  state.mapView.zoom = clamp(zoom, MAP_MIN_ZOOM, MAP_MAX_ZOOM);
  state.mapView.panX = 0;
  state.mapView.panY = 0;
  const tileSize = getTileSize(map.width, map.height);
  const baseOffset = {
    x: Math.floor((elements.canvasHost.clientWidth - map.width * tileSize) / 2),
    y: Math.floor((elements.canvasHost.clientHeight - map.height * tileSize) / 2),
  };
  state.mapView.panX =
    elements.canvasHost.clientWidth / 2 - (baseOffset.x + (target.x + 0.5) * tileSize);
  state.mapView.panY =
    elements.canvasHost.clientHeight / 2 - (baseOffset.y + (target.y + 0.5) * tileSize);
  clampMapView();
  renderFrame(state.currentFrameIndex);
}

function currentEpisodeFocusTarget() {
  const tick = state.payload?.viewer?.frames?.[state.currentFrameIndex]?.tick ?? 0;
  const context = episodeContextForTick(tick);
  const event = (state.eventsByTick.get(Number(context?.episode?.tick ?? tick)) ?? [])
    .map((candidate) => eventMarkerPosition(candidate, new Map(), candidate.tick))
    .find((marker) => marker && isPointOnMap(marker.x, marker.y));
  return event ? { x: event.x, y: event.y } : mapFocusTarget();
}

function primaryPressureTarget() {
  const payload = state.payload;
  if (!payload) return null;
  const frame = payload.viewer.frames[state.currentFrameIndex];
  const map = payload.viewer.map;
  let best = null;
  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const terrainCode = map.terrain_codes?.[y]?.[x];
      if (terrainCode === terrainCodeByName("water")) continue;
      const hazard = (frame.hazard_level_codes?.[y]?.[x] ?? 0) / 100;
      const carcass = (frame.carcass_energy_codes?.[y]?.[x] ?? 0) / 100;
      const recoveryDebt = (frame.tile_recovery_debt_codes?.[y]?.[x] ?? 0) / 100;
      const vegetation = Number(effectiveFieldValue(payload, frame, "fertility", x, y) ?? 0);
      const score = hazard * 1.25 + carcass + recoveryDebt * 0.6 + Math.max(0, 0.5 - vegetation) * 0.45;
      if (!best || score > best.score) {
        best = { x, y, score };
      }
    }
  }
  return best ? { x: best.x, y: best.y } : mapFocusTarget();
}

function mapFocusTarget() {
  const decodedAgents = decodedAgentsForFrame(state.currentFrameIndex);
  const selectedAgent = decodedAgents.find((agent) => agent.agentId === state.selectedAgentId);
  if (selectedAgent) return { x: selectedAgent.x, y: selectedAgent.y };
  if (state.selectedSpeciesId != null) {
    const agents = decodedAgents.filter((agent) => agent.speciesId === state.selectedSpeciesId);
    if (agents.length > 0) {
      return {
        x: agents.reduce((total, agent) => total + agent.x, 0) / agents.length,
        y: agents.reduce((total, agent) => total + agent.y, 0) / agents.length,
      };
    }
  }
  return state.hoveredTile ? { ...state.hoveredTile } : null;
}

function clampMapView() {
  if (!state.payload) return;
  const map = state.payload.viewer.map;
  state.mapView.zoom = clamp(Number(state.mapView.zoom) || 1, MAP_MIN_ZOOM, MAP_MAX_ZOOM);
  const tileSize = getTileSize(map.width, map.height);
  const maxPanX = Math.max(48, (map.width * tileSize - elements.canvasHost.clientWidth) / 2 + 72);
  const maxPanY = Math.max(48, (map.height * tileSize - elements.canvasHost.clientHeight) / 2 + 72);
  state.mapView.panX = clamp(Number(state.mapView.panX) || 0, -maxPanX, maxPanX);
  state.mapView.panY = clamp(Number(state.mapView.panY) || 0, -maxPanY, maxPanY);
}

function updateMapNavigation() {
  if (!elements.mapZoomLabel) return;
  elements.mapZoomLabel.textContent = `${Math.round(state.mapView.zoom * 100)}%`;
  elements.mapCameraPresets?.querySelectorAll("[data-map-camera-preset]").forEach((button) => {
    button.classList.toggle("selected", button.dataset.mapCameraPreset === state.activeCameraPreset);
  });
}

function setStatus(message) {
  elements.loadStatus.textContent = message;
  syncDebugState();
}

function clearPixiLayer(layer) {
  for (const child of layer.removeChildren()) {
    child.destroy({ children: true });
  }
}

function overlayColorForTile(payload, frame, x, y, terrainCode) {
  if (!isKnownTerrainCode(terrainCode)) {
    return UNKNOWN_TERRAIN_COLOR;
  }
  if (state.overlayMode === "hydrology") {
    return hydrologyReasonColor(frame.hydrology_primary_codes?.[y]?.[x], terrainCode);
  }
  if (state.overlayMode === "shoreline") {
    return shorelineColor(frame.hydrology_support_codes?.[y]?.[x], terrainCode);
  }
  if (state.overlayMode === "refuge") {
    return refugeColor(
      frame.refuge_codes?.[y]?.[x],
      frame.refuge_score_codes?.[y]?.[x],
      terrainCode,
    );
  }
  if (state.overlayMode === "hazard") {
    return hazardColor(
      frame.hazard_type_codes?.[y]?.[x],
      frame.hazard_level_codes?.[y]?.[x],
      terrainCode,
    );
  }
  if (state.overlayMode === "carcass") {
    return carcassColor(
      frame.carcass_energy_codes?.[y]?.[x],
      frame.carcass_freshness_codes?.[y]?.[x],
      terrainCode,
    );
  }
  if (state.overlayMode === "reproductive_signal") {
    return signalColor(frame.signal_fields?.reproductive_signal?.[y]?.[x], terrainCode);
  }
  if (state.overlayMode === "trophic") {
    return trophicColor(frame.trophic_role_codes?.[y]?.[x] ?? 0, terrainCode);
  }
  if (state.overlayMode === "habitat") {
    return habitatStateColor(frame.habitat_state_codes?.[y]?.[x] ?? 0);
  }
  if (state.overlayMode === "ecology") {
    return ecologyStateColor(frame.ecology_state_codes?.[y]?.[x], terrainCode);
  }
  return fieldColor(
    state.overlayMode,
    effectiveFieldValue(payload, frame, state.overlayMode, x, y),
  );
}

function effectiveFieldValue(payload, frame, fieldName, x, y) {
  if (fieldName === "habitat") {
    return frame.habitat_state_codes?.[y]?.[x] ?? 0;
  }
  if (fieldName === "reproductive_signal" || fieldName === "communication_signal") {
    return frame.signal_fields?.[fieldName]?.[y]?.[x] ?? 0;
  }
  const base = payload.viewer.map.base_tile_fields ?? payload.viewer.map.environment_fields;
  const config = payload.config?.environment ?? {};
  const fieldState = frame?.field_state ?? {};
  const terrainCode = payload.viewer.map.terrain_codes[y][x];
  const terrainName = terrainNameFromCode(payload.viewer.map, terrainCode);
  const width = payload.viewer.map.width;
  const height = payload.viewer.map.height;
  const xNorm = x / Math.max(width - 1, 1);
  const yNorm = y / Math.max(height - 1, 1);

  let fertility = base.fertility[y][x];
  let moisture = base.moisture[y][x] + (fieldState.moisture_shift ?? 0);
  let heat = base.heat[y][x] + (fieldState.heat_shift ?? 0);

  moisture +=
    (config.moisture_front_strength ?? 0) *
    bandInfluence(xNorm, fieldState.moisture_front_x ?? 0.5, config.front_width ?? 0);
  heat +=
    (config.heat_front_strength ?? 0) *
    bandInfluence(yNorm, fieldState.heat_front_y ?? 0.5, config.front_width ?? 0);

  if (
    terrainName !== "water" &&
    hasAdjacentWaterTerrain(payload.viewer.map.terrain_codes, x, y)
  ) {
    moisture += config.adjacent_water_moisture_bonus ?? 0;
  }

  const disturbanceInfluence = radialInfluence(
    xNorm,
    yNorm,
    fieldState.disturbance_center_x ?? 0.5,
    fieldState.disturbance_center_y ?? 0.5,
    config.disturbance_radius ?? 0,
  );
  const disturbanceStrength = fieldState.disturbance_strength ?? 0;
  if (fieldState.disturbance_type === "storm") {
    moisture += disturbanceStrength * disturbanceInfluence;
    heat -= disturbanceStrength * 0.62 * disturbanceInfluence;
    fertility += disturbanceStrength * 0.16 * disturbanceInfluence;
  } else if (fieldState.disturbance_type === "drought") {
    moisture -= disturbanceStrength * disturbanceInfluence;
    heat += disturbanceStrength * 0.75 * disturbanceInfluence;
    fertility -= disturbanceStrength * 0.18 * disturbanceInfluence;
  }

  moisture = clamp01(moisture);
  heat = clamp01(heat);
  fertility = clamp01(
    fertility + (moisture - 0.5) * (config.fertility_moisture_coupling ?? 0),
  );

  if (fieldName === "fertility") return fertility;
  if (fieldName === "moisture") return moisture;
  return heat;
}

function bandInfluence(position, center, width) {
  if (!width || width <= 0) return 0;
  return Math.max(0, 1 - Math.abs(position - center) / width);
}

function radialInfluence(x, y, centerX, centerY, radius) {
  if (!radius || radius <= 0) return 0;
  const distance = Math.hypot(x - centerX, y - centerY);
  return Math.max(0, 1 - distance / radius);
}

function hasAdjacentWaterTerrain(terrainCodes, x, y) {
  const moves = [
    [0, -1],
    [0, 1],
    [1, 0],
    [-1, 0],
  ];
  return moves.some(([dx, dy]) => {
    const nextX = x + dx;
    const nextY = y + dy;
    return (
      nextY >= 0 &&
      nextY < terrainCodes.length &&
      nextX >= 0 &&
      nextX < terrainCodes[0].length &&
      terrainCodes[nextY][nextX] === terrainCodeByName("water")
    );
  });
}

function emptyTerrainOccupancy(map) {
  return Object.fromEntries([
    ...terrainEntries(map)
      .filter(({ name }) => name !== "water")
      .map(({ name }) => [name, 0]),
    ["water_access", 0],
  ]);
}

function emptyHydrologyExposureCounts() {
  return {
    primary_none: 0,
    primary_adjacent_water: 0,
    primary_wetland: 0,
    primary_flooded: 0,
    shoreline_support: 0,
    wetland_support: 0,
    flooded_support: 0,
    refuge_exposed: 0,
  };
}

function terrainEntries(map) {
  return Object.entries(map.terrain_legend ?? {})
    .map(([code, name]) => ({ code, name }))
    .sort((left, right) => Number(left.code) - Number(right.code));
}

function terrainNameFromCode(map, code) {
  return map.terrain_legend?.[String(code)] ?? `unknown_${code}`;
}

function terrainCodeByName(name) {
  return {
    plain: 0,
    forest: 1,
    wetland: 2,
    rocky: 3,
    water: 4,
  }[name] ?? 0;
}

function isKnownTerrainCode(code) {
  return Object.prototype.hasOwnProperty.call(
    {
      0: true,
      1: true,
      2: true,
      3: true,
      4: true,
    },
    Number(code),
  );
}

function habitatStateNameFromCode(code) {
  return {
    0: "stable",
    1: "bloom",
    2: "flooded",
    3: "parched",
  }[code] ?? "stable";
}

function ecologyStateNameFromCode(code) {
  return {
    0: "stable",
    1: "lush",
    2: "recovering",
    3: "depleted",
  }[code] ?? "stable";
}

function hazardTypeNameFromCode(code) {
  return {
    0: "none",
    1: "exposure",
    2: "instability",
  }[code] ?? "none";
}

function hydrologyReasonNameFromCode(code) {
  return {
    0: "none",
    1: "adjacent_water",
    2: "wetland",
    3: "flooded",
  }[code] ?? "none";
}

function hydrologySupportFromCode(code) {
  const normalized = Number(code ?? 0);
  return {
    adjacentToWater: Boolean(normalized & HYDROLOGY_SUPPORT_BITS.adjacent_to_water),
    wetland: Boolean(normalized & HYDROLOGY_SUPPORT_BITS.wetland),
    flooded: Boolean(normalized & HYDROLOGY_SUPPORT_BITS.flooded),
  };
}

function enumLabel(domain, value, fallback = UNKNOWN_LABEL) {
  if (value == null || value === "") return fallback;
  const key = String(value);
  const labels = VIEWER_DISPLAY_LABELS[domain] ?? {};
  return labels[key] ?? titleCase(key);
}

function trophicRoleLabel(value) {
  return enumLabel("trophic_role", value, "Unknown role");
}

function meatModeLabel(value) {
  return enumLabel("meat_mode", value ?? "none", "Unknown diet mode");
}

function waterAccessLabel(value) {
  return enumLabel("water_access_reason", value ?? "none", "Unknown water state");
}

function hydrologyLabel(value) {
  return enumLabel("water_access_reason", value ?? "none", "Unknown hydrology");
}

function softRefugeLabel(value) {
  return enumLabel("soft_refuge_reason", value ?? "none", "Unknown refuge state");
}

function hazardLabel(value) {
  return enumLabel("hazard_type", value ?? "none", "Unknown hazard");
}

function damageSourceLabel(value) {
  return enumLabel("damage_source", value ?? "none", "Unknown damage source");
}

function foodSourceLabel(value) {
  return enumLabel("food_source", value ?? "food", "Food");
}

function reproductionModeLabel(value) {
  return enumLabel("reproduction_mode", value ?? "reproduction", "Reproduction");
}

function deathCauseLabel(value) {
  return enumLabel("death_cause", value ?? "unknown", "Unknown death cause");
}

function deathStatusLabel(deathEvent, current, tick) {
  if (!deathEvent) {
    return current ? VIEWER_EMPTY_LABELS.still_alive : VIEWER_EMPTY_LABELS.no_recorded_death;
  }
  const label = `${deathCauseLabel(deathEvent.data?.cause)} at tick ${deathEvent.tick}`;
  if (Number(deathEvent.tick) > Number(tick)) {
    return `${VIEWER_EMPTY_LABELS.still_alive} at this tick; later ${label}`;
  }
  return label;
}

function terrainLabel(value) {
  return value ? titleCase(value) : `${UNKNOWN_LABEL} terrain`;
}

function habitatLabel(value) {
  return value ? enumLabel("habitat_state", value, `${UNKNOWN_LABEL} habitat`) : `${UNKNOWN_LABEL} habitat`;
}

function ecologyLabel(value) {
  return value ? enumLabel("ecology_state", value, `${UNKNOWN_LABEL} ecology`) : `${UNKNOWN_LABEL} ecology`;
}

function buildEncodingMap(fields) {
  return Object.fromEntries(fields.map((field, index) => [field, index]));
}

function decodeAgent(encoded) {
  const fieldMap = state.agentEncoding;
  return {
    agentId: encoded[fieldMap.agent_id],
    x: encoded[fieldMap.x],
    y: encoded[fieldMap.y],
    energy: encoded[fieldMap.energy],
    energyRatio: fieldMap.energy_ratio != null ? encoded[fieldMap.energy_ratio] : 0,
    hydration: encoded[fieldMap.hydration],
    hydrationRatio: fieldMap.hydration_ratio != null ? encoded[fieldMap.hydration_ratio] : 0,
    health: encoded[fieldMap.health],
    healthRatio: encoded[fieldMap.health_ratio],
    injuryLoad: encoded[fieldMap.injury_load],
    age: encoded[fieldMap.age],
    energyModifier: encoded[fieldMap.energy_modifier],
    hydrationModifier: encoded[fieldMap.hydration_modifier],
    tileVegetation:
      fieldMap.tile_vegetation != null ? encoded[fieldMap.tile_vegetation] : 0,
    tileRecoveryDebt:
      fieldMap.tile_recovery_debt != null ? encoded[fieldMap.tile_recovery_debt] : 0,
    reproductionReady:
      fieldMap.reproduction_ready != null ? Boolean(encoded[fieldMap.reproduction_ready]) : false,
    trophicRole: fieldMap.trophic_role != null ? encoded[fieldMap.trophic_role] : "herbivore",
    meatMode: fieldMap.meat_mode != null ? encoded[fieldMap.meat_mode] : "none",
    lastDamageSource:
      fieldMap.last_damage_source != null ? encoded[fieldMap.last_damage_source] : "none",
    waterAccessReason: encoded[fieldMap.water_access_reason],
    softRefugeReason: fieldMap.soft_refuge_reason != null ? encoded[fieldMap.soft_refuge_reason] : "none",
    hydrologySupportCode:
      fieldMap.hydrology_support_code != null ? encoded[fieldMap.hydrology_support_code] : 0,
    refugeScore: fieldMap.refuge_score != null ? encoded[fieldMap.refuge_score] : 0,
    matchedDietRatio:
      fieldMap.matched_diet_ratio != null ? encoded[fieldMap.matched_diet_ratio] : 0,
    ecotypeId: fieldMap.ecotype_id != null ? encoded[fieldMap.ecotype_id] : null,
    speciesId: encoded[fieldMap.species_id],
  };
}

function syncDebugState() {
  const payload = state.payload;
  if (!payload) {
    window.__viewerDebug = { loaded: false, status: elements.loadStatus.textContent };
    return;
  }

  const map = payload.viewer.map;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);
  window.__viewerDebug = {
    loaded: true,
    status: elements.loadStatus.textContent,
    currentFrameIndex: state.currentFrameIndex,
    selectedAgentId: state.selectedAgentId,
    selectedSpeciesId: state.selectedSpeciesId,
    overlayMode: state.overlayMode,
    overlayOpacityDefaults: Object.fromEntries(
      Object.keys(OVERLAY_CONTRACTS).map((mode) => [mode, defaultOverlayOpacityForMode(mode)]),
    ),
    decisionOverlayMode: state.decisionOverlayMode,
    eventLensMode: state.eventLensMode,
    overlayOpacity: state.overlayOpacity,
    blendTerrain: state.blendTerrain,
    activeCameraPreset: state.activeCameraPreset,
    activeEventCluster: state.activeEventCluster ? { ...state.activeEventCluster } : null,
    presentationMode: state.presentationMode,
    episodeFilters: { ...state.episodeFilters },
    shareUrl: state.lastShareUrl ?? window.location.href,
    snapshotStats: {
      ...state.snapshotStats,
      cacheSize: state.snapshotCache.size,
    },
    viewerStateStorageVersion: VIEWER_STATE_STORAGE_VERSION,
    viewerPreferences: serializeViewerPreferences(state),
    viewerUrlState: serializeViewerUrlState(state),
    comparison: {
      baselineFrameIndex: state.comparisonBaselineFrameIndex,
      baselineTick: payload.viewer.frames[state.comparisonBaselineFrameIndex]?.tick ?? null,
    },
    hoveredTile: state.hoveredTile ? { ...state.hoveredTile } : null,
    explainedTile: state.explainedTile ? { ...state.explainedTile } : null,
    mapView: { ...state.mapView },
    loadGeneration: state.loadGeneration,
    replaySource: state.replaySource,
    activeDetailView: state.activeDetailView,
    tileSize,
    offset,
    frame: payload.viewer.frames[state.currentFrameIndex],
    summary: payload.summary,
    significantEventTicks: state.significantEventTicks.map((entry) => entry.tick),
    eventBookmarks: state.eventBookmarks.map((entry) => ({ ...entry })),
    episodePlayback: state.episodePlayback ? { ...state.episodePlayback } : null,
    lastStoryboardExport: state.lastStoryboardExport ? { ...state.lastStoryboardExport } : null,
    replayModel: { ...(state.replayModel?.cacheStats ?? {}) },
    visualStats: { ...state.visualStats },
  };
}
