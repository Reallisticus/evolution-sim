import * as PIXI from "./vendor/pixi.min.mjs";

const state = {
  payload: null,
  currentFrameIndex: 0,
  selectedAgentId: null,
  selectedSpeciesId: null,
  overlayMode: "terrain",
  playing: false,
  timerId: null,
  pixiApp: null,
  terrainLayer: null,
  agentLayer: null,
  agentEncoding: null,
};

const elements = {
  replayUrl: document.getElementById("replay-url"),
  replayFile: document.getElementById("replay-file"),
  loadUrl: document.getElementById("load-url"),
  loadStatus: document.getElementById("load-status"),
  playToggle: document.getElementById("play-toggle"),
  playbackSpeed: document.getElementById("playback-speed"),
  overlayMode: document.getElementById("overlay-mode"),
  timeline: document.getElementById("timeline"),
  frameLabel: document.getElementById("frame-label"),
  aliveLabel: document.getElementById("alive-label"),
  speciesLabel: document.getElementById("species-label"),
  seasonLabel: document.getElementById("season-label"),
  climateLabel: document.getElementById("climate-label"),
  birthsLabel: document.getElementById("births-label"),
  deathsLabel: document.getElementById("deaths-label"),
  overlayLabel: document.getElementById("overlay-label"),
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
  canvasHost: document.getElementById("canvas-host"),
};

const REQUIRED_AGENT_FIELDS = [
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

const HYDROLOGY_SUPPORT_BITS = {
  adjacent_to_water: 1,
  wetland: 2,
  flooded: 4,
};

await initPixi();
bindEvents();
await bootstrapDefaultReplay();

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

  elements.canvasHost.appendChild(app.canvas);

  state.pixiApp = app;
  state.terrainLayer = new PIXI.Container();
  state.agentLayer = new PIXI.Container();
  app.stage.addChild(state.terrainLayer);
  app.stage.addChild(state.agentLayer);

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
    const text = await file.text();
    loadReplay(JSON.parse(text), file.name);
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
    state.overlayMode = elements.overlayMode.value;
    drawTerrain();
    renderFrame(state.currentFrameIndex);
  });

  elements.speciesList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-species-id]");
    if (!button || !state.payload) return;

    const speciesId = Number(button.dataset.speciesId);
    const frame = state.payload.viewer.frames[state.currentFrameIndex];
    const match = frame.agents
      .map((encoded) => decodeAgent(encoded))
      .find((agent) => agent.speciesId === speciesId);

    state.selectedSpeciesId = speciesId;
    state.selectedAgentId = match?.agentId ?? null;
    renderFrame(state.currentFrameIndex);
  });
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
  state.payload = validateReplayPayload(payload);
  state.selectedAgentId = null;
  state.selectedSpeciesId = null;
  state.currentFrameIndex = 0;
  state.overlayMode = elements.overlayMode.value;
  state.agentEncoding = buildEncodingMap(state.payload.viewer.agent_encoding);

  const frames = state.payload.viewer.frames;
  elements.timeline.disabled = frames.length === 0;
  elements.playToggle.disabled = frames.length === 0;
  elements.timeline.min = "0";
  elements.timeline.max = String(Math.max(frames.length - 1, 0));
  elements.timeline.value = "0";

  populateSummary(state.payload.summary);
  populateTraits(state.payload.summary);
  populateTerrainLegend(state.payload.viewer.map);
  drawTerrain();
  renderFrame(0);
  setStatus(`Loaded ${sourceLabel}`);
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
  const items = terrainEntries(map).map(({ code, name }) => {
    const count = map.terrain_counts?.[name];
    return `
      <span class="terrain-legend-item">
        <i class="swatch" style="background:${terrainCssColor(Number(code))};"></i>
        ${escapeHtml(titleCase(name))}
        ${count != null ? `<small>${count}</small>` : ""}
      </span>
    `;
  });
  elements.terrainLegend.innerHTML = items.join("");
}

function drawTerrain() {
  const app = state.pixiApp;
  const payload = state.payload;
  if (!app || !payload) return;

  const terrainLayer = state.terrainLayer;
  terrainLayer.removeChildren();

  const map = payload.viewer.map;
  const frame = payload.viewer.frames[state.currentFrameIndex] ?? null;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);

  const graphics = new PIXI.Graphics();
  graphics.rect(
    offset.x,
    offset.y,
    map.width * tileSize,
    map.height * tileSize,
  ).fill({ color: 0x08101c });

  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const code = map.terrain_codes[y][x];
      const terrainFill =
        state.overlayMode === "terrain" ? terrainColor(code) : terrainBaseColor(code);
      graphics.rect(
        offset.x + x * tileSize,
        offset.y + y * tileSize,
        tileSize,
        tileSize,
      ).fill({ color: terrainFill });

      if (state.overlayMode !== "terrain" && frame) {
        const isWaterTile = code === terrainCodeByName("water");
        const overlayColor = overlayColorForTile(payload, frame, x, y, code);
        graphics.rect(
          offset.x + x * tileSize,
          offset.y + y * tileSize,
          tileSize,
          tileSize,
        ).fill({
          color: overlayColor,
          alpha:
            isWaterTile &&
            ["hydrology", "shoreline", "refuge", "hazard", "carcass", "ecology"].includes(
              state.overlayMode,
            )
              ? 0.84
              : 0.72,
        });
      }
    }
  }

  terrainLayer.addChild(graphics);
  elements.overlayLabel.textContent = `Overlay: ${titleCase(state.overlayMode)}`;
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
  const decodedAgents = frame.agents.map((encoded) => decodeAgent(encoded));
  const map = payload.viewer.map;
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);

  const agentLayer = state.agentLayer;
  agentLayer.removeChildren();

  const graphics = new PIXI.Graphics();
  for (const agent of decodedAgents) {
    const color = colorForSpecies(agent.speciesId);
    const centerX = offset.x + agent.x * tileSize + tileSize / 2;
    const centerY = offset.y + agent.y * tileSize + tileSize / 2;
    const selectedSpecies = state.selectedSpeciesId === agent.speciesId;

    graphics
      .circle(centerX, centerY, Math.max(tileSize * 0.35, 4))
      .fill({ color, alpha: selectedSpecies ? 1 : 0.92 });

    if (selectedSpecies) {
      graphics.circle(centerX, centerY, Math.max(tileSize * 0.42, 5)).stroke({
        color: 0xf8fafc,
        width: 1,
        alpha: 0.24,
      });
    }

    if (state.selectedAgentId === agent.agentId) {
      graphics.circle(centerX, centerY, Math.max(tileSize * 0.52, 6)).stroke({
        color: 0xf8fafc,
        width: 2,
      });
    }
  }

  agentLayer.addChild(graphics);

  elements.frameLabel.textContent = `Tick ${frame.tick}`;
  elements.aliveLabel.textContent = `Alive ${frame.alive_agents}`;
  elements.speciesLabel.textContent = `Species ${frame.species_counts.length}`;
  elements.seasonLabel.textContent = `Season ${frame.season}`;
  elements.climateLabel.textContent = `Climate ${frame.field_state?.disturbance_type ?? "-"}`;
  elements.birthsLabel.textContent = `Births ${frame.births}`;
  elements.deathsLabel.textContent = `Deaths ${frame.deaths}`;
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
  syncDebugState();
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
      const title = record?.label ?? `Species ${speciesId}`;
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
              ${escapeHtml(title)}
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
  const current = frame.agents
    .map((encoded) => decodeAgent(encoded))
    .find((agent) => agent.agentId === state.selectedAgentId);
  const isAliveNow = Boolean(current);
  const speciesId = current?.speciesId ?? state.selectedSpeciesId;
  const speciesRecord =
    speciesId != null ? payload.viewer.species_catalog[String(speciesId)] : null;
  const ecotypeRecord =
    isAliveNow && current?.ecotypeId != null
      ? payload.viewer.ecotype_catalog?.[String(current.ecotypeId)] ?? null
      : null;
  const liveSpeciesCount =
    speciesId == null
      ? "-"
      : frame.species_counts.find(([candidate]) => candidate === speciesId)?.[1] ?? 0;
  const genome = catalog.genome;
  const currentTerrainName =
    isAliveNow ? terrainNameFromCode(payload.viewer.map, payload.viewer.map.terrain_codes[current.y][current.x]) : null;
  const currentHabitatState =
    isAliveNow ? habitatStateNameFromCode(frame.habitat_state_codes?.[current.y]?.[current.x] ?? 0) : null;
  const currentEcologyState =
    isAliveNow ? ecologyStateNameFromCode(frame.ecology_state_codes?.[current.y]?.[current.x] ?? 0) : null;
  const currentHazardType =
    isAliveNow ? hazardTypeNameFromCode(frame.hazard_type_codes?.[current.y]?.[current.x] ?? 0) : null;
  const currentHazardLevel =
    isAliveNow ? (frame.hazard_level_codes?.[current.y]?.[current.x] ?? 0) / 100 : null;
  const currentCarcassEnergy =
    isAliveNow ? (frame.carcass_energy_codes?.[current.y]?.[current.x] ?? 0) / 100 : null;
  const waterAccessReason = isAliveNow ? current.waterAccessReason ?? "none" : "none";
  const softRefugeReason = isAliveNow ? current.softRefugeReason ?? "none" : "none";
  const hydrologySupport = hydrologySupportFromCode(
    isAliveNow ? current.hydrologySupportCode ?? 0 : 0,
  );

  const entries = [
    ["Agent", state.selectedAgentId],
    ["Species", speciesRecord?.label ?? speciesId ?? "-"],
    ["Ecotype", isAliveNow ? ecotypeRecord?.label ?? current?.ecotypeId ?? "-" : "-"],
    ["Species Members", liveSpeciesCount],
    ["Species Peak", speciesRecord?.peak_members ?? "-"],
    [
      "Species Lineages",
      speciesRecord?.lineages?.length ? speciesRecord.lineages.join(", ") : "-",
    ],
    ["Lineage", catalog.lineage_id],
    ["Parent", catalog.parent_id ?? "root"],
    ["Alive Now", isAliveNow ? "yes" : "no"],
    ["Position", isAliveNow ? `${current.x}, ${current.y}` : "-"],
    ["Terrain Here", currentTerrainName ? titleCase(currentTerrainName) : "-"],
    ["Has Water Access", isAliveNow ? (waterAccessReason !== "none" ? "yes" : "no") : "-"],
    ["Water Access Reason", isAliveNow ? titleCase(waterAccessReason) : "-"],
    ["Adjacent To Water", isAliveNow ? (hydrologySupport.adjacentToWater ? "yes" : "no") : "-"],
    ["Wetland Substrate", isAliveNow ? (hydrologySupport.wetland ? "yes" : "no") : "-"],
    ["Flooded Support", isAliveNow ? (hydrologySupport.flooded ? "yes" : "no") : "-"],
    ["Soft Refuge", isAliveNow ? titleCase(softRefugeReason) : "-"],
    ["Refuge Score", isAliveNow ? formatPercent(current.refugeScore ?? 0) : "-"],
    ["Trophic Role", isAliveNow ? titleCase(current.trophicRole) : "-"],
    ["Health", isAliveNow ? roundValue(current.health) : "-"],
    ["Health Ratio", isAliveNow ? formatPercent(current.healthRatio ?? 0) : "-"],
    ["Injury Load", isAliveNow ? formatPercent(current.injuryLoad ?? 0) : "-"],
    ["Last Damage Source", isAliveNow ? titleCase(current.lastDamageSource ?? "none") : "-"],
    ["Hazard Here", currentHazardType ? titleCase(currentHazardType) : "-"],
    ["Hazard Level", isAliveNow ? formatPercent(currentHazardLevel ?? 0) : "-"],
    ["Carcass Here", isAliveNow ? roundValue(currentCarcassEnergy ?? 0) : "-"],
    ["Habitat State", currentHabitatState ? titleCase(currentHabitatState) : "-"],
    ["Ecology State", currentEcologyState ? titleCase(currentEcologyState) : "-"],
    ["Energy Modifier", isAliveNow ? roundValue(current.energyModifier) : "-"],
    ["Hydration Modifier", isAliveNow ? roundValue(current.hydrationModifier) : "-"],
    ["Energy", isAliveNow ? roundValue(current.energy) : "-"],
    ["Hydration", isAliveNow ? roundValue(current.hydration) : "-"],
    ["Age", isAliveNow ? current.age : "-"],
    [
      "Fertility Here",
      isAliveNow
        ? roundValue(effectiveFieldValue(payload, frame, "fertility", current.x, current.y))
        : "-",
    ],
    [
      "Moisture Here",
      isAliveNow
        ? roundValue(effectiveFieldValue(payload, frame, "moisture", current.x, current.y))
        : "-",
    ],
    [
      "Heat Here",
      isAliveNow
        ? roundValue(effectiveFieldValue(payload, frame, "heat", current.x, current.y))
        : "-",
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
    ["Species", record?.label ?? speciesKey],
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
          label: `${record?.label ?? speciesKey} population`,
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

function renderCollapseEvents(frame, analytics) {
  const events = analytics?.collapse_events ?? [];
  const visibleEvents = events.filter((event) => event.tick <= frame.tick).slice(-12).reverse();
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
      const record = state.payload.viewer.species_catalog[String(event.species_id)];
      const classes = [
        "event-item",
        event.tick === frame.tick ? "current" : "",
        state.selectedSpeciesId === event.species_id ? "selected" : "",
      ]
        .filter(Boolean)
        .join(" ");
      return `
        <div class="${classes}">
          <div class="event-type">${escapeHtml(event.type)}</div>
          <div>${escapeHtml(record?.label ?? `Species ${event.species_id}`)}</div>
          <div class="event-meta">
            Tick ${event.tick} - peak ${event.peak} - current ${event.current}
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

  const map = payload.viewer.map;
  const frame = payload.viewer.frames[state.currentFrameIndex];
  const tileSize = getTileSize(map.width, map.height);
  const offset = getMapOffset(map.width, map.height, tileSize);
  const local = event.global;
  const gridX = Math.floor((local.x - offset.x) / tileSize);
  const gridY = Math.floor((local.y - offset.y) / tileSize);

  if (gridX < 0 || gridY < 0 || gridX >= map.width || gridY >= map.height) {
    return;
  }

  const hit = frame.agents
    .map((encoded) => decodeAgent(encoded))
    .find((agent) => agent.x === gridX && agent.y === gridY);

  if (!hit) {
    state.selectedAgentId = null;
    state.selectedSpeciesId = null;
    renderFrame(state.currentFrameIndex);
    return;
  }

  state.selectedAgentId = hit.agentId;
  state.selectedSpeciesId = hit.speciesId;
  renderFrame(state.currentFrameIndex);
}

function startPlayback() {
  if (!state.payload || state.playing) return;
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

function stopPlayback() {
  state.playing = false;
  elements.playToggle.textContent = "Play";
  if (state.timerId != null) {
    window.clearTimeout(state.timerId);
    state.timerId = null;
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
  return palette[code] ?? palette[0];
}

function terrainBaseColor(code) {
  const palette = {
    0: 0x253028,
    1: 0x183626,
    2: 0x314b34,
    3: 0x403732,
    4: 0x143262,
  };
  return palette[code] ?? palette[0];
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

function getTileSize(width, height) {
  const hostWidth = elements.canvasHost.clientWidth;
  const hostHeight = elements.canvasHost.clientHeight;
  return Math.max(8, Math.floor(Math.min(hostWidth / width, hostHeight / height)));
}

function getMapOffset(width, height, tileSize) {
  return {
    x: Math.floor((elements.canvasHost.clientWidth - width * tileSize) / 2),
    y: Math.floor((elements.canvasHost.clientHeight - height * tileSize) / 2),
  };
}

function setStatus(message) {
  elements.loadStatus.textContent = message;
  syncDebugState();
}

function overlayColorForTile(payload, frame, x, y, terrainCode) {
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
  return map.terrain_legend?.[String(code)] ?? "plain";
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

function titleCase(value) {
  return String(value)
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function blendColor(start, end, ratio) {
  const clamped = clamp01(ratio);
  const startRed = (start >> 16) & 0xff;
  const startGreen = (start >> 8) & 0xff;
  const startBlue = start & 0xff;
  const endRed = (end >> 16) & 0xff;
  const endGreen = (end >> 8) & 0xff;
  const endBlue = end & 0xff;
  const red = Math.round(startRed + (endRed - startRed) * clamped);
  const green = Math.round(startGreen + (endGreen - startGreen) * clamped);
  const blue = Math.round(startBlue + (endBlue - startBlue) * clamped);
  return (red << 16) + (green << 8) + blue;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function validateReplayPayload(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("Replay payload must be a JSON object.");
  }
  if (!payload.summary || typeof payload.summary !== "object") {
    throw new Error("Replay payload is missing summary data.");
  }
  if (!payload.viewer || typeof payload.viewer !== "object") {
    throw new Error("Replay payload is missing viewer data.");
  }

  const { viewer } = payload;
  if (!Array.isArray(viewer.frames)) {
    throw new Error("Replay viewer.frames must be an array.");
  }
  if (!viewer.map || !Array.isArray(viewer.map.terrain_codes)) {
    throw new Error("Replay viewer.map is missing terrain codes.");
  }
  if (!Array.isArray(viewer.agent_encoding)) {
    throw new Error("Replay viewer.agent_encoding must be an array.");
  }

  for (const field of REQUIRED_AGENT_FIELDS) {
    if (!viewer.agent_encoding.includes(field)) {
      throw new Error(`Replay agent encoding is missing required field ${field}.`);
    }
  }
  if (!viewer.agent_catalog || typeof viewer.agent_catalog !== "object") {
    throw new Error("Replay viewer.agent_catalog must be an object.");
  }
  if (!viewer.species_catalog || typeof viewer.species_catalog !== "object") {
    throw new Error("Replay viewer.species_catalog must be an object.");
  }

  return payload;
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
    hydration: encoded[fieldMap.hydration],
    health: encoded[fieldMap.health],
    healthRatio: encoded[fieldMap.health_ratio],
    injuryLoad: encoded[fieldMap.injury_load],
    age: encoded[fieldMap.age],
    energyModifier: encoded[fieldMap.energy_modifier],
    hydrationModifier: encoded[fieldMap.hydration_modifier],
    trophicRole: fieldMap.trophic_role != null ? encoded[fieldMap.trophic_role] : "herbivore",
    lastDamageSource:
      fieldMap.last_damage_source != null ? encoded[fieldMap.last_damage_source] : "none",
    waterAccessReason: encoded[fieldMap.water_access_reason],
    softRefugeReason: fieldMap.soft_refuge_reason != null ? encoded[fieldMap.soft_refuge_reason] : "none",
    hydrologySupportCode:
      fieldMap.hydrology_support_code != null ? encoded[fieldMap.hydrology_support_code] : 0,
    refugeScore: fieldMap.refuge_score != null ? encoded[fieldMap.refuge_score] : 0,
    ecotypeId: fieldMap.ecotype_id != null ? encoded[fieldMap.ecotype_id] : null,
    speciesId: encoded[fieldMap.species_id],
  };
}

function formatValue(value, allowFloat = false) {
  if (value == null) return "-";
  if (typeof value === "number" && allowFloat) {
    return Number.isInteger(value) ? String(value) : String(roundValue(value));
  }
  return String(value);
}

function formatPercent(value) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function formatClimateField(value) {
  if (value == null) return "-";
  if (typeof value === "number") return String(roundValue(value));
  return String(value);
}

function roundValue(value) {
  return Math.round(value * 10000) / 10000;
}

function hslToRgb(hue, saturation, lightness) {
  if (saturation === 0) {
    const value = Math.round(lightness * 255);
    return [value, value, value];
  }

  const q =
    lightness < 0.5
      ? lightness * (1 + saturation)
      : lightness + saturation - lightness * saturation;
  const p = 2 * lightness - q;
  const convert = (channel) => {
    let t = channel;
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };

  return [
    Math.round(convert(hue + 1 / 3) * 255),
    Math.round(convert(hue) * 255),
    Math.round(convert(hue - 1 / 3) * 255),
  ];
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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
    tileSize,
    offset,
    frame: payload.viewer.frames[state.currentFrameIndex],
    summary: payload.summary,
  };
}
