import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { buildReplayModel } from "./replay_model.mjs";
import { ensureViewerServer, stopViewerServer } from "./smoke_server.mjs";

const replayPath = process.env.REPLAY_PATH ?? "../output/sim-runs/species-check.json";
const viewerUrl = `http://127.0.0.1:4173/viewer/index.html?replay=${encodeURIComponent(replayPath)}`;
const viewerDir = path.dirname(fileURLToPath(import.meta.url));

assertReplayModelNestedEventIndex();

const viewerServer = await ensureViewerServer();
let browser;

function assertReplayModelNestedEventIndex() {
  const localReplayPath = resolveLocalReplayPath(replayPath);
  if (!localReplayPath || !fs.existsSync(localReplayPath)) return;

  const payload = JSON.parse(fs.readFileSync(localReplayPath, "utf8"));
  const model = buildReplayModel(payload, (encoded) => decodeSmokeAgent(payload, encoded));
  const nestedSourceEvent = (payload.events ?? []).find((event) => {
    const data = event.data ?? {};
    return (
      data.source_breakdown?.some((entry) => Number.isFinite(Number(entry.source_agent_id))) ||
      data.deposit_breakdown?.some((entry) => Number.isFinite(Number(entry.source_agent_id))) ||
      data.tile_source_breakdown_after?.some((entry) => Number.isFinite(Number(entry.source_agent_id))) ||
      data.tile_fresh_kill_source_breakdown_after?.some((entry) => Number.isFinite(Number(entry.source_agent_id)))
    );
  });
  if (!nestedSourceEvent) return;

  const sourceId = firstNestedSourceAgentId(nestedSourceEvent);
  const indexedEvents = model.agentEventsById.get(sourceId) ?? [];
  const indexed = indexedEvents.some(
    (event) =>
      event.tick === nestedSourceEvent.tick &&
      event.type === nestedSourceEvent.type &&
      event.agent_id === nestedSourceEvent.agent_id,
  );
  if (!indexed) {
    throw new Error(`Expected nested source agent ${sourceId} to index ${nestedSourceEvent.type} at tick ${nestedSourceEvent.tick}.`);
  }
}

function resolveLocalReplayPath(value) {
  if (/^https?:\/\//i.test(value)) return null;
  if (path.isAbsolute(value)) return value;
  return path.resolve(viewerDir, value);
}

function firstNestedSourceAgentId(event) {
  const data = event.data ?? {};
  for (const key of [
    "source_breakdown",
    "deposit_breakdown",
    "tile_source_breakdown_after",
    "tile_fresh_kill_source_breakdown_after",
  ]) {
    for (const entry of data[key] ?? []) {
      const agentId = Number(entry.source_agent_id);
      if (Number.isFinite(agentId)) return agentId;
    }
  }
  return null;
}

function decodeSmokeAgent(payload, encoded) {
  const fields = Object.fromEntries((payload.viewer?.agent_encoding ?? []).map((key, index) => [key, index]));
  return {
    agentId: Number(encoded[fields.agent_id]),
    x: Number(encoded[fields.x]),
    y: Number(encoded[fields.y]),
  };
}

async function assertNoControlClipping(page, label) {
  const issues = await page.evaluate(() => {
    const visible = (node) => {
      const style = window.getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        rect.width > 0 &&
        rect.height > 0
      );
    };
    const problems = [];
    const viewportWidth = window.innerWidth;
    const overflow = document.documentElement.scrollWidth - viewportWidth;
    if (overflow > 1) {
      problems.push(`document overflows horizontally by ${Math.round(overflow)}px`);
    }
    for (const node of document.querySelectorAll(
      [
        "#canvas-panel",
        ".map-legend",
        "#terrain-legend",
        "#glyph-legend",
        "#map-control-strip",
        "#map-navigation",
        "#tile-inspector",
        "#story-timeline",
      ].join(","),
    )) {
      if (!visible(node)) continue;
      const rect = node.getBoundingClientRect();
      if (rect.left < -1 || rect.right > viewportWidth + 1) {
        problems.push(`${node.id || node.className} crosses viewport (${Math.round(rect.left)}-${Math.round(rect.right)})`);
      }
    }
    for (const selector of ["#map-control-strip", "#glyph-legend", ".terrain-key-grid", "#tile-inspector"]) {
      const parent = document.querySelector(selector);
      if (!parent || !visible(parent)) continue;
      const parentRect = parent.getBoundingClientRect();
      for (const child of parent.children) {
        if (!visible(child)) continue;
        const childRect = child.getBoundingClientRect();
        if (childRect.left < parentRect.left - 1 || childRect.right > parentRect.right + 1) {
          problems.push(`${selector} child ${child.id || child.className || child.tagName} clips horizontally`);
        }
      }
    }
    return problems;
  });
  if (issues.length > 0) {
    throw new Error(`Viewer layout clipping detected in ${label}: ${issues.join("; ")}`);
  }
}

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  await page.goto(viewerUrl, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.__viewerDebug?.loaded === true);
  await page.waitForFunction(() => window.__viewerDebug?.frame?.agents?.length > 0);
  await page.waitForFunction(() => {
    return document.querySelectorAll("#species-list [data-species-id]").length > 0;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#population-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    const model = window.__viewerDebug?.replayModel;
    return (
      model &&
      model.decodedFrameCount === window.__viewerDebug?.summary?.ticks_executed &&
      model.agentEventIndexSize > 0 &&
      model.agentPositionIndexSize > 0 &&
      model.episodeCount > 0
    );
  });
  await page.waitForFunction(() => {
    return (
      document.querySelector('[data-detail-view="overview"]') &&
      document.querySelector('[data-detail-view="agent"]') &&
      document.querySelector('[data-detail-view="species"]') &&
      document.querySelector('[data-detail-view="events"]')
    );
  });
  await page.locator('[data-detail-view="events"]').click();
  await page.waitForFunction(() => {
    const panel = document.querySelector('[data-detail-panel="events"]');
    return panel && !panel.classList.contains("hidden");
  });
  await page.waitForFunction(() => {
    return document.querySelectorAll("#episode-list [data-episode-tick]").length > 0;
  });
  const episodeTick = await page.evaluate(() => {
    return Number(document.querySelector("#episode-list [data-episode-tick]")?.dataset.episodeTick);
  });
  const episodeLocator = page.locator(`#episode-list [data-episode-tick="${episodeTick}"]`);
  if ((await episodeLocator.count()) === 0) {
    throw new Error(`Expected an event episode at tick ${episodeTick}.`);
  }
  await episodeLocator.first().click();
  await page.waitForFunction((targetTick) => {
    return window.__viewerDebug?.frame?.tick === targetTick;
  }, episodeTick);
  await page.waitForFunction(() => {
    const inspector = document.querySelector("#episode-inspector");
    return (
      inspector &&
      inspector.textContent.includes("Involved Agents") &&
      inspector.textContent.includes("Event Ledger") &&
      inspector.querySelectorAll("[data-episode-lens]").length >= 2 &&
      inspector.querySelectorAll("[data-episode-play]").length === 1 &&
      inspector.querySelectorAll("[data-episode-agent]").length > 0
    );
  });
  const causalInspector = await page.evaluate(() => {
    const roles = [...document.querySelectorAll("#episode-inspector .episode-agent-pill span")].map((node) =>
      node.textContent.trim().toLowerCase(),
    );
    const ledgerTitles = [...document.querySelectorAll("#episode-inspector .episode-ledger-item span")].map((node) =>
      node.textContent.trim(),
    );
    return { roles, ledgerTitles };
  });
  if (!causalInspector.roles.includes("attacker")) {
    throw new Error(`Expected attack episode roles to include attacker; saw ${causalInspector.roles.join(", ")}.`);
  }
  if (!causalInspector.roles.some((role) => role === "target" || role === "killed")) {
    throw new Error(`Expected attack episode roles to include target/killed; saw ${causalInspector.roles.join(", ")}.`);
  }
  if (causalInspector.roles.some((role) => ["actor", "mover", "damaged", "healed", "drinker"].includes(role))) {
    throw new Error(`Expected primary episode roles without support noise; saw ${causalInspector.roles.join(", ")}.`);
  }
  if (!causalInspector.ledgerTitles.includes("Support Events")) {
    throw new Error(`Expected grouped support event summary; saw ${causalInspector.ledgerTitles.join(", ")}.`);
  }
  if (causalInspector.ledgerTitles.includes("Movement")) {
    throw new Error("Expected primary episode ledger to group movement instead of listing it as causal detail.");
  }
  await page.locator("#episode-inspector [data-episode-play]").first().click();
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.episodePlayback?.scope?.primaryEventCount > 0 &&
      window.__viewerDebug?.episodePlayback?.scope?.agentIds?.length >= 2 &&
      window.__viewerDebug?.eventLensMode === "causal" &&
      document.querySelector("#event-lens-summary")?.textContent?.includes("episode") &&
      document.querySelector("#episode-inspector")?.textContent?.includes("Playing causal window")
    );
  });
  await page.locator("#episode-inspector [data-episode-play]").first().click();
  await page.waitForFunction(() => window.__viewerDebug?.episodePlayback === null);
  await page.locator('#episode-inspector [data-episode-lens="delta"]').first().click();
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.eventLensMode === "delta" &&
      document.querySelector("#event-lens-summary")?.textContent?.includes("delta")
    );
  });
  await page.locator('#episode-inspector [data-episode-lens="causal"]').first().click();
  await page.waitForFunction(() => window.__viewerDebug?.eventLensMode === "causal");
  await page.locator("#bookmark-current-event").click();
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.eventBookmarks?.length === 1 &&
      document.querySelector("#bookmark-list")?.textContent?.includes("Tick")
    );
  });
  await page.locator("#export-storyboard").click();
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.lastStoryboardExport?.entryCount > 0 &&
      document.querySelector("#storyboard-export-status")?.textContent?.includes("Exported")
    );
  });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForFunction(() => window.__viewerDebug?.loaded === true);
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.eventBookmarks?.length === 1 &&
      document.querySelector("#bookmark-list")?.textContent?.includes("Tick")
    );
  });
  await page.locator('[data-detail-view="overview"]').click();
  await page.waitForFunction(() => {
    const panel = document.querySelector('[data-detail-panel="overview"]');
    return panel && !panel.classList.contains("hidden");
  });
  await page.waitForFunction(() => {
    return document.querySelector("#habitat-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#hydrology-primary-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#hydrology-support-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#hazard-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#carcass-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#combat-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    return document.querySelector("#ecology-chart svg") !== null;
  });
  await page.waitForFunction(() => {
    const hydrology = document.querySelector("#hydrology-primary-grid");
    return (
      hydrology &&
      hydrology.textContent.includes("Hard Access Tiles") &&
      hydrology.textContent.includes("Primary Adjacent Water") &&
      hydrology.textContent.includes("Primary None")
    );
  });
  await page.waitForFunction(() => {
    const support = document.querySelector("#hydrology-support-grid");
    return (
      support &&
      support.textContent.includes("Shoreline Support") &&
      support.textContent.includes("Flooded Support") &&
      support.textContent.includes("Avg Refuge Score (Forest Tiles)")
    );
  });
  await page.waitForFunction(() => {
    const hazard = document.querySelector("#hazard-grid");
    return (
      hazard &&
      hazard.textContent.includes("Hazardous Tiles") &&
      hazard.textContent.includes("Exposure") &&
      hazard.textContent.includes("Avg Hazard Level")
    );
  });
  await page.waitForFunction(() => {
    const carcass = document.querySelector("#carcass-grid");
    return (
      carcass &&
      carcass.textContent.includes("Carcass Tiles") &&
      carcass.textContent.includes("Avg Freshness") &&
      carcass.textContent.includes("Deposition Events") &&
      carcass.textContent.includes("Consumption Events") &&
      carcass.textContent.includes("Energy Consumed")
    );
  });
  await page.waitForFunction(() => {
    const ecology = document.querySelector("#ecology-grid");
    return (
      ecology &&
      ecology.textContent.includes("Lush") &&
      ecology.textContent.includes("Avg Vegetation") &&
      ecology.textContent.includes("Avg Recovery Debt")
    );
  });
  await page.waitForFunction(() => {
    const legend = document.querySelector("#terrain-legend");
    return (
      legend &&
      legend.textContent.includes("Terrain coverage") &&
      legend.textContent.includes("48 x 32 grid") &&
      legend.textContent.includes("1,536 tiles") &&
      legend.textContent.includes("Wetland") &&
      legend.textContent.includes("Rocky") &&
      legend.querySelectorAll("[data-terrain-key]").length >= 4 &&
      [...legend.querySelectorAll("[data-terrain-share-label]")].some((node) =>
        node.dataset.terrainShareLabel?.includes("%")
      )
    );
  });
  await page.waitForFunction(() => {
    const legend = document.querySelector("#glyph-legend");
    return (
      legend &&
      legend.textContent.includes("Agent forms") &&
      legend.textContent.includes("Vital rings") &&
      legend.textContent.includes("Activity cues") &&
      legend.textContent.includes("Event pulses") &&
      legend.textContent.includes("Herbivore") &&
      legend.textContent.includes("Omnivore") &&
      legend.textContent.includes("Carnivore") &&
      legend.textContent.includes("Energy") &&
      legend.textContent.includes("Water") &&
      legend.textContent.includes("Health") &&
      legend.textContent.includes("Ready") &&
      legend.textContent.includes("Diet match") &&
      legend.textContent.includes("Meat stroke") &&
      legend.textContent.includes("Event")
    );
  });
  await page.waitForFunction(() => {
    const controls = document.querySelector("#map-control-strip");
    return (
      controls &&
      controls.textContent.includes("Decision Layer") &&
      controls.textContent.includes("Event Lens") &&
      controls.textContent.includes("Overlay Opacity") &&
      controls.textContent.includes("Blend Terrain") &&
      controls.textContent.includes("Reset View") &&
      controls.textContent.includes("Fit Focus") &&
      document.querySelector("#decision-overlay-mode") &&
      document.querySelector("#event-lens-mode") &&
      document.querySelector("#event-lens-summary") &&
      document.querySelector("#overlay-opacity") &&
      document.querySelector("#blend-terrain-toggle") &&
      document.querySelector("#map-navigation") &&
      document.querySelector("#map-minimap [data-minimap-viewport]")
    );
  });
  await assertNoControlClipping(page, "desktop startup");
  await page.locator("#map-zoom-in").click();
  await page.waitForFunction(() => (window.__viewerDebug?.mapView?.zoom ?? 1) > 1);
  await page.locator("#map-reset-view").click();
  await page.waitForFunction(() => Math.round((window.__viewerDebug?.mapView?.zoom ?? 0) * 100) === 100);
  await page.locator("#decision-overlay-mode").selectOption("recent");
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.decisionOverlayMode === "recent" &&
      window.__viewerDebug?.visualStats?.decisionOverlays > 0
    );
  });
  await page.locator("#decision-overlay-mode").selectOption("reward");
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.decisionOverlayMode === "reward" &&
      window.__viewerDebug?.visualStats?.decisionOverlays > 0
    );
  });
  await page.locator("#decision-overlay-mode").selectOption("selected");
  await page.waitForFunction(() => {
    const canvasPanel = document.querySelector("#canvas-panel");
    const timeline = document.querySelector("#story-timeline");
    return (
      canvasPanel?.classList.contains("glass-panel") &&
      timeline?.classList.contains("glass-panel")
    );
  });
  await page.waitForFunction(() => {
    return window.__viewerDebug?.visualStats?.glyphs > 0;
  });
  await page.waitForFunction(() => {
    return window.__viewerDebug?.visualStats?.clusterHalos > 0;
  });
  await page.waitForFunction(() => {
    return window.__viewerDebug?.visualStats?.terrainBlends > 0;
  });
  await page.waitForFunction(() => {
    return (
      document.querySelectorAll("#map-annotations .map-annotation").length > 0 &&
      [...document.querySelectorAll("#map-annotations [data-map-species-id]")].every((node) =>
        node.getAttribute("aria-label")?.startsWith("Focus ")
      )
    );
  });
  await page.waitForFunction(() => {
    const boxes = [...document.querySelectorAll("#map-annotations .map-annotation")].map((node) =>
      node.getBoundingClientRect()
    );
    return boxes.every((box, index) =>
      boxes.slice(index + 1).every((other) =>
        box.right + 4 < other.left ||
        box.left - 4 > other.right ||
        box.bottom + 4 < other.top ||
        box.top - 4 > other.bottom
      )
    );
  });
  await page.waitForFunction(() => {
    const speciesName = document.querySelector("#species-list .species-display-name");
    const speciesCode = document.querySelector("#species-list .species-code");
    return (
      speciesName &&
      speciesCode &&
      !/^S\d+$/i.test(speciesName.textContent?.trim() ?? "") &&
      /^S\d+$/i.test(speciesCode.textContent?.trim() ?? "")
    );
  });
  await page.locator("#blend-terrain-toggle").setChecked(false);
  await page.waitForFunction(() => window.__viewerDebug?.blendTerrain === false);
  await page.locator("#blend-terrain-toggle").setChecked(true);
  await page.locator("#overlay-opacity").evaluate((node) => {
    node.value = "58";
    node.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.blendTerrain === true &&
      Math.round((window.__viewerDebug?.overlayOpacity ?? 0) * 100) === 58
    );
  });
  await page.locator("#overlay-mode").selectOption("hazard");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Hazard");
  });
  await page.waitForFunction(() => {
    return document.querySelectorAll("#event-strip [data-event-tick]").length > 0;
  });
  await page.locator("#jump-next-event").click();
  await page.waitForFunction(() => {
    return window.__viewerDebug?.currentFrameIndex > 0;
  });
  await page.locator("#event-lens-mode").selectOption("causal");
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.eventLensMode === "causal" &&
      window.__viewerDebug?.visualStats?.eventLensMarkers > 0 &&
      document.querySelector("#event-lens-summary")?.textContent?.includes("causal")
    );
  });
  await page.locator("#event-lens-mode").selectOption("delta");
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.eventLensMode === "delta" &&
      window.__viewerDebug?.visualStats?.deltaMarkers > 0 &&
      document.querySelector("#event-lens-summary")?.textContent?.includes("delta")
    );
  });
  await page.locator("#event-lens-mode").selectOption("current");
  await page.waitForFunction(() => window.__viewerDebug?.eventLensMode === "current");
  await page.waitForFunction(() => {
    const summary = document.querySelector("#event-summary");
    return summary && /tick/i.test(summary.textContent ?? "");
  });

  await page.locator("#canvas-host").scrollIntoViewIfNeeded();
  let canvasBox = await page.locator("#canvas-host").boundingBox();
  if (!canvasBox) {
    throw new Error("Canvas host bounding box was not available.");
  }
  await page.locator("#canvas-host").focus();
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(() => window.__viewerDebug?.hoveredTile !== null);
  await page.mouse.move(canvasBox.x + canvasBox.width / 2, canvasBox.y + canvasBox.height / 2);
  await page.waitForFunction(() => {
    const inspector = document.querySelector("#tile-inspector");
    return (
      inspector &&
      inspector.textContent.includes("Tile") &&
      inspector.textContent.includes("Terrain") &&
      inspector.textContent.includes("Hydrology") &&
      inspector.textContent.includes("Hazard") &&
      inspector.textContent.includes("Ecology")
    );
  });
  await page.locator("#canvas-host").scrollIntoViewIfNeeded();
  const debugState = await page.evaluate(() => window.__viewerDebug);
  const selectedTarget =
    debugState.frame.agents.find(
      (agent) => agent[1] > 4 && agent[1] < 44 && agent[2] > 4 && agent[2] < 26,
    ) ??
    debugState.frame.agents.find((agent) => agent[17] === "none" && agent[19] === "none") ??
    debugState.frame.agents[0];
  const [, gridX, gridY] = selectedTarget;
  canvasBox = await page.locator("#canvas-host").boundingBox();
  if (!canvasBox) {
    throw new Error("Canvas host bounding box was not available after tile inspection.");
  }
  const clickX = debugState.offset.x + gridX * debugState.tileSize + debugState.tileSize / 2;
  const clickY = debugState.offset.y + gridY * debugState.tileSize + debugState.tileSize / 2;

  await page.locator("#canvas-host").click({ position: { x: clickX, y: clickY } });
  await page.waitForFunction(() => {
    return window.__viewerDebug?.selectedAgentId !== null && window.__viewerDebug?.explainedTile !== null;
  });
  await page.waitForFunction(() => {
    const explainer = document.querySelector("#tile-explainer");
    return (
      window.__viewerDebug?.explainedTile !== null &&
      explainer &&
      explainer.textContent.includes("Tile Explanation") &&
      explainer.textContent.includes("Agents Here") &&
      explainer.textContent.includes("Recent Local Events")
    );
  });
  await page.waitForFunction(() => {
    return window.__viewerDebug?.activeDetailView === "agent";
  });
  await page.locator("#map-fit-focus").click();
  await page.waitForFunction(() => (window.__viewerDebug?.mapView?.zoom ?? 1) > 1);
  await page.waitForFunction(() => document.querySelector("#map-minimap [data-minimap-viewport]") !== null);
  await page.locator("#map-reset-view").click();
  await page.waitForFunction(() => Math.round((window.__viewerDebug?.mapView?.zoom ?? 0) * 100) === 100);
  await page.waitForFunction(() => {
    const dossier = document.querySelector("#agent-dossier");
    return (
      dossier &&
      !dossier.classList.contains("hidden") &&
      dossier.textContent.includes("Agent Dossier") &&
      dossier.textContent.includes("Decision Trace") &&
      dossier.textContent.includes("Requested") &&
      dossier.textContent.includes("Resolved") &&
      dossier.textContent.includes("Reward Components") &&
      dossier.textContent.includes("Observation Patch") &&
      document.querySelectorAll("#agent-decision-trace [data-decision-tick]").length > 0 &&
      /Current Condition|Last Known Condition/.test(dossier.textContent) &&
      dossier.textContent.includes("Life Timeline") &&
      dossier.textContent.includes("Energy") &&
      dossier.textContent.includes("Hydration") &&
      dossier.textContent.includes("Health")
    );
  });
  await page.waitForFunction(() => {
    const patch = document.querySelector("#agent-observation-patch");
    return (
      patch &&
      patch.textContent.includes("Terrain") &&
      patch.querySelectorAll("[data-observation-cell]").length >= 9
    );
  });
  const agentFacts = await page.evaluate(() => {
    return Object.fromEntries(
      [...document.querySelectorAll("#agent-dossier .agent-fact-grid div")].map((row) => [
        row.querySelector("dt")?.textContent?.trim() ?? "",
        row.querySelector("dd")?.textContent?.trim() ?? "",
      ]),
    );
  });
  const rawDossierLabels = [
    ["Mode", "None"],
    ["Water", "None"],
    ["Hazard", "None"],
    ["Terrain", "-"],
    ["Habitat", "-"],
    ["Death", "-"],
  ];
  for (const [label, disallowed] of rawDossierLabels) {
    if (agentFacts[label] === disallowed) {
      throw new Error(`Expected ${label} to use a human viewer label instead of ${disallowed}.`);
    }
  }
  await page.waitForFunction(() => {
    return document.querySelectorAll("#agent-action-timeline [data-agent-event-tick]").length > 0;
  });
  const agentEventTick = await page.evaluate(() => {
    const currentTick = window.__viewerDebug?.frame?.tick;
    const items = [...document.querySelectorAll("#agent-action-timeline [data-agent-event-tick]")];
    const target = items
      .map((node) => Number(node.dataset.agentEventTick))
      .find((tick) => Number.isFinite(tick) && tick !== currentTick);
    return target ?? Number(items[0]?.dataset.agentEventTick);
  });
  const agentEventLocator = page.locator(
    `#agent-action-timeline [data-agent-event-tick="${agentEventTick}"]`,
  );
  if ((await agentEventLocator.count()) === 0) {
    throw new Error(`Expected an agent event at tick ${agentEventTick}.`);
  }
  await agentEventLocator.first().click();
  await page.waitForFunction((targetTick) => {
    return window.__viewerDebug?.frame?.tick === targetTick;
  }, agentEventTick);
  await page.waitForFunction(() => {
    const inspector = document.querySelector("#inspector-grid");
    return (
      inspector &&
      inspector.textContent.includes("Species") &&
      inspector.textContent.includes("Terrain Here") &&
      inspector.textContent.includes("Parents") &&
      inspector.textContent.includes("Reproductive Group") &&
      inspector.textContent.includes("Reproductive Stage") &&
      inspector.textContent.includes("Reproductive Expression") &&
      inspector.textContent.includes("Trophic Role") &&
      inspector.textContent.includes("Health") &&
      inspector.textContent.includes("Last Damage Source") &&
      inspector.textContent.includes("Hazard Here") &&
      inspector.textContent.includes("Reproductive Signal Here") &&
      inspector.textContent.includes("Carcass Dominant Source") &&
      inspector.textContent.includes("Carcass Source Mix") &&
      inspector.textContent.includes("Water Access Reason") &&
      inspector.textContent.includes("Adjacent To Water") &&
      inspector.textContent.includes("Refuge Score") &&
      inspector.textContent.includes("Hydration Modifier") &&
      inspector.textContent.includes("Habitat State") &&
      inspector.textContent.includes("Ecology State")
    );
  });
  const inspectorFacts = await page.evaluate(() => {
    return Object.fromEntries(
      [...document.querySelectorAll("#inspector-grid div")].map((row) => [
        row.querySelector("dt")?.textContent?.trim() ?? "",
        row.querySelector("dd")?.textContent?.trim() ?? "",
      ]),
    );
  });
  const rawInspectorLabels = [
    ["Water Access Reason", "None"],
    ["Soft Refuge", "None"],
    ["Last Damage Source", "None"],
    ["Hazard Here", "None"],
    ["Terrain Here", "-"],
    ["Habitat State", "-"],
    ["Ecology State", "-"],
  ];
  for (const [label, disallowed] of rawInspectorLabels) {
    if (inspectorFacts[label] === disallowed) {
      throw new Error(`Expected inspector ${label} to use a human label instead of ${disallowed}.`);
    }
  }
  await page.waitForFunction(() => {
    const ecology = document.querySelector("#species-ecology-grid");
    return (
      ecology &&
      ecology.textContent.includes("Alive Now") &&
      ecology.textContent.includes("Shoreline Support Exposure") &&
      ecology.textContent.includes("Avg Refuge Score (Occupied Tiles)") &&
      ecology.textContent.includes("Avg Health") &&
      ecology.textContent.includes("Injury Rate") &&
      ecology.textContent.includes("Hazard Exposure Rate") &&
      ecology.textContent.includes("Attack Attempts") &&
      ecology.textContent.includes("Carcass Consumption") &&
      ecology.textContent.includes("Refuge Exposure Rate") &&
      ecology.textContent.includes("Refuge Exposure") &&
      ecology.textContent.includes("Bloom Occupancy") &&
      ecology.textContent.includes("Parched Exposure") &&
      ecology.textContent.includes("Depleted Exposure")
    );
  });
  await page.waitForFunction(() => {
    const occupancy = document.querySelector("#species-occupancy-bars");
    return (
      occupancy &&
      occupancy.textContent.includes("Wetland") &&
      occupancy.textContent.includes("Rocky") &&
      occupancy.textContent.includes("Water Access")
    );
  });
  await page.locator('[data-detail-view="species"]').click();
  await page.waitForFunction(() => {
    const panel = document.querySelector('[data-detail-panel="species"]');
    return panel && !panel.classList.contains("hidden");
  });
  await page.waitForFunction(() => {
    const tree = document.querySelector("#lineage-tree");
    return (
      tree &&
      tree.textContent.includes("Lineage Tree") &&
      tree.textContent.includes("Replay Taxonomy") &&
      tree.textContent.includes("Extant") &&
      tree.textContent.includes("Extinct") &&
      tree.querySelectorAll("[data-lineage-species-id]").length > 0 &&
      tree.querySelector("[data-lineage-depth]") &&
      Number.isFinite(window.__viewerDebug?.visualStats?.lineageBranches)
    );
  });
  const lineageTarget = await page.evaluate(() => {
    return document.querySelector("#lineage-tree [data-lineage-species-id]")?.dataset.lineageSpeciesId;
  });
  const lineageLocator = page.locator(`#lineage-tree [data-lineage-species-id="${lineageTarget}"]`);
  if ((await lineageLocator.count()) === 0) {
    throw new Error(`Expected lineage tree item for species ${lineageTarget}.`);
  }
  await lineageLocator.first().click();
  await page.waitForFunction((speciesId) => {
    return window.__viewerDebug?.selectedSpeciesId === Number(speciesId);
  }, lineageTarget);

  const frameCount = debugState.summary.ticks_executed;
  const eventFrameIndex = Math.min(3, frameCount - 1);
  await page.locator("#timeline").evaluate((node, targetIndex) => {
    node.value = String(targetIndex);
    node.dispatchEvent(new Event("input", { bubbles: true }));
  }, eventFrameIndex);
  await page.waitForFunction((targetIndex) => {
    return window.__viewerDebug?.currentFrameIndex === targetIndex;
  }, eventFrameIndex);
  await page.waitForFunction(() => {
    return window.__viewerDebug?.visualStats?.eventMarkers > 0;
  });
  await page.waitForFunction(() => {
    return window.__viewerDebug?.visualStats?.trailSegments > 0;
  });

  await page.locator("#timeline").evaluate((node, targetIndex) => {
    node.value = String(targetIndex);
    node.dispatchEvent(new Event("input", { bubbles: true }));
  }, frameCount - 1);

  await page.waitForFunction((targetIndex) => {
    return window.__viewerDebug?.currentFrameIndex === targetIndex;
  }, frameCount - 1);
  await page.waitForFunction(() => {
    const speciesItems = [...document.querySelectorAll("#species-list [data-species-id]")];
    return speciesItems.some((node) => /alive/i.test(node.textContent ?? ""));
  });
  await page.waitForFunction(() => {
    return document.querySelector("#turnover-chart svg") !== null;
  });
  await page.locator("#overlay-mode").selectOption("habitat");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Habitat");
  });
  await page.locator("#overlay-mode").selectOption("hydrology");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Hydrology");
  });
  await page.locator("#overlay-mode").selectOption("shoreline");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Shoreline");
  });
  await page.locator("#overlay-mode").selectOption("refuge");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Refuge");
  });
  await page.locator("#overlay-mode").selectOption("hazard");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Hazard");
  });
  await page.locator("#overlay-mode").selectOption("carcass");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Carcass");
  });
  await page.locator("#overlay-mode").selectOption("reproductive_signal");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Reproductive Signal");
  });
  await page.locator("#overlay-mode").selectOption("trophic");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Trophic");
  });
  await page.locator("#overlay-mode").selectOption("ecology");
  await page.waitForFunction(() => {
    return document.querySelector("#overlay-label")?.textContent?.includes("Ecology");
  });

  await page.screenshot({ path: "output/playwright/viewer-smoke.png", fullPage: true });

  const selectedAgentId = await page.evaluate(() => window.__viewerDebug?.selectedAgentId);
  console.log(`viewer_smoke_ok selected_agent=${selectedAgentId} final_frame=${frameCount - 1}`);
} finally {
  if (browser) {
    await browser.close();
  }
  await stopViewerServer(viewerServer);
}
