import { chromium } from "playwright";

const replayPath = process.env.REPLAY_PATH ?? "../output/sim-runs/species-check.json";
const viewerUrl = `http://127.0.0.1:4173/viewer/index.html?replay=${encodeURIComponent(replayPath)}`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

try {
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
      legend.textContent.includes("Wetland") &&
      legend.textContent.includes("Rocky")
    );
  });

  const debugState = await page.evaluate(() => window.__viewerDebug);
  const firstAgent = debugState.frame.agents[0];
  const [, gridX, gridY] = firstAgent;
  const canvasBox = await page.locator("#canvas-host").boundingBox();
  if (!canvasBox) {
    throw new Error("Canvas host bounding box was not available.");
  }
  const clickX =
    canvasBox.x + debugState.offset.x + gridX * debugState.tileSize + debugState.tileSize / 2;
  const clickY =
    canvasBox.y + debugState.offset.y + gridY * debugState.tileSize + debugState.tileSize / 2;

  await page.mouse.click(clickX, clickY);
  await page.waitForFunction(() => window.__viewerDebug?.selectedAgentId !== null);
  await page.waitForFunction(() => {
    const inspector = document.querySelector("#inspector-grid");
    return (
      inspector &&
      inspector.textContent.includes("Species") &&
      inspector.textContent.includes("Terrain Here") &&
      inspector.textContent.includes("Trophic Role") &&
      inspector.textContent.includes("Health") &&
      inspector.textContent.includes("Last Damage Source") &&
      inspector.textContent.includes("Hazard Here") &&
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

  const frameCount = debugState.summary.ticks_executed;
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
  await browser.close();
}
