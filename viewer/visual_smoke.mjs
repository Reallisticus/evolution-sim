import { mkdir } from "node:fs/promises";
import { chromium } from "playwright";
import { ensureViewerServer, stopViewerServer } from "./smoke_server.mjs";

const replayPath = process.env.REPLAY_PATH ?? "../output/sim-runs/species-check.json";
const outputDir = process.env.VIEWER_VISUAL_OUTPUT ?? "output/playwright/viewer-visual";
const viewerUrl = `http://127.0.0.1:4173/viewer/index.html?replay=${encodeURIComponent(replayPath)}`;

const viewports = [
  { label: "desktop", viewport: { width: 1440, height: 960 } },
  { label: "mobile", viewport: { width: 390, height: 844 } },
];

const viewerServer = await ensureViewerServer();
let browser;

async function waitForReplay(page) {
  await page.goto(viewerUrl, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.__viewerDebug?.loaded === true);
  await page.waitForFunction(() => window.__viewerDebug?.frame?.agents?.length > 0);
  await page.waitForFunction(() => document.querySelector("#map-minimap [data-minimap-viewport]"));
  await page.waitForFunction(() => document.querySelectorAll("#species-list [data-species-id]").length > 0);
}

async function assertNoHorizontalClipping(page, label) {
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
      problems.push(`document horizontal overflow ${Math.round(overflow)}px`);
    }
    for (const selector of [
      "#canvas-panel",
      ".map-legend",
      "#terrain-legend",
      "#glyph-legend",
      "#map-control-strip",
      "#map-navigation",
      "#tile-inspector",
      "#story-timeline",
      "#story-panel",
    ]) {
      for (const node of document.querySelectorAll(selector)) {
        if (!visible(node)) continue;
        const rect = node.getBoundingClientRect();
        if (rect.left < -1 || rect.right > viewportWidth + 1) {
          problems.push(`${selector} ${Math.round(rect.left)}-${Math.round(rect.right)}`);
        }
      }
    }
    return problems;
  });
  if (issues.length > 0) {
    throw new Error(`${label} clipping: ${issues.join("; ")}`);
  }
}

async function selectFirstAgent(page) {
  const speciesTab = page.locator('[data-detail-view="species"]');
  if ((await speciesTab.count()) > 0) {
    await speciesTab.click();
    await page.waitForFunction(() => {
      const panel = document.querySelector('[data-detail-panel="species"]');
      return panel && !panel.classList.contains("hidden");
    });
  }

  const storySpeciesButton = page.locator("#focus-cards [data-story-species-id]").first();
  if ((await storySpeciesButton.count()) > 0) {
    await storySpeciesButton.click();
    await page.waitForFunction(() => window.__viewerDebug?.selectedAgentId !== null);
    await page.locator("#map-fit-focus").click();
    await page.waitForFunction(() => (window.__viewerDebug?.mapView?.zoom ?? 1) > 1);
    return;
  }

  const speciesButton = page.locator("#species-list [data-species-id]").first();
  if ((await speciesButton.count()) > 0) {
    await speciesButton.click();
    await page.waitForFunction(() => window.__viewerDebug?.selectedAgentId !== null);
    await page.locator("#map-fit-focus").click();
    await page.waitForFunction(() => (window.__viewerDebug?.mapView?.zoom ?? 1) > 1);
    return;
  }

  const target = await page.evaluate(() => {
    const debug = window.__viewerDebug;
    const agent = debug?.frame?.agents?.[0];
    if (!agent) return null;
    return {
      gridX: agent[1],
      gridY: agent[2],
      tileSize: debug.tileSize,
      offset: debug.offset,
    };
  });
  if (!target) {
    throw new Error("No agent available for visual focus smoke.");
  }
  const box = await page.locator("#canvas-host").boundingBox();
  if (!box) {
    throw new Error("Canvas host bounding box was unavailable.");
  }
  await page.mouse.click(
    box.x + target.offset.x + target.gridX * target.tileSize + target.tileSize / 2,
    box.y + target.offset.y + target.gridY * target.tileSize + target.tileSize / 2,
  );
  await page.waitForFunction(() => window.__viewerDebug?.selectedAgentId !== null);
  await page.locator("#map-fit-focus").click();
  await page.waitForFunction(() => (window.__viewerDebug?.mapView?.zoom ?? 1) > 1);
}

try {
  await mkdir(outputDir, { recursive: true });
  browser = await chromium.launch({ headless: true });

  for (const { label, viewport } of viewports) {
    const page = await browser.newPage({ viewport });
    await waitForReplay(page);
    await assertNoHorizontalClipping(page, `${label} overview`);
    await page.screenshot({ path: `${outputDir}/${label}-overview.png`, fullPage: true });

    await selectFirstAgent(page);
    await assertNoHorizontalClipping(page, `${label} agent-focus`);
    await page.screenshot({ path: `${outputDir}/${label}-agent-focus.png`, fullPage: true });
    await page.close();
  }

  console.log(`viewer_visual_smoke_ok output=${outputDir}`);
} finally {
  if (browser) {
    await browser.close();
  }
  await stopViewerServer(viewerServer);
}
