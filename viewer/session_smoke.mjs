import { mkdir, readFile, writeFile } from "node:fs/promises";
import { chromium } from "playwright";
import { ensureViewerServer, stopViewerServer } from "./smoke_server.mjs";

const baseReplayPath = process.env.REPLAY_PATH ?? "output/sim-runs/species-check.json";
const outputDir = "output/playwright/session-replays";

const viewerServer = await ensureViewerServer();
let browser;

try {
  await mkdir(outputDir, { recursive: true });
  const basePayload = JSON.parse(await readFile(baseReplayPath, "utf8"));
  const firstReplay = replayVariant(basePayload, "session-a");
  const secondReplay = replayVariant(basePayload, "session-b");
  await writeFile(`${outputDir}/session-a.json`, JSON.stringify(firstReplay));
  await writeFile(`${outputDir}/session-b.json`, JSON.stringify(secondReplay));

  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(
    "http://127.0.0.1:4173/viewer/index.html?replay=../output/playwright/session-replays/session-a.json",
    { waitUntil: "networkidle" },
  );
  await page.waitForFunction(() => window.__viewerDebug?.loaded === true);
  await page.waitForFunction(() => window.__viewerDebug?.replaySource?.includes("session-a.json"));

  await page.locator('[data-detail-view="species"]').click();
  await page.waitForFunction(() => {
    const panel = document.querySelector('[data-detail-panel="species"]');
    return panel && !panel.classList.contains("hidden");
  });
  const speciesItems = page.locator("#lineage-tree [data-lineage-species-id]");
  if ((await speciesItems.count()) === 0) {
    throw new Error("Expected lineage species items before session reload.");
  }
  await speciesItems.first().click();
  await page.waitForFunction(() => window.__viewerDebug?.selectedSpeciesId !== null);

  await page.locator("#timeline").evaluate((node) => {
    node.value = "5";
    node.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForFunction(() => window.__viewerDebug?.currentFrameIndex === 5);

  await page.locator("#replay-url").fill("../output/playwright/session-replays/session-b.json");
  await page.locator("#load-url").click();
  await page.waitForFunction(() => window.__viewerDebug?.loaded === true);
  await page.waitForFunction(() => window.__viewerDebug?.replaySource?.includes("session-b.json"));
  await page.waitForFunction(() => {
    return (
      window.__viewerDebug?.loadGeneration >= 2 &&
      window.__viewerDebug?.currentFrameIndex === 0 &&
      window.__viewerDebug?.selectedAgentId === null &&
      window.__viewerDebug?.selectedSpeciesId === null &&
      window.__viewerDebug?.replayModel?.trajectoryAgentIndexSize > 0
    );
  });

  console.log(`viewer_session_smoke_ok generation=${(await page.evaluate(() => window.__viewerDebug.loadGeneration))}`);
} finally {
  if (browser) {
    await browser.close();
  }
  await stopViewerServer(viewerServer);
}

function replayVariant(payload, runId) {
  return {
    ...payload,
    run_id: runId,
    summary: {
      ...payload.summary,
      run_id: runId,
    },
    events: payload.events.map((event) =>
      event.type === "run_started"
        ? { ...event, data: { ...(event.data ?? {}), run_id: runId } }
        : event,
    ),
  };
}
