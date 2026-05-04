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

async function assertMapChromeDiscipline(page, label) {
  const issues = await page.evaluate(() => {
    const visible = (node) => {
      if (!node) return false;
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
    const canvasHost = document.querySelector("#canvas-host");
    const mapLegend = document.querySelector(".map-legend");
    const toolsDrawer = document.querySelector(".map-tools-drawer");
    const viewportHeight = window.innerHeight;
    const compactLegendHeight = window.innerWidth <= 480 ? 280 : 190;

    if (!toolsDrawer) {
      problems.push("missing map tools drawer");
    } else if (toolsDrawer.open) {
      problems.push("map tools drawer starts open");
    }

    for (const selector of [
      "#map-navigation",
      "label[for='overlay-opacity']",
      "label[for='blend-terrain-toggle']",
    ]) {
      const node = document.querySelector(selector);
      if (visible(node)) {
        problems.push(`${selector} is visible in the primary map chrome`);
      }
    }

    if (visible(mapLegend)) {
      const legendRect = mapLegend.getBoundingClientRect();
      if (legendRect.height > compactLegendHeight) {
        problems.push(`map legend chrome too tall: ${Math.round(legendRect.height)}px`);
      }
    }

    if (!visible(canvasHost)) {
      problems.push("canvas host is not visible");
    } else {
      const canvasRect = canvasHost.getBoundingClientRect();
      const visibleHeight = Math.min(canvasRect.bottom, viewportHeight) - Math.max(canvasRect.top, 0);
      const minHeight =
        window.innerWidth <= 480
          ? Math.min(280, Math.max(220, viewportHeight * 0.26))
          : Math.min(420, Math.max(240, viewportHeight * 0.38));
      if (canvasRect.height < minHeight) {
        problems.push(`canvas host too short: ${Math.round(canvasRect.height)}px`);
      }
      const minVisibleHeight =
        window.innerWidth <= 480 ? Math.min(240, viewportHeight * 0.26) : Math.min(260, viewportHeight * 0.3);
      if (visibleHeight < minVisibleHeight) {
        problems.push(`canvas host too low in viewport: ${Math.round(visibleHeight)}px visible`);
      }
      if (window.innerWidth <= 480 && canvasRect.height > canvasRect.width * 1.15) {
        problems.push(
          `mobile canvas host is too portrait for the map: ${Math.round(canvasRect.width)}x${Math.round(
            canvasRect.height,
          )}`,
        );
      }
    }

    return problems;
  });
  if (issues.length > 0) {
    throw new Error(`${label} map chrome: ${issues.join("; ")}`);
  }
}

async function assertCanvasPanelWrapsChildren(page, label) {
  const issues = await page.evaluate(() => {
    const panel = document.querySelector("#canvas-panel");
    if (!panel) return ["missing canvas panel"];
    const panelStyle = window.getComputedStyle(panel);
    const panelRect = panel.getBoundingClientRect();
    const visibleChildren = [...panel.children].filter((node) => {
      const style = window.getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        rect.width > 0 &&
        rect.height > 0
      );
    });
    if (visibleChildren.length === 0) return ["canvas panel has no visible children"];
    const paddingBottom = Number.parseFloat(panelStyle.paddingBottom) || 0;
    const deepestChildBottom = Math.max(...visibleChildren.map((node) => node.getBoundingClientRect().bottom));
    const allowedBottom = panelRect.bottom - Math.min(8, paddingBottom);
    if (deepestChildBottom > allowedBottom + 1) {
      return [
        `canvas panel bottom ${Math.round(panelRect.bottom)} does not wrap child bottom ${Math.round(
          deepestChildBottom,
        )}`,
      ];
    }
    return [];
  });
  if (issues.length > 0) {
    throw new Error(`${label} panel containment: ${issues.join("; ")}`);
  }
}

async function assertVisualBudgets(page, label) {
  const issues = await page.evaluate(() => {
    const visibleRect = (selector) => {
      const node = document.querySelector(selector);
      if (!node) return null;
      const style = window.getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        rect.width <= 0 ||
        rect.height <= 0
      ) {
        return null;
      }
      return rect;
    };
    const problems = [];
    const toolsOpen = document.querySelector(".map-tools-drawer")?.open === true;
    const limits = window.innerWidth <= 480
      ? { timelineHeight: 178, controlHeight: toolsOpen ? 320 : 238, explainerHeight: 118, inspectorHeight: 72 }
      : { timelineHeight: 132, controlHeight: toolsOpen ? 300 : 178, explainerHeight: 88, inspectorHeight: 48 };
    const checks = [
      ["#story-timeline", "timeline", limits.timelineHeight],
      ["#map-control-strip", "map controls", limits.controlHeight],
      ["#tile-inspector", "tile inspector", limits.inspectorHeight],
      ["#tile-explainer", "tile explainer", limits.explainerHeight],
    ];
    for (const [selector, name, maxHeight] of checks) {
      const rect = visibleRect(selector);
      if (rect && rect.height > maxHeight) {
        problems.push(`${name} over budget: ${Math.round(rect.height)}px > ${maxHeight}px`);
      }
    }

    const panel = document.querySelector("#canvas-panel");
    if (panel) {
      const panelStyle = window.getComputedStyle(panel);
      const gap = Number.parseFloat(panelStyle.gap) || 0;
      if (gap > 12) {
        problems.push(`canvas panel gap over budget: ${Math.round(gap)}px`);
      }
    }
    return problems;
  });
  if (issues.length > 0) {
    throw new Error(`${label} visual budgets: ${issues.join("; ")}`);
  }
}

async function assertEventTimelineSemantics(page, label) {
  const issues = await page.evaluate(() => {
    const strip = document.querySelector("#event-strip");
    const significant = window.__viewerDebug?.significantEventTicks?.length ?? 0;
    if (!strip || significant === 0) return [];
    const markerCount = strip.querySelectorAll("[data-event-tick]").length;
    const bandCount = strip.querySelectorAll("[data-event-band]").length;
    const stripWidth = strip.getBoundingClientRect().width || window.innerWidth;
    const compactMax = Math.max(24, Math.floor(stripWidth / 10));
    const problems = [];
    if (significant > compactMax && markerCount > compactMax + 4) {
      problems.push(
        `event strip renders ${markerCount} markers for ${significant} significant ticks; max ${compactMax}`,
      );
    }
    if (significant > compactMax && bandCount === 0) {
      problems.push("dense event strip does not render semantic episode bands");
    }
    for (const band of strip.querySelectorAll("[data-event-band]")) {
      const label = band.querySelector(".event-band-label")?.textContent?.trim() ?? "";
      if (!label || /^\d+$/.test(label)) {
        problems.push("event band label is not semantic");
        break;
      }
    }
    return problems;
  });
  if (issues.length > 0) {
    throw new Error(`${label} event timeline semantics: ${issues.join("; ")}`);
  }
}

async function assertAnnotationLayout(page, label) {
  const issues = await page.evaluate(() => {
    const host = document.querySelector("#canvas-host");
    const annotations = [...document.querySelectorAll("#map-annotations .map-annotation")];
    if (!host || annotations.length === 0) return [];
    const hostRect = host.getBoundingClientRect();
    const rects = annotations
      .map((node, index) => ({ index, rect: node.getBoundingClientRect() }))
      .filter(({ rect }) => rect.width > 0 && rect.height > 0);
    const problems = [];

    for (const node of annotations) {
      const mode = node.getAttribute("data-callout-mode");
      if (!["label", "pin"].includes(mode ?? "")) {
        problems.push("annotation is missing an explicit placement mode");
        break;
      }
    }

    for (const { index, rect } of rects) {
      if (
        rect.left < hostRect.left - 1 ||
        rect.top < hostRect.top - 1 ||
        rect.right > hostRect.right + 1 ||
        rect.bottom > hostRect.bottom + 1
      ) {
        problems.push(`annotation ${index} escapes canvas host`);
      }
    }

    for (let i = 0; i < rects.length; i += 1) {
      for (let j = i + 1; j < rects.length; j += 1) {
        const a = rects[i].rect;
        const b = rects[j].rect;
        const overlapX = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        const overlapY = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        const overlapArea = overlapX * overlapY;
        const smallerArea = Math.min(a.width * a.height, b.width * b.height);
        if (smallerArea > 0 && overlapArea / smallerArea > 0.48) {
          problems.push(`annotations ${rects[i].index}/${rects[j].index} overlap heavily`);
        }
      }
    }

    return problems;
  });
  if (issues.length > 0) {
    throw new Error(`${label} annotation layout: ${issues.join("; ")}`);
  }
}

async function openMapTools(page) {
  const drawer = page.locator(".map-tools-drawer");
  if ((await drawer.count()) === 0) return;
  const open = await drawer.evaluate((node) => node.open === true);
  if (!open) {
    await page.locator(".map-tools-drawer summary").click();
    await page.waitForFunction(() => document.querySelector(".map-tools-drawer")?.open === true);
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
    await openMapTools(page);
    await page.locator("#map-fit-focus").click();
    await page.waitForFunction(() => (window.__viewerDebug?.mapView?.zoom ?? 1) > 1);
    return;
  }

  const speciesButton = page.locator("#species-list [data-species-id]").first();
  if ((await speciesButton.count()) > 0) {
    await speciesButton.click();
    await page.waitForFunction(() => window.__viewerDebug?.selectedAgentId !== null);
    await openMapTools(page);
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
  await openMapTools(page);
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
    await assertMapChromeDiscipline(page, `${label} overview`);
    await assertCanvasPanelWrapsChildren(page, `${label} overview`);
    await assertVisualBudgets(page, `${label} overview`);
    await assertEventTimelineSemantics(page, `${label} overview`);
    await page.screenshot({ path: `${outputDir}/${label}-overview.png`, fullPage: true });

    await selectFirstAgent(page);
    await assertNoHorizontalClipping(page, `${label} agent-focus`);
    await assertCanvasPanelWrapsChildren(page, `${label} agent-focus`);
    await assertVisualBudgets(page, `${label} agent-focus`);
    await assertAnnotationLayout(page, `${label} agent-focus`);
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
