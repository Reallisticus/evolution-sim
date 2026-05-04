export const SNAPSHOT_MODEL_VERSION = 1;
export const DEFAULT_SNAPSHOT_BUDGET_MS = 24;
export const DEFAULT_SNAPSHOT_CACHE_LIMIT = 48;

const DELTA_FIELDS = [
  ["field_state.vegetation", (frame) => frame?.field_state?.vegetation],
  ["field_state.fertility_codes", (frame) => frame?.field_state?.fertility_codes],
  ["field_state.moisture_codes", (frame) => frame?.field_state?.moisture_codes],
  ["field_state.heat_codes", (frame) => frame?.field_state?.heat_codes],
  ["hydrology_primary_codes", (frame) => frame?.hydrology_primary_codes],
  ["hydrology_support_codes", (frame) => frame?.hydrology_support_codes],
  ["hazard_type_codes", (frame) => frame?.hazard_type_codes],
  ["hazard_level_codes", (frame) => frame?.hazard_level_codes],
  ["carcass_energy_codes", (frame) => frame?.carcass_energy_codes],
  ["carcass_freshness_codes", (frame) => frame?.carcass_freshness_codes],
  ["ecology_state_codes", (frame) => frame?.ecology_state_codes],
  ["habitat_state_codes", (frame) => frame?.habitat_state_codes],
];

export function snapshotScaleForMap(map, maxPixels = 300) {
  const width = Math.max(1, Number(map?.width ?? 1));
  const height = Math.max(1, Number(map?.height ?? 1));
  const maxDimension = Math.max(width, height);
  return Math.max(3, Math.floor(Number(maxPixels ?? 300) / maxDimension));
}

export function snapshotCacheKey({
  frameIndex,
  baselineIndex = null,
  overlayMode = "terrain",
  overlayOpacity = 1,
  blendTerrain = true,
  maxPixels = 300,
  delta = false,
}) {
  const frame = Math.trunc(Number(frameIndex) || 0);
  const baseline = baselineIndex == null ? "none" : Math.trunc(Number(baselineIndex) || 0);
  const opacity = Math.round(Math.max(0, Math.min(Number(overlayOpacity) || 0, 1)) * 100);
  const pixels = Math.trunc(Number(maxPixels) || 300);
  return `snapshot-v${SNAPSHOT_MODEL_VERSION}:f${frame}:b${baseline}:o-${overlayMode}:op${opacity}:blend${
    blendTerrain ? 1 : 0
  }:p${pixels}:d${delta ? 1 : 0}`;
}

export function changedTileCoordinates({ frame, baseline, map, maxChanges = 512 }) {
  if (!frame || !baseline || !map) return [];
  const width = Math.max(0, Number(map.width ?? 0));
  const height = Math.max(0, Number(map.height ?? 0));
  const changes = [];
  const limit = Math.max(1, Math.trunc(Number(maxChanges) || 512));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const fields = [];
      for (const [name, getter] of DELTA_FIELDS) {
        if (!sameTileValue(getter(frame), getter(baseline), x, y)) {
          fields.push(shortFieldName(name));
        }
      }
      if (fields.length > 0) {
        changes.push({ x, y, fields });
        if (changes.length >= limit) return changes;
      }
    }
  }
  return changes;
}

export function buildSnapshotBudgetReport({
  durationMs,
  budgetMs = DEFAULT_SNAPSHOT_BUDGET_MS,
  cacheHit = false,
}) {
  const duration = Number(durationMs);
  const budget = Number(budgetMs);
  return {
    durationMs: Number.isFinite(duration) ? Math.round(duration * 10) / 10 : 0,
    budgetMs: Number.isFinite(budget) ? budget : DEFAULT_SNAPSHOT_BUDGET_MS,
    cacheHit: Boolean(cacheHit),
    status: cacheHit || duration <= budget ? "pass" : "warn",
  };
}

export function trimSnapshotCache(cache, limit = DEFAULT_SNAPSHOT_CACHE_LIMIT) {
  const maxEntries = Math.max(1, Math.trunc(Number(limit) || DEFAULT_SNAPSHOT_CACHE_LIMIT));
  while (cache?.size > maxEntries) {
    const oldest = cache.keys().next().value;
    cache.delete(oldest);
  }
}

function sameTileValue(leftGrid, rightGrid, x, y) {
  const left = leftGrid?.[y]?.[x];
  const right = rightGrid?.[y]?.[x];
  if (typeof left === "number" || typeof right === "number") {
    return Math.abs(Number(left ?? 0) - Number(right ?? 0)) < 0.0001;
  }
  return String(left ?? "") === String(right ?? "");
}

function shortFieldName(name) {
  return name.startsWith("field_state.") ? name.slice("field_state.".length) : name;
}
