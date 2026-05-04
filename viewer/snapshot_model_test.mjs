import assert from "node:assert/strict";
import {
  buildSnapshotBudgetReport,
  changedTileCoordinates,
  snapshotCacheKey,
  snapshotScaleForMap,
  trimSnapshotCache,
} from "./snapshot_model.mjs";

const map = {
  width: 3,
  height: 2,
  terrain_codes: [
    [0, 1, 2],
    [0, 1, 2],
  ],
};
const baseline = {
  tick: 1,
  field_state: {
    vegetation: [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ],
  },
  carcass_energy_codes: [
    [0, 0, 0],
    [0, 0, 0],
  ],
};
const frame = {
  tick: 2,
  field_state: {
    vegetation: [
      [0.1, 0.9, 0.3],
      [0.4, 0.5, 0.6],
    ],
  },
  carcass_energy_codes: [
    [0, 0, 0],
    [0, 2, 0],
  ],
};

assert.equal(snapshotScaleForMap({ width: 48, height: 32 }, 180), 3);
assert.equal(snapshotScaleForMap({ width: 20, height: 20 }, 180), 9);
assert.equal(
  snapshotCacheKey({
    frameIndex: 2,
    baselineIndex: 1,
    overlayMode: "hazard",
    overlayOpacity: 0.86,
    blendTerrain: true,
    maxPixels: 180,
    delta: true,
  }),
  "snapshot-v1:f2:b1:o-hazard:op86:blend1:p180:d1",
);

const deltas = changedTileCoordinates({ frame, baseline, map, maxChanges: 8 });
assert.deepEqual(deltas.map((entry) => [entry.x, entry.y, entry.fields]), [
  [1, 0, ["vegetation"]],
  [1, 1, ["carcass_energy_codes"]],
]);

const cache = new Map([
  ["a", 1],
  ["b", 2],
  ["c", 3],
]);
trimSnapshotCache(cache, 2);
assert.deepEqual([...cache.keys()], ["b", "c"]);

assert.deepEqual(buildSnapshotBudgetReport({ durationMs: 12.4, budgetMs: 24, cacheHit: false }), {
  durationMs: 12.4,
  budgetMs: 24,
  cacheHit: false,
  status: "pass",
});
assert.equal(buildSnapshotBudgetReport({ durationMs: 30, budgetMs: 24, cacheHit: false }).status, "warn");

console.log("snapshot_model_test_ok");
