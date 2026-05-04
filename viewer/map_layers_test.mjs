import assert from "node:assert/strict";
import {
  EVENT_MARKER_STYLES,
  countChangedTileDeltas,
  renderAgentLayer,
  renderDecisionOverlayLayer,
  renderTerrainLayer,
  renderTrailLayer,
} from "./map_layers.mjs";

class TestLayer {
  constructor() {
    this.children = [];
  }

  addChild(child) {
    this.children.push(child);
  }

  removeChildren() {
    const children = this.children;
    this.children = [];
    return children;
  }
}

const previousFrame = {
  carcass_energy_codes: [
    [0, 0, 0],
    [0, 0, 0],
  ],
  fresh_kill_energy_codes: [
    [0, 0, 0],
    [0, 0, 0],
  ],
  hazard_type_codes: [
    ["none", "none", "none"],
    ["none", "none", "none"],
  ],
  habitat_state_codes: [
    ["stable", "stable", "stable"],
    ["stable", "stable", "stable"],
  ],
  ecology_state_codes: [
    ["quiet", "quiet", "quiet"],
    ["quiet", "quiet", "quiet"],
  ],
};

const currentFrame = {
  carcass_energy_codes: [
    [0, 1, 0],
    [0, 0, 0],
  ],
  fresh_kill_energy_codes: [
    [0, 1, 0],
    [0, 0, 0],
  ],
  hazard_type_codes: [
    ["none", "none", "none"],
    ["none", "exposure", "none"],
  ],
  habitat_state_codes: [
    ["stable", "stable", "stable"],
    ["stable", "stable", "bloom"],
  ],
  ecology_state_codes: [
    ["quiet", "quiet", "quiet"],
    ["quiet", "quiet", "bloom"],
  ],
};

assert.equal(countChangedTileDeltas(currentFrame, previousFrame), 3);
assert.equal(countChangedTileDeltas(currentFrame, null), 0);
assert.equal(EVENT_MARKER_STYLES.agent_reproduced.priority, 0);
assert(EVENT_MARKER_STYLES.agent_attacked.priority <= EVENT_MARKER_STYLES.agent_ate.priority);

const terrainLayer = new TestLayer();
const terrainStats = renderTerrainLayer({
  layer: terrainLayer,
  map: {
    width: 2,
    height: 2,
    terrain_codes: [
      [0, 1],
      [0, 4],
    ],
  },
  frame: currentFrame,
  tileSize: 12,
  offset: { x: 0, y: 0 },
  overlayMode: "terrain",
  overlayOpacity: 0.7,
  blendTerrain: true,
  terrainColor: (code) => [0x253028, 0x183626, 0x314b34, 0x143262][code] ?? 0xff00ff,
  terrainBaseColor: (code) => [0x253028, 0x183626, 0x314b34, 0x143262][code] ?? 0xff00ff,
  overlayColorForTile: () => 0xffffff,
  terrainNameFromCode: (code) => ({ 0: "plain", 1: "forest", 4: "water" })[code] ?? "plain",
  waterCode: 4,
  blendColor: (start, end) => Math.round((start + end) / 2),
});
assert.equal(terrainStats.terrainBlends, 3);
assert.equal(terrainLayer.children.length, 1);

const agentLayer = new TestLayer();
const agentStats = renderAgentLayer({
  layer: agentLayer,
  decodedAgents: [
    {
      agentId: 1,
      speciesId: 3,
      x: 0,
      y: 0,
      meatMode: "hunter",
      trophicRole: "carnivore",
      reproductionReady: true,
      healthRatio: 0.8,
      hydrationRatio: 0.7,
      energyRatio: 0.6,
      matchedDietRatio: 0.9,
    },
    {
      agentId: 2,
      speciesId: 4,
      x: 1,
      y: 1,
      meatMode: "none",
      trophicRole: "herbivore",
      reproductionReady: false,
      healthRatio: 0.5,
      hydrationRatio: 0.5,
      energyRatio: 0.5,
      matchedDietRatio: 0.3,
    },
  ],
  tileSize: 12,
  offset: { x: 0, y: 0 },
  selectedAgentId: 1,
  selectedSpeciesId: null,
  colorForSpecies: (speciesId) => 0x111111 + speciesId,
});
assert.equal(agentStats.glyphs, 2);
assert.equal(agentLayer.children.length, 1);

const trailLayer = new TestLayer();
const trailStats = renderTrailLayer({
  layer: trailLayer,
  decodedAgents: [
    { agentId: 1, speciesId: 3, x: 2, y: 2 },
    { agentId: 2, speciesId: 3, x: 3, y: 2 },
    { agentId: 3, speciesId: 4, x: 0, y: 1 },
  ],
  recentFrames: [
    [
      { agentId: 1, speciesId: 3, x: 0, y: 2 },
      { agentId: 2, speciesId: 3, x: 1, y: 2 },
    ],
    [
      { agentId: 1, speciesId: 3, x: 1, y: 2 },
      { agentId: 2, speciesId: 3, x: 2, y: 2 },
    ],
    [
      { agentId: 1, speciesId: 3, x: 2, y: 2 },
      { agentId: 2, speciesId: 3, x: 3, y: 2 },
    ],
  ],
  selectedPositions: [
    { agentId: 1, speciesId: 3, x: 0, y: 2 },
    { agentId: 1, speciesId: 3, x: 1, y: 2 },
    { agentId: 1, speciesId: 3, x: 2, y: 2 },
  ],
  selectedAgentId: 1,
  selectedSpeciesId: 3,
  dominantSpeciesId: 3,
  tileSize: 12,
  offset: { x: 0, y: 0 },
  colorForSpecies: (speciesId) => 0x222222 + speciesId,
  maxTrailAgents: 32,
});
assert.equal(trailStats.clusterHalos, 2);
assert.equal(trailStats.trailSegments, 4);
assert.equal(trailLayer.children.length, 3);

const decisionLayer = new TestLayer();
const decisionStats = renderDecisionOverlayLayer({
  layer: decisionLayer,
  decodedAgents: [
    { agentId: 1, speciesId: 3, x: 2, y: 2 },
    { agentId: 2, speciesId: 4, x: 1, y: 1 },
  ],
  tick: 4,
  tileSize: 12,
  offset: { x: 0, y: 0 },
  mode: "reward",
  selectedAgentId: null,
  selectedSpeciesId: null,
  recordForAgent: (agentId) => ({
    action_valid: true,
    resolution_action_valid: true,
    before: { x: agentId - 1, y: agentId - 1 },
    after: { x: agentId, y: agentId },
    reward: { total: agentId === 1 ? 0.2 : -0.2 },
  }),
  isPointOnMap: (x, y) => x >= 0 && y >= 0 && x < 5 && y < 5,
  maxDecisionOverlays: 56,
});
assert.equal(decisionStats.decisionOverlays, 2);
assert.equal(decisionLayer.children.length, 1);

console.log("map_layers_test_ok");
