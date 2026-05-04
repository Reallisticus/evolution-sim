import * as PIXI from "./vendor/pixi.min.mjs";
import { eventLensCategory } from "./episode_model.mjs";

export const EVENT_MARKER_STYLES = {
  agent_reproduced: { color: 0x4ade80, shape: "birth", priority: 0 },
  agent_died: { color: 0xf87171, shape: "death", priority: 0 },
  carcass_deposited: { color: 0xfbbf24, shape: "carcass", priority: 1 },
  agent_attacked: { color: 0xfb7185, shape: "attack", priority: 1 },
  agent_damaged: { color: 0xf97316, shape: "damage", priority: 2 },
  agent_healed: { color: 0x86efac, shape: "heal", priority: 3 },
  agent_ate: { color: 0xfacc15, shape: "food", priority: 3 },
  agent_drank: { color: 0x38bdf8, shape: "water", priority: 3 },
};

const EVENT_LENS_STYLES = {
  birth: { color: 0x4ade80, label: "births" },
  death: { color: 0xf87171, label: "deaths" },
  combat: { color: 0xfb7185, label: "combat" },
  carcass: { color: 0xfbbf24, label: "carcass flow" },
  delta: { color: 0x8cc6ff, label: "tick delta" },
};

const DELTA_TILE_FIELDS = [
  ["carcass_energy_codes", "carcass"],
  ["fresh_kill_energy_codes", "carcass"],
  ["hazard_type_codes", "hazard"],
  ["habitat_state_codes", "habitat"],
  ["ecology_state_codes", "ecology"],
];

const CONDITION_COLORS = {
  energy: 0xfacc15,
  hydration: 0x38bdf8,
  health: 0x4ade80,
};

const MEAT_MODE_STROKES = {
  none: 0xf8fafc,
  scavenger: 0xf59e0b,
  hunter: 0xfb7185,
  mixed: 0xc084fc,
};

export function renderTerrainLayer(options) {
  const {
    layer,
    map,
    frame,
    tileSize,
    offset,
    overlayMode,
    overlayOpacity,
    blendTerrain,
    terrainColor,
    terrainBaseColor,
    overlayColorForTile,
    terrainNameFromCode,
    waterCode,
    blendColor,
  } = options;
  clearPixiLayer(layer);
  if (!layer || !map) return { terrainBlends: 0 };

  const graphics = new PIXI.Graphics();
  graphics.rect(offset.x, offset.y, map.width * tileSize, map.height * tileSize).fill({ color: 0x08101c });

  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const code = map.terrain_codes[y][x];
      const terrainFill = overlayMode === "terrain" ? terrainColor(code) : terrainBaseColor(code);
      graphics.rect(offset.x + x * tileSize, offset.y + y * tileSize, tileSize, tileSize).fill({
        color: terrainFill,
      });

      if (overlayMode !== "terrain" && frame) {
        const isWaterTile = code === waterCode;
        const overlayColor = overlayColorForTile(frame, x, y, code);
        graphics.rect(offset.x + x * tileSize, offset.y + y * tileSize, tileSize, tileSize).fill({
          color: overlayColor,
          alpha:
            isWaterTile &&
            [
              "hydrology",
              "shoreline",
              "refuge",
              "hazard",
              "carcass",
              "reproductive_signal",
              "ecology",
            ].includes(overlayMode)
              ? Math.min(0.94, overlayOpacity + 0.12)
              : overlayOpacity,
        });
      }

      drawTerrainTexture(
        graphics,
        terrainNameFromCode(code),
        offset.x + x * tileSize,
        offset.y + y * tileSize,
        tileSize,
      );
    }
  }

  const terrainBlends = blendTerrain
    ? drawTerrainBlends(graphics, map, tileSize, offset, terrainBaseColor, blendColor)
    : 0;

  graphics.rect(offset.x + 0.5, offset.y + 0.5, map.width * tileSize - 1, map.height * tileSize - 1).stroke({
    color: 0xdbeafe,
    width: 1,
    alpha: 0.18,
  });

  layer.addChild(graphics);
  return { terrainBlends };
}

export function renderAgentLayer(options) {
  const {
    layer,
    decodedAgents,
    tileSize,
    offset,
    selectedAgentId,
    selectedSpeciesId,
    colorForSpecies,
  } = options;
  clearPixiLayer(layer);
  if (!layer) return { glyphs: 0 };
  if (decodedAgents.length === 0) return { glyphs: 0 };

  const graphics = new PIXI.Graphics();
  const focusActive = selectedAgentId != null || selectedSpeciesId != null;
  for (const agent of decodedAgents) {
    const center = tileCenter(agent.x, agent.y, tileSize, offset);
    const selectedAgent = selectedAgentId === agent.agentId;
    const selectedSpecies = selectedSpeciesId === agent.speciesId;
    const inFocus = !focusActive || selectedAgent || selectedSpecies;
    const alpha = inFocus ? 0.96 : 0.2;
    drawAgentGlyph(graphics, agent, center.x, center.y, tileSize, {
      alpha,
      selectedAgent,
      selectedSpecies,
      dimmed: !inFocus,
      colorForSpecies,
    });
  }

  layer.addChild(graphics);
  return { glyphs: decodedAgents.length };
}

export function renderTrailLayer(options) {
  const {
    layer,
    decodedAgents,
    recentFrames,
    selectedPositions,
    selectedAgentId,
    selectedSpeciesId,
    dominantSpeciesId,
    tileSize,
    offset,
    colorForSpecies,
    maxTrailAgents = 32,
  } = options;
  clearPixiLayer(layer);
  if (!layer) return { trailSegments: 0, clusterHalos: 0 };

  const clusterHalos = drawSpeciesClusterHalos({
    layer,
    decodedAgents,
    tileSize,
    offset,
    selectedSpeciesId,
    colorForSpecies,
  });
  const selectedPathSegments = drawSelectedAgentLifePath({
    layer,
    selectedPositions,
    selectedSpeciesId,
    tileSize,
    offset,
    colorForSpecies,
  });
  const targetAgents = trailTargetAgents({
    decodedAgents,
    selectedAgentId,
    selectedSpeciesId,
    dominantSpeciesId,
    maxTrailAgents,
  });
  if (targetAgents.length === 0) {
    return { trailSegments: selectedPathSegments, clusterHalos };
  }

  const targetIds = new Set(targetAgents.map((agent) => agent.agentId));
  const positionsByFrame = recentFrames.map((agents) => {
    const positions = new Map();
    for (const agent of agents) {
      if (targetIds.has(agent.agentId)) {
        positions.set(agent.agentId, agent);
      }
    }
    return positions;
  });

  const graphics = new PIXI.Graphics();
  let segmentCount = 0;
  for (const agent of targetAgents) {
    let previous = null;
    positionsByFrame.forEach((positions, frameOffset) => {
      const current = positions.get(agent.agentId);
      if (!current) {
        previous = null;
        return;
      }
      if (previous && adjacentTrailStep(previous, current)) {
        const from = tileCenter(previous.x, previous.y, tileSize, offset);
        const to = tileCenter(current.x, current.y, tileSize, offset);
        const ageRatio = (frameOffset + 1) / positionsByFrame.length;
        graphics.moveTo(from.x, from.y).lineTo(to.x, to.y).stroke({
          color: colorForSpecies(agent.speciesId),
          width: selectedAgentId === agent.agentId ? 2.6 : 1.45,
          alpha: selectedAgentId === agent.agentId ? 0.64 : 0.14 + ageRatio * 0.28,
        });
        segmentCount += 1;
      }
      previous = current;
    });
  }

  if (segmentCount > 0) {
    layer.addChild(graphics);
  }
  return {
    trailSegments: selectedPathSegments + segmentCount,
    clusterHalos,
  };
}

export function renderDecisionOverlayLayer(options) {
  const {
    layer,
    decodedAgents,
    tick,
    tileSize,
    offset,
    mode,
    selectedAgentId,
    selectedSpeciesId,
    recordForAgent,
    isPointOnMap,
    maxDecisionOverlays = 56,
  } = options;
  if (!layer || mode === "off") return { decisionOverlays: 0 };

  const graphics = new PIXI.Graphics();
  let overlayCount = 0;
  if (mode === "selected") {
    const record = selectedAgentId == null ? null : recordForAgent(selectedAgentId, tick);
    if (drawDecisionRecordOverlay(graphics, record, tileSize, offset, { selected: true, isPointOnMap })) {
      overlayCount += 1;
    }
  } else {
    const records = decisionOverlayRecords({
      decodedAgents,
      tick,
      selectedAgentId,
      selectedSpeciesId,
      recordForAgent,
      maxDecisionOverlays,
    });
    for (const item of records) {
      const selected =
        item.agentId === selectedAgentId ||
        (selectedAgentId == null && item.speciesId === selectedSpeciesId);
      if (
        drawDecisionRecordOverlay(graphics, item.record, tileSize, offset, {
          selected,
          compact: true,
          reward: mode === "reward",
          isPointOnMap,
        })
      ) {
        overlayCount += 1;
      }
    }
  }

  if (overlayCount > 0) {
    layer.addChild(graphics);
  }
  return { decisionOverlays: overlayCount };
}

export function renderEventMapLayer(options) {
  const {
    layer,
    frame,
    decodedAgents,
    previousFrame,
    previousAgents = [],
    tileSize,
    offset,
    map,
    mode,
    events,
    markerForEvent,
    colorForSpecies,
    maxDeltaMarkers = 160,
    eventLensFrameWindow = 24,
  } = options;
  clearPixiLayer(layer);
  const emptyStats = {
    eventMarkers: 0,
    eventLensMarkers: 0,
    eventFlowLinks: 0,
    deltaMarkers: 0,
    markerCount: 0,
  };
  if (!layer || !frame || !map) return emptyStats;

  if (mode === "delta") {
    const deltaStats = drawMapDelta({
      layer,
      frame,
      previousFrame,
      decodedAgents,
      previousAgents,
      tileSize,
      offset,
      colorForSpecies,
      maxDeltaMarkers,
    });
    return {
      ...emptyStats,
      eventFlowLinks: deltaStats.links,
      deltaMarkers: deltaStats.markers,
      markerCount: deltaStats.markers,
    };
  }

  if (events.length === 0) return emptyStats;

  const agentById = new Map(decodedAgents.map((agent) => [agent.agentId, agent]));
  const graphics = new PIXI.Graphics();
  let markerCount = 0;
  let linkCount = 0;
  for (const event of events) {
    const marker = markerForEvent(event, agentById, event.tick);
    if (!marker || !isPointOnMap(marker.x, marker.y, map)) continue;
    if (mode === "current") {
      drawEventMarker(graphics, event, marker, tileSize, offset);
    } else {
      const result = drawEventLensMarker(graphics, event, marker, frame.tick, tileSize, offset, eventLensFrameWindow);
      linkCount += result.links;
    }
    markerCount += 1;
  }
  if (markerCount > 0) {
    layer.addChild(graphics);
  }
  return {
    eventMarkers: markerCount,
    eventLensMarkers: markerCount,
    eventFlowLinks: linkCount,
    deltaMarkers: 0,
    markerCount,
  };
}

function drawEventLensMarker(graphics, event, marker, currentTick, tileSize, offset, frameWindow) {
  const category = eventLensCategory(event);
  const style = EVENT_LENS_STYLES[category] ?? EVENT_LENS_STYLES.delta;
  const age = Math.max(0, Number(currentTick) - Number(event.tick));
  const recency = clamp01(1 - age / Math.max(frameWindow, 1));
  const center = tileCenter(marker.x, marker.y, tileSize, offset);
  const radius = Math.max(tileSize * (0.28 + recency * 0.22), 4);
  const alpha = 0.22 + recency * 0.62;
  let links = 0;

  if (marker.from) {
    const from = tileCenter(marker.from.x, marker.from.y, tileSize, offset);
    graphics.moveTo(from.x, from.y).lineTo(center.x, center.y).stroke({
      color: style.color,
      width: Math.max(1, tileSize * 0.08),
      alpha: 0.24 + recency * 0.42,
    });
    drawArrowHead(graphics, from, center, style.color, 0.35 + recency * 0.38, tileSize * 0.82);
    links += 1;
  }

  if (category === "birth") {
    graphics.circle(center.x, center.y, radius).fill({ color: style.color, alpha: 0.16 });
    graphics.circle(center.x, center.y, radius + 2).stroke({ color: style.color, width: 1.8, alpha });
    graphics.moveTo(center.x - radius * 0.45, center.y).lineTo(center.x + radius * 0.45, center.y).stroke({
      color: style.color,
      width: 1.5,
      alpha,
    });
    graphics.moveTo(center.x, center.y - radius * 0.45).lineTo(center.x, center.y + radius * 0.45).stroke({
      color: style.color,
      width: 1.5,
      alpha,
    });
  } else if (category === "death") {
    graphics.circle(center.x, center.y, radius + 1).stroke({ color: style.color, width: 1.8, alpha });
    graphics.moveTo(center.x - radius * 0.5, center.y - radius * 0.5).lineTo(
      center.x + radius * 0.5,
      center.y + radius * 0.5,
    ).stroke({ color: style.color, width: 1.8, alpha });
    graphics.moveTo(center.x + radius * 0.5, center.y - radius * 0.5).lineTo(
      center.x - radius * 0.5,
      center.y + radius * 0.5,
    ).stroke({ color: style.color, width: 1.8, alpha });
  } else if (category === "combat") {
    graphics.circle(center.x, center.y, radius + 2).stroke({ color: style.color, width: 1.7, alpha });
    graphics.moveTo(center.x - radius * 0.62, center.y + radius * 0.4).lineTo(
      center.x + radius * 0.62,
      center.y - radius * 0.4,
    ).stroke({ color: style.color, width: 1.8, alpha });
  } else {
    graphics.rect(center.x - radius * 0.55, center.y - radius * 0.55, radius * 1.1, radius * 1.1).fill({
      color: style.color,
      alpha: 0.14 + recency * 0.16,
    });
    graphics.rect(center.x - radius * 0.55, center.y - radius * 0.55, radius * 1.1, radius * 1.1).stroke({
      color: style.color,
      width: 1.6,
      alpha,
    });
  }

  graphics.circle(center.x, center.y, Math.max(1.2, radius * 0.18)).fill({
    color: 0xf8fafc,
    alpha: 0.22 + recency * 0.36,
  });
  return { links };
}

function drawSpeciesClusterHalos(options) {
  const { layer, decodedAgents, tileSize, offset, selectedSpeciesId, colorForSpecies } = options;
  const groups = speciesClusterGroups(decodedAgents, selectedSpeciesId).slice(0, 4);
  if (groups.length === 0) return 0;

  const graphics = new PIXI.Graphics();
  let haloCount = 0;
  for (const group of groups) {
    const centerX = group.points.reduce((total, point) => total + point.x, 0) / group.points.length;
    const centerY = group.points.reduce((total, point) => total + point.y, 0) / group.points.length;
    const center = tileCenter(centerX, centerY, tileSize, offset);
    const bounds = group.points.reduce(
      (box, point) => ({
        minX: Math.min(box.minX, point.x),
        maxX: Math.max(box.maxX, point.x),
        minY: Math.min(box.minY, point.y),
        maxY: Math.max(box.maxY, point.y),
      }),
      { minX: centerX, maxX: centerX, minY: centerY, maxY: centerY },
    );
    const selected = selectedSpeciesId === group.speciesId;
    const color = colorForSpecies(group.speciesId);
    const radiusX = clamp(
      (bounds.maxX - bounds.minX + 1) * tileSize * 0.62 + tileSize * 1.9,
      tileSize * 2.4,
      tileSize * 8.4,
    );
    const radiusY = clamp(
      (bounds.maxY - bounds.minY + 1) * tileSize * 0.62 + tileSize * 1.9,
      tileSize * 2.4,
      tileSize * 8.4,
    );
    graphics.ellipse(center.x, center.y, radiusX, radiusY).fill({
      color,
      alpha: selected ? 0.18 : 0.085,
    });
    graphics.ellipse(center.x, center.y, radiusX, radiusY).stroke({
      color,
      width: selected ? 1.5 : 0.9,
      alpha: selected ? 0.36 : 0.18,
    });
    graphics.ellipse(center.x, center.y, radiusX * 0.62, radiusY * 0.62).stroke({
      color: 0xf8fafc,
      width: 0.7,
      alpha: selected ? 0.15 : 0.07,
    });
    haloCount += 1;
  }

  if (haloCount > 0) {
    layer.addChild(graphics);
  }
  return haloCount;
}

function speciesClusterGroups(decodedAgents, selectedSpeciesId) {
  const groups = new Map();
  for (const agent of decodedAgents) {
    if (!groups.has(agent.speciesId)) {
      groups.set(agent.speciesId, []);
    }
    groups.get(agent.speciesId).push(agent);
  }
  return [...groups.entries()]
    .filter(([, points]) => points.length > 0)
    .map(([speciesId, points]) => ({
      speciesId,
      points,
      selected: selectedSpeciesId === speciesId,
    }))
    .sort((left, right) => {
      if (left.selected !== right.selected) return left.selected ? -1 : 1;
      return right.points.length - left.points.length;
    });
}

function drawSelectedAgentLifePath(options) {
  const { layer, selectedPositions, selectedSpeciesId, tileSize, offset, colorForSpecies } = options;
  if (selectedPositions.length < 2) return 0;

  const graphics = new PIXI.Graphics();
  const color = colorForSpecies(selectedSpeciesId ?? selectedPositions.at(-1)?.speciesId ?? 0);
  let segmentCount = 0;
  const visiblePoints = selectedPositions.slice(Math.max(0, selectedPositions.length - 64));
  for (let index = 1; index < visiblePoints.length; index += 1) {
    const previous = visiblePoints[index - 1];
    const current = visiblePoints[index];
    if (!adjacentTrailStep(previous, current)) continue;
    const from = tileCenter(previous.x, previous.y, tileSize, offset);
    const to = tileCenter(current.x, current.y, tileSize, offset);
    const ratio = index / Math.max(visiblePoints.length - 1, 1);
    graphics.moveTo(from.x, from.y).lineTo(to.x, to.y).stroke({
      color,
      width: 3.2,
      alpha: 0.18 + ratio * 0.52,
    });
    if (index % 3 === 0 || index === visiblePoints.length - 1) {
      graphics.circle(to.x, to.y, Math.max(tileSize * 0.12, 1.8)).fill({
        color: 0xf8fafc,
        alpha: 0.32 + ratio * 0.36,
      });
    }
    segmentCount += 1;
  }
  if (segmentCount > 0) {
    layer.addChild(graphics);
  }
  return segmentCount;
}

function trailTargetAgents(options) {
  const {
    decodedAgents,
    selectedAgentId,
    selectedSpeciesId,
    dominantSpeciesId,
    maxTrailAgents,
  } = options;
  const sortedAgents = [...decodedAgents].sort((left, right) => left.agentId - right.agentId);
  if (selectedAgentId != null) {
    const selectedAgent = sortedAgents.filter((agent) => agent.agentId === selectedAgentId);
    if (selectedAgent.length > 0) {
      return selectedAgent;
    }
    if (selectedSpeciesId != null) {
      return sortedAgents
        .filter((agent) => agent.speciesId === selectedSpeciesId)
        .slice(0, maxTrailAgents);
    }
  }
  const speciesId = selectedSpeciesId ?? dominantSpeciesId;
  if (speciesId == null) return [];
  return sortedAgents.filter((agent) => agent.speciesId === speciesId).slice(0, maxTrailAgents);
}

function adjacentTrailStep(previous, current) {
  return Math.abs(previous.x - current.x) <= 2 && Math.abs(previous.y - current.y) <= 2;
}

function decisionOverlayRecords(options) {
  const {
    decodedAgents,
    tick,
    selectedAgentId,
    selectedSpeciesId,
    recordForAgent,
    maxDecisionOverlays,
  } = options;
  const focusActive = selectedAgentId != null || selectedSpeciesId != null;
  const records = [];
  const sortedAgents = [...decodedAgents].sort((left, right) => {
    const leftFocus = left.agentId === selectedAgentId || left.speciesId === selectedSpeciesId;
    const rightFocus = right.agentId === selectedAgentId || right.speciesId === selectedSpeciesId;
    if (leftFocus !== rightFocus) return leftFocus ? -1 : 1;
    return left.agentId - right.agentId;
  });

  for (const agent of sortedAgents) {
    if (focusActive && selectedAgentId == null && agent.speciesId !== selectedSpeciesId) {
      continue;
    }
    const record = recordForAgent(agent.agentId, tick);
    if (!record) continue;
    records.push({ agentId: agent.agentId, speciesId: agent.speciesId, record });
    if (records.length >= maxDecisionOverlays) break;
  }

  if (records.length > 0 || focusActive) {
    return records;
  }

  return sortedAgents
    .map((agent) => ({
      agentId: agent.agentId,
      speciesId: agent.speciesId,
      record: recordForAgent(agent.agentId, tick),
    }))
    .filter((item) => item.record)
    .slice(0, maxDecisionOverlays);
}

function drawDecisionRecordOverlay(graphics, record, tileSize, offset, options = {}) {
  if (!record?.before || !record?.after) return false;
  const before = record.before;
  const after = record.after;
  if (!options.isPointOnMap(before.x, before.y) || !options.isPointOnMap(after.x, after.y)) return false;

  const start = tileCenter(before.x, before.y, tileSize, offset);
  const end = tileCenter(after.x, after.y, tileSize, offset);
  const valid = record.action_valid && record.resolution_action_valid;
  const rewardTotal = Number(record.reward?.total ?? 0);
  const rewardMagnitude = clamp(Math.abs(rewardTotal) / 2.5, 0.18, 1);
  const color = options.reward
    ? rewardColor(rewardTotal)
    : valid
      ? options.selected
        ? 0xf8fafc
        : 0xbae6fd
      : 0xfb7185;
  const alpha = options.reward ? 0.36 + rewardMagnitude * 0.5 : valid ? 0.58 : 0.92;
  const lineWidth = options.compact ? Math.max(1, tileSize * 0.08) : Math.max(1.4, tileSize * 0.13);
  const dotRadius = options.compact ? Math.max(tileSize * 0.14, 2.4) : Math.max(tileSize * 0.2, 3);
  const endRadius = options.reward
    ? Math.max(tileSize * (0.18 + rewardMagnitude * 0.18), 3)
    : Math.max(tileSize * 0.28, 4);

  graphics.circle(start.x, start.y, dotRadius).fill({ color, alpha: options.compact ? 0.25 : 0.42 });
  graphics.circle(end.x, end.y, endRadius).stroke({
    color,
    width: options.selected ? 2 : 1.35,
    alpha,
  });
  if (options.reward) {
    graphics.circle(end.x, end.y, endRadius * 0.58).fill({
      color,
      alpha: 0.16 + rewardMagnitude * 0.18,
    });
  }
  if (before.x !== after.x || before.y !== after.y) {
    graphics.moveTo(start.x, start.y).lineTo(end.x, end.y).stroke({
      color,
      width: lineWidth,
      alpha,
    });
    if (!options.compact || options.selected) {
      drawArrowHead(graphics, start, end, color, alpha, tileSize);
    }
  }
  if (!valid) {
    const radius = Math.max(tileSize * 0.3, 4);
    graphics.moveTo(end.x - radius, end.y - radius).lineTo(end.x + radius, end.y + radius).stroke({
      color: 0xfb7185,
      width: 1.6,
      alpha: 0.92,
    });
    graphics.moveTo(end.x + radius, end.y - radius).lineTo(end.x - radius, end.y + radius).stroke({
      color: 0xfb7185,
      width: 1.6,
      alpha: 0.92,
    });
  }
  return true;
}

function rewardColor(total) {
  if (total > 0.05) return 0x4ade80;
  if (total < -0.05) return 0xfb7185;
  return 0xf8fafc;
}

function drawTerrainBlends(graphics, map, tileSize, offset, terrainBaseColor, blendColor) {
  let blendCount = 0;
  const width = Math.max(1, tileSize * 0.1);
  for (let y = 0; y < map.height; y += 1) {
    for (let x = 0; x < map.width; x += 1) {
      const code = map.terrain_codes[y][x];
      const baseX = offset.x + x * tileSize;
      const baseY = offset.y + y * tileSize;
      const neighbors = [
        { dx: 1, dy: 0, edge: "right" },
        { dx: 0, dy: 1, edge: "bottom" },
      ];
      for (const neighbor of neighbors) {
        const nextX = x + neighbor.dx;
        const nextY = y + neighbor.dy;
        if (nextX >= map.width || nextY >= map.height) continue;
        const nextCode = map.terrain_codes[nextY][nextX];
        if (nextCode === code) continue;
        const color = blendColor(terrainBaseColor(code), terrainBaseColor(nextCode), 0.5);
        if (neighbor.edge === "right") {
          graphics.rect(baseX + tileSize - width * 0.5, baseY, width, tileSize).fill({ color, alpha: 0.28 });
        } else {
          graphics.rect(baseX, baseY + tileSize - width * 0.5, tileSize, width).fill({ color, alpha: 0.24 });
        }
        blendCount += 1;
      }
    }
  }
  return blendCount;
}

function drawTerrainTexture(graphics, terrainName, x, y, tileSize) {
  if (tileSize < 9) return;
  const inset = Math.max(1, tileSize * 0.16);
  if (terrainName === "forest") {
    graphics.moveTo(x + inset, y + tileSize - inset).lineTo(x + tileSize - inset, y + inset).stroke({
      color: 0xd9f99d,
      width: 0.45,
      alpha: 0.16,
    });
    graphics.circle(x + tileSize * 0.36, y + tileSize * 0.36, Math.max(0.7, tileSize * 0.045)).fill({
      color: 0xbbf7d0,
      alpha: 0.16,
    });
  } else if (terrainName === "wetland") {
    graphics.circle(x + tileSize * 0.68, y + tileSize * 0.34, Math.max(0.8, tileSize * 0.055)).fill({
      color: 0xbae6fd,
      alpha: 0.22,
    });
    graphics.circle(x + tileSize * 0.34, y + tileSize * 0.68, Math.max(0.8, tileSize * 0.05)).fill({
      color: 0xbbf7d0,
      alpha: 0.18,
    });
  } else if (terrainName === "rocky") {
    graphics.moveTo(x + inset, y + inset).lineTo(x + tileSize - inset, y + tileSize - inset).stroke({
      color: 0xf5f5f4,
      width: 0.5,
      alpha: 0.12,
    });
    graphics.moveTo(x + inset, y + tileSize * 0.62).lineTo(x + tileSize * 0.56, y + inset).stroke({
      color: 0xf5f5f4,
      width: 0.42,
      alpha: 0.1,
    });
  } else if (terrainName === "water") {
    graphics.moveTo(x + inset, y + tileSize * 0.42).lineTo(x + tileSize - inset, y + tileSize * 0.42).stroke({
      color: 0xbfdbfe,
      width: 0.42,
      alpha: 0.18,
    });
    graphics.moveTo(x + inset, y + tileSize * 0.68).lineTo(x + tileSize - inset, y + tileSize * 0.68).stroke({
      color: 0xbfdbfe,
      width: 0.36,
      alpha: 0.12,
    });
  } else if (terrainName === "plain") {
    graphics.circle(x + tileSize * 0.72, y + tileSize * 0.72, Math.max(0.6, tileSize * 0.035)).fill({
      color: 0xfef3c7,
      alpha: 0.1,
    });
  }
}

function drawAgentGlyph(graphics, agent, centerX, centerY, tileSize, options) {
  const radius = Math.max(tileSize * 0.39, 5);
  const fillColor = options.colorForSpecies(agent.speciesId);
  const strokeColor = MEAT_MODE_STROKES[agent.meatMode] ?? MEAT_MODE_STROKES.none;
  const strokeWidth = agent.meatMode === "none" ? 1 : 1.8;

  graphics.circle(centerX, centerY, radius + 3.8).fill({
    color: 0x07111f,
    alpha: options.dimmed ? 0.08 : 0.34,
  });

  if (options.selectedSpecies || options.selectedAgent) {
    graphics.circle(centerX, centerY, radius + 7).stroke({
      color: 0xf8fafc,
      width: options.selectedAgent ? 2 : 1,
      alpha: options.selectedAgent ? 0.62 : 0.32,
    });
  }

  if (agent.reproductionReady) {
    graphics.circle(centerX, centerY, radius + 5.2).stroke({
      color: 0x2dd4bf,
      width: 1.4,
      alpha: options.dimmed ? 0.18 : 0.68,
    });
  }

  drawRatioRing(graphics, centerX, centerY, radius + 4.2, agent.healthRatio, CONDITION_COLORS.health, options.alpha);
  drawRatioRing(
    graphics,
    centerX,
    centerY,
    radius + 6.2,
    agent.hydrationRatio,
    CONDITION_COLORS.hydration,
    options.alpha,
  );
  drawRatioRing(graphics, centerX, centerY, radius + 8.2, agent.energyRatio, CONDITION_COLORS.energy, options.alpha);

  drawRoleShape(
    graphics,
    agent.trophicRole,
    centerX,
    centerY,
    radius,
    { color: fillColor, alpha: options.alpha },
    { color: strokeColor, width: strokeWidth, alpha: options.dimmed ? 0.28 : 0.88 },
  );

  if ((agent.matchedDietRatio ?? 0) > 0.75 && !options.dimmed) {
    graphics.circle(centerX, centerY, Math.max(radius * 0.25, 1.5)).fill({
      color: 0xf8fafc,
      alpha: 0.72,
    });
  }
}

function drawRoleShape(graphics, role, centerX, centerY, radius, fillStyle, strokeStyle) {
  const drawPath = () => {
    if (role === "carnivore") {
      return graphics.poly([
        centerX,
        centerY - radius,
        centerX + radius * 0.94,
        centerY + radius * 0.78,
        centerX - radius * 0.94,
        centerY + radius * 0.78,
      ]);
    }
    if (role === "omnivore") {
      return graphics.poly([
        centerX,
        centerY - radius,
        centerX + radius,
        centerY,
        centerX,
        centerY + radius,
        centerX - radius,
        centerY,
      ]);
    }
    return graphics.circle(centerX, centerY, radius);
  };

  drawPath().fill(fillStyle);
  drawPath().stroke(strokeStyle);
}

function drawRatioRing(graphics, centerX, centerY, radius, ratio, color, alpha) {
  const clamped = clamp01(Number(ratio ?? 0));
  graphics.circle(centerX, centerY, radius).stroke({
    color: 0xf8fafc,
    width: 0.8,
    alpha: 0.07 * alpha,
  });
  if (clamped <= 0) return;
  const start = -Math.PI / 2;
  const end = start + Math.PI * 2 * clamped;
  graphics.moveTo(centerX, centerY - radius).arc(centerX, centerY, radius, start, end).stroke({
    color,
    width: 1.25,
    alpha: Math.max(0.18, alpha * (0.32 + clamped * 0.68)),
  });
}

function drawMapDelta(options) {
  const {
    layer,
    frame,
    previousFrame,
    decodedAgents,
    previousAgents,
    tileSize,
    offset,
    colorForSpecies,
    maxDeltaMarkers,
  } = options;
  if (!previousFrame) return { markers: 0, links: 0 };
  const previousById = new Map(previousAgents.map((agent) => [agent.agentId, agent]));
  const currentById = new Map(decodedAgents.map((agent) => [agent.agentId, agent]));
  const graphics = new PIXI.Graphics();
  let markers = 0;
  let links = 0;

  for (const agent of decodedAgents) {
    const previous = previousById.get(agent.agentId);
    if (!previous) {
      drawDeltaBirth(graphics, agent, tileSize, offset);
      markers += 1;
    } else if (previous.x !== agent.x || previous.y !== agent.y) {
      drawDeltaMove(graphics, previous, agent, tileSize, offset, colorForSpecies);
      markers += 1;
      links += 1;
    }
    if (markers >= maxDeltaMarkers) break;
  }

  if (markers < maxDeltaMarkers) {
    for (const agent of previousAgents) {
      if (currentById.has(agent.agentId)) continue;
      drawDeltaDeath(graphics, agent, tileSize, offset);
      markers += 1;
      if (markers >= maxDeltaMarkers) break;
    }
  }

  if (markers < maxDeltaMarkers) {
    const changedTiles = changedTileDeltas(frame, previousFrame, maxDeltaMarkers);
    for (const tile of changedTiles.slice(0, maxDeltaMarkers - markers)) {
      drawDeltaTile(graphics, tile, tileSize, offset);
      markers += 1;
    }
  }

  if (markers > 0) {
    layer.addChild(graphics);
  }
  return { markers, links };
}

function drawDeltaBirth(graphics, agent, tileSize, offset) {
  const center = tileCenter(agent.x, agent.y, tileSize, offset);
  const radius = Math.max(tileSize * 0.38, 4);
  graphics.circle(center.x, center.y, radius + 2).stroke({ color: 0x4ade80, width: 2, alpha: 0.84 });
  graphics.moveTo(center.x - radius * 0.5, center.y).lineTo(center.x + radius * 0.5, center.y).stroke({
    color: 0x4ade80,
    width: 1.8,
    alpha: 0.9,
  });
  graphics.moveTo(center.x, center.y - radius * 0.5).lineTo(center.x, center.y + radius * 0.5).stroke({
    color: 0x4ade80,
    width: 1.8,
    alpha: 0.9,
  });
}

function drawDeltaDeath(graphics, agent, tileSize, offset) {
  const center = tileCenter(agent.x, agent.y, tileSize, offset);
  const radius = Math.max(tileSize * 0.4, 4);
  graphics.circle(center.x, center.y, radius + 2).stroke({ color: 0xf87171, width: 2, alpha: 0.82 });
  graphics.moveTo(center.x - radius * 0.55, center.y - radius * 0.55).lineTo(
    center.x + radius * 0.55,
    center.y + radius * 0.55,
  ).stroke({ color: 0xf87171, width: 1.9, alpha: 0.92 });
  graphics.moveTo(center.x + radius * 0.55, center.y - radius * 0.55).lineTo(
    center.x - radius * 0.55,
    center.y + radius * 0.55,
  ).stroke({ color: 0xf87171, width: 1.9, alpha: 0.92 });
}

function drawDeltaMove(graphics, previous, current, tileSize, offset, colorForSpecies) {
  const from = tileCenter(previous.x, previous.y, tileSize, offset);
  const to = tileCenter(current.x, current.y, tileSize, offset);
  const color = colorForSpecies(current.speciesId);
  graphics.moveTo(from.x, from.y).lineTo(to.x, to.y).stroke({
    color,
    width: Math.max(1.2, tileSize * 0.1),
    alpha: 0.66,
  });
  drawArrowHead(graphics, from, to, color, 0.72, tileSize * 0.82);
  graphics.circle(to.x, to.y, Math.max(tileSize * 0.18, 2.4)).fill({ color, alpha: 0.42 });
}

function drawDeltaTile(graphics, tile, tileSize, offset) {
  const color = tile.deltaKind === "carcass" ? 0xfbbf24 : tile.deltaKind === "hazard" ? 0xfb7185 : 0x8cc6ff;
  const x = offset.x + tile.x * tileSize;
  const y = offset.y + tile.y * tileSize;
  graphics.rect(x + 1, y + 1, Math.max(1, tileSize - 2), Math.max(1, tileSize - 2)).fill({
    color,
    alpha: 0.16,
  });
  graphics.rect(x + 1, y + 1, Math.max(1, tileSize - 2), Math.max(1, tileSize - 2)).stroke({
    color,
    width: 1,
    alpha: 0.46,
  });
}

function drawEventMarker(graphics, event, marker, tileSize, offset) {
  const style = EVENT_MARKER_STYLES[event.type];
  const center = tileCenter(marker.x, marker.y, tileSize, offset);
  const radius = Math.max(tileSize * 0.45, 5);

  if (marker.from) {
    const from = tileCenter(marker.from.x, marker.from.y, tileSize, offset);
    graphics.moveTo(from.x, from.y).lineTo(center.x, center.y).stroke({
      color: style.color,
      width: 1.6,
      alpha: 0.58,
    });
  }

  if (style.shape === "birth") {
    graphics.circle(center.x, center.y, radius + 2).stroke({ color: style.color, width: 2.2, alpha: 0.82 });
    graphics.moveTo(center.x - radius * 0.45, center.y).lineTo(center.x + radius * 0.45, center.y).stroke({
      color: style.color,
      width: 1.8,
      alpha: 0.9,
    });
    graphics.moveTo(center.x, center.y - radius * 0.45).lineTo(center.x, center.y + radius * 0.45).stroke({
      color: style.color,
      width: 1.8,
      alpha: 0.9,
    });
    return;
  }

  if (style.shape === "death") {
    graphics.circle(center.x, center.y, radius + 2).stroke({ color: style.color, width: 2.2, alpha: 0.78 });
    graphics.moveTo(center.x - radius * 0.55, center.y - radius * 0.55).lineTo(
      center.x + radius * 0.55,
      center.y + radius * 0.55,
    ).stroke({ color: style.color, width: 2, alpha: 0.92 });
    graphics.moveTo(center.x + radius * 0.55, center.y - radius * 0.55).lineTo(
      center.x - radius * 0.55,
      center.y + radius * 0.55,
    ).stroke({ color: style.color, width: 2, alpha: 0.92 });
    return;
  }

  if (style.shape === "carcass") {
    graphics.rect(center.x - radius * 0.52, center.y - radius * 0.52, radius * 1.04, radius * 1.04).fill({
      color: style.color,
      alpha: 0.2,
    });
    graphics.rect(center.x - radius * 0.52, center.y - radius * 0.52, radius * 1.04, radius * 1.04).stroke({
      color: style.color,
      width: 1.8,
      alpha: 0.82,
    });
    return;
  }

  if (style.shape === "attack" || style.shape === "damage") {
    graphics.circle(center.x, center.y, radius).stroke({ color: style.color, width: 1.8, alpha: 0.68 });
    graphics.moveTo(center.x - radius * 0.6, center.y + radius * 0.42).lineTo(
      center.x + radius * 0.6,
      center.y - radius * 0.42,
    ).stroke({ color: style.color, width: 1.9, alpha: 0.84 });
    return;
  }

  if (style.shape === "heal") {
    graphics.circle(center.x, center.y, radius * 0.68).fill({ color: style.color, alpha: 0.22 });
    graphics.moveTo(center.x - radius * 0.42, center.y).lineTo(center.x + radius * 0.42, center.y).stroke({
      color: style.color,
      width: 1.6,
      alpha: 0.72,
    });
    graphics.moveTo(center.x, center.y - radius * 0.42).lineTo(center.x, center.y + radius * 0.42).stroke({
      color: style.color,
      width: 1.6,
      alpha: 0.72,
    });
    return;
  }

  graphics.circle(center.x, center.y, radius * 0.58).fill({
    color: style.color,
    alpha: style.shape === "water" ? 0.3 : 0.24,
  });
  graphics.circle(center.x, center.y, radius * 0.58).stroke({
    color: style.color,
    width: 1.2,
    alpha: 0.72,
  });
}

function changedTileDeltas(frame, previousFrame, maxDeltaMarkers) {
  if (!frame || !previousFrame) return [];
  const changes = [];
  const seen = new Set();
  for (const [field, deltaKind] of DELTA_TILE_FIELDS) {
    const currentGrid = frame[field];
    const previousGrid = previousFrame[field];
    if (!Array.isArray(currentGrid) || !Array.isArray(previousGrid)) continue;
    for (let y = 0; y < currentGrid.length; y += 1) {
      const currentRow = currentGrid[y] ?? [];
      const previousRow = previousGrid[y] ?? [];
      for (let x = 0; x < currentRow.length; x += 1) {
        if (currentRow[x] === previousRow[x]) continue;
        const key = `${x},${y}`;
        if (seen.has(key)) continue;
        seen.add(key);
        changes.push({ x, y, deltaKind });
        if (changes.length >= maxDeltaMarkers) return changes;
      }
    }
  }
  return changes;
}

export function countChangedTileDeltas(frame, previousFrame) {
  if (!frame || !previousFrame) return 0;
  const seen = new Set();
  for (const [field] of DELTA_TILE_FIELDS) {
    const currentGrid = frame[field];
    const previousGrid = previousFrame[field];
    if (!Array.isArray(currentGrid) || !Array.isArray(previousGrid)) continue;
    for (let y = 0; y < currentGrid.length; y += 1) {
      const currentRow = currentGrid[y] ?? [];
      const previousRow = previousGrid[y] ?? [];
      for (let x = 0; x < currentRow.length; x += 1) {
        if (currentRow[x] === previousRow[x]) continue;
        seen.add(`${x},${y}`);
      }
    }
  }
  return seen.size;
}

function drawArrowHead(graphics, start, end, color, alpha, tileSize) {
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const length = Math.max(tileSize * 0.22, 4);
  const width = length * 0.62;
  const tip = {
    x: end.x - Math.cos(angle) * Math.max(tileSize * 0.12, 2),
    y: end.y - Math.sin(angle) * Math.max(tileSize * 0.12, 2),
  };
  const left = {
    x: tip.x - Math.cos(angle) * length + Math.cos(angle + Math.PI / 2) * width,
    y: tip.y - Math.sin(angle) * length + Math.sin(angle + Math.PI / 2) * width,
  };
  const right = {
    x: tip.x - Math.cos(angle) * length + Math.cos(angle - Math.PI / 2) * width,
    y: tip.y - Math.sin(angle) * length + Math.sin(angle - Math.PI / 2) * width,
  };
  graphics.poly([tip.x, tip.y, left.x, left.y, right.x, right.y]).fill({ color, alpha });
}

function tileCenter(x, y, tileSize, offset) {
  return {
    x: offset.x + x * tileSize + tileSize / 2,
    y: offset.y + y * tileSize + tileSize / 2,
  };
}

function isPointOnMap(x, y, map) {
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

function clearPixiLayer(layer) {
  if (!layer) return;
  for (const child of [...layer.children]) {
    child.destroy?.();
  }
  layer.removeChildren();
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}
