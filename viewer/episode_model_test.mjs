import assert from "node:assert/strict";
import {
  buildEpisodePlaybackScope,
  buildStoryboardExportModel,
  episodeAgentRoleEntries,
  episodeCausalitySentence,
  episodeLedger,
  eventBookmarkSummary,
  eventCategoryCounts,
  eventLensTicks,
  eventTickSummary,
  eventTouchesPlaybackScope,
  tileExplanationSentence,
  tilePressureTags,
} from "./episode_model.mjs";

const attackEvents = [
  {
    tick: 3,
    type: "agent_moved",
    agent_id: 2,
    data: { x: 7, y: 7 },
  },
  {
    tick: 3,
    type: "agent_healed",
    agent_id: 3,
    data: {},
  },
  {
    tick: 3,
    type: "agent_attacked",
    agent_id: 12,
    data: { target_id: 20, damage: 1.1256, success: true, kill: true },
  },
  {
    tick: 3,
    type: "agent_ate",
    agent_id: 12,
    data: {
      food_source: "fresh_kill",
      gained_energy: 0.0638,
      x: 24,
      y: 14,
      source_breakdown: [{ source_agent_id: 20, killer_id: 12, energy: 0.0397 }],
      deposit_breakdown: [{ source_agent_id: 20, killer_id: 12, consumed: 0.0397 }],
    },
  },
  {
    tick: 3,
    type: "agent_ate",
    agent_id: 4,
    data: { food_source: "plant", gained_energy: 0.2, x: 8, y: 8 },
  },
  {
    tick: 3,
    type: "agent_drank",
    agent_id: 5,
    data: { x: 9, y: 8 },
  },
];

const counts = eventCategoryCounts(attackEvents);
assert.equal(counts.combat, 1);
assert.equal(counts.carcass, 1);
assert.equal(counts.meatMeals, 1);
assert.equal(counts.moves, 1);

const roles = episodeAgentRoleEntries(attackEvents).map((entry) => entry.roleLabel);
assert.deepEqual(roles, ["killed / source", "attacker / killer / consumer"]);
assert(!roles.some((role) => ["actor", "mover", "healed", "drinker"].includes(role)));

const ledger = episodeLedger(attackEvents, {
  foodSourceLabel: (value) => String(value).replace("_", " "),
});
assert.deepEqual(
  ledger.map((item) => item.title),
  ["Attack", "Feeding", "Support Events"],
);
assert(!ledger.some((item) => item.title === "Movement"));
assert(ledger.at(-1).detail.includes("1 moved"));
assert(ledger.at(-1).detail.includes("1 drink"));
assert(ledger[1].detail.includes("fresh kill"));

const markerForEvent = (event) => {
  if (event.data?.x != null && event.data?.y != null) return { x: event.data.x, y: event.data.y };
  if (event.type === "agent_attacked") return { x: 24, y: 14, from: { x: 23, y: 14 } };
  if (event.type === "agent_ate") return { x: event.data.x, y: event.data.y };
  if (event.type === "agent_died") return { x: event.data.x, y: event.data.y };
  return { x: event.data?.x, y: event.data?.y };
};
const scope = buildEpisodePlaybackScope(attackEvents, markerForEvent);
assert.deepEqual(scope.agentIds, [12, 20]);
assert.deepEqual(scope.tileKeys, ["23,14", "24,14"]);
assert.equal(scope.primaryEventCount, 2);
assert.equal(eventTouchesPlaybackScope(attackEvents[3], scope, markerForEvent), true);
assert.equal(
  eventTouchesPlaybackScope(
    { tick: 4, type: "agent_attacked", agent_id: 99, data: { target_id: 100, x: 1, y: 1 } },
    scope,
    markerForEvent,
  ),
  false,
);

assert.deepEqual(eventLensTicks(7, "causal", { startTick: 3, endTick: 9 }), [3, 4, 5, 6, 7]);
assert.deepEqual(eventLensTicks(2, "current", null), [2]);

assert(episodeCausalitySentence({ ...counts, death: 1 }, { carcassTileDelta: 1 }).includes("Predation"));
assert.equal(
  eventTickSummary({ births: 1, deaths: 2, attacks: 1, carcasses: 0, total: 4 }),
  "1 birth, 2 deaths, 1 attack",
);
assert.equal(eventBookmarkSummary(5, [], attackEvents), "1 combat event, 1 carcass flow event, 4 support events");

const storyboard = buildStoryboardExportModel({
  runId: "seed-test",
  replaySource: "../output/test.json",
  eventBookmarks: [],
  episodes: [{ tick: 3, births: 0, deaths: 1, attacks: 1, carcasses: 0, total: 2 }],
  exportedAt: "2026-05-04T00:00:00.000Z",
  entryForTick: (entry) => ({ tick: entry.tick, note: entry.note }),
});
assert.equal(storyboard.schema_version, "viewer_storyboard_v1");
assert.equal(storyboard.source, "major_episodes");
assert.equal(storyboard.entry_count, 1);
assert.deepEqual(storyboard.entries, [{ tick: 3, note: "1 death, 1 attack" }]);

const bookmarkStoryboard = buildStoryboardExportModel({
  runId: "seed-test",
  eventBookmarks: [{ tick: 9, summary: "Manual note", createdAt: "2026-05-04T00:00:00.000Z" }],
  episodes: [],
  entryForTick: (entry) => ({ tick: entry.tick, bookmarked_at: entry.bookmarked_at }),
});
assert.equal(bookmarkStoryboard.source, "bookmarks");
assert.deepEqual(bookmarkStoryboard.entries, [
  { tick: 9, bookmarked_at: "2026-05-04T00:00:00.000Z" },
]);

const tileContext = {
  hazardType: "exposure",
  hazardLevel: 0.42,
  carcassEnergy: 0.31,
  vegetation: 0.74,
};
const tags = tilePressureTags(tileContext, [{ agentId: 12 }], counts);
assert.deepEqual(
  tags.map((tag) => tag.label),
  ["Hazard pressure", "Carrion flow", "Vegetation rich", "1 occupant", "Combat recent"],
);
const sentence = tileExplanationSentence(tileContext, [{ agentId: 12 }], counts, {
  hazardLabel: (value) => value,
  formatPercent: (value) => `${Math.round(value * 100)}%`,
});
assert(sentence.includes("1 living agent"));
assert(sentence.includes("exposure hazard is active"));
assert(sentence.includes("carcass energy is available at 31%"));
assert(sentence.includes("Recent local events: 1 attack, 1 carcass-flow event."));

console.log("episode_model_test_ok");
