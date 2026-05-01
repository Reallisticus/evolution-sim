import { mkdirSync, writeFileSync } from "node:fs";
import { chromium } from "playwright";
import { ensureViewerServer, stopViewerServer } from "./smoke_server.mjs";

const outputDir = "output/playwright/malformed-replays";
mkdirSync(outputDir, { recursive: true });

const cases = [
  {
    name: "empty_frames",
    fileName: "empty-frames.json",
    payload: malformedValidBase((payload) => {
      payload.summary.ticks_executed = 0;
      payload.viewer.frames = [];
    }),
    expected: "viewer.frames must be a non-empty array",
  },
  {
    name: "missing_signal_emissions",
    fileName: "missing-signal-emissions.json",
    payload: malformedValidBase((payload) => {
      delete payload.viewer.frames[0].signal_emissions;
    }),
    expected: "signal_emissions",
  },
  {
    name: "ragged_terrain",
    fileName: "ragged-terrain.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.map.terrain_codes[1] = [0];
    }),
    expected: "must have 2 columns",
  },
  {
    name: "mismatched_action_signal_capacity",
    fileName: "mismatched-action-signal-capacity.json",
    payload: malformedValidBase((payload) => {
      payload.viewer.trajectory.action_contract.communication.token_count = 3;
    }),
    expected: "must match signal contract capacity",
  },
];

const viewerServer = await ensureViewerServer();
let browser;

try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 760 } });
  for (const testCase of cases) {
    const path = `${outputDir}/${testCase.fileName}`;
    writeFileSync(path, JSON.stringify(testCase.payload), "utf8");
    const replayPath = `../${path}`;
    const viewerUrl = `http://127.0.0.1:4173/viewer/index.html?replay=${encodeURIComponent(replayPath)}`;

    await page.goto(viewerUrl, { waitUntil: "networkidle" });
    await page.waitForFunction(
      (expected) => {
        const status = document.querySelector("#load-status")?.textContent ?? "";
        return status.includes("Failed to load replay") && status.includes(expected);
      },
      testCase.expected,
    );
    const debugState = await page.evaluate(() => window.__viewerDebug);
    if (debugState?.loaded !== false) {
      throw new Error(`${testCase.name}: malformed replay was marked loaded`);
    }
  }

  console.log(`viewer_malformed_smoke_ok cases=${cases.length}`);
} finally {
  if (browser) {
    await browser.close();
  }
  await stopViewerServer(viewerServer);
}

function malformedValidBase(mutator) {
  const payload = validPayload();
  mutator(payload);
  return payload;
}

function validPayload() {
  return {
    summary: {
      run_id: "malformed-smoke",
      ticks_executed: 1,
    },
    viewer: {
      map: {
        width: 2,
        height: 2,
        terrain_codes: matrix(0),
        base_tile_fields: {
          fertility: matrix(0.5),
          moisture: matrix(0.5),
          heat: matrix(0.5),
        },
      },
      frames: [frame()],
      agent_encoding: agentEncoding(),
      agent_catalog: {},
      species_catalog: {},
      trajectory: {
        action_contract: actionContract(),
        observation_contract: {
          action_names: actionNames(),
          action_contract: actionContract(),
          signal_contract: signalContract(),
        },
        records: [
          {
            agent_id: 1,
            observation_metadata: { agent_id: 1 },
            observation_input: {},
            observation_digest: "0".repeat(64),
            action_mask: {},
            resolution_action_mask: {},
            requested_action: "stay",
            policy_id: "smoke",
            policy_version: "v0",
            action_valid: true,
            resolution_action_valid: true,
            resolved_action: "stay",
            outcome: {},
            reward: {},
          },
        ],
      },
    },
  };
}

function frame() {
  return {
    tick: 0,
    agents: [[1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, "herbivore", "none", "none", 1]],
    species_counts: [[1, 1]],
    species_metrics: {},
    field_state: {},
    signal_flow: {
      reproductive_emissions: 0,
      communication_emissions: 0,
      energy_spent: 0,
    },
    fresh_kill_energy_codes: matrix(0),
    carcass_energy_codes: matrix(0),
    carcass_freshness_codes: matrix(0),
    habitat_state_codes: matrix(0),
    hydrology_primary_codes: matrix(0),
    hydrology_support_codes: matrix(0),
    refuge_codes: matrix(0),
    refuge_score_codes: matrix(0),
    hazard_type_codes: matrix(0),
    hazard_level_codes: matrix(0),
    trophic_role_codes: matrix(1),
    meat_mode_codes: matrix(0),
    ecology_state_codes: matrix(0),
    biotic_fields: {
      prey_biomass: matrix(0),
      carrion: matrix(0),
      predator_risk: matrix(0),
    },
    signal_fields: {
      reproductive_signal: matrix(0),
      communication_signal: matrix(0),
    },
    signal_emissions: {
      schema_version: "foundation_signal_contract_v2",
      policy_visible: false,
      events: [],
      active_counts: {
        reproductive_signal: 0,
        communication_signal: 0,
      },
    },
  };
}

function matrix(value) {
  return [
    [value, value],
    [value, value],
  ];
}

function signalEmissionFields() {
  return [
    "field_name",
    "profile_id",
    "source_kind",
    "source_agent_id",
    "token_id",
    "profile_index",
    "x",
    "y",
    "intensity",
    "radius",
    "duration_ticks",
    "remaining_ticks",
    "decay_rate",
    "energy_cost",
    "emitted_tick",
  ];
}

function signalContract() {
  return {
    schema_version: "foundation_signal_contract_v2",
    profile_metadata_policy_visible: false,
    emission_debug_metadata_fields: signalEmissionFields(),
    communication_token_count: 4,
    communication_profiles_per_token: 2,
    reserved_profiles: reservedProfiles(),
  };
}

function actionContract() {
  const communicationActions = communicationActionKeys();
  return {
    communication: {
      token_count: 4,
      profiles_per_token: 2,
      action_keys: communicationActions,
    },
    reserved_action_keys: ["mate", ...communicationActions],
  };
}

function actionNames() {
  const movementActions = ["move_north", "move_south", "move_east", "move_west"];
  return [
    "stay",
    "eat",
    "drink",
    ...movementActions,
    ...movementActions.map((action) => action.replace("move_", "attack_")),
    "mate",
    ...communicationActionKeys(),
  ];
}

function communicationActionKeys() {
  const keys = [];
  for (let tokenIndex = 0; tokenIndex < 4; tokenIndex += 1) {
    for (let profileIndex = 0; profileIndex < 2; profileIndex += 1) {
      keys.push(`signal_${tokenIndex}_profile_${profileIndex}`);
    }
  }
  return keys;
}

function reservedProfiles() {
  const profiles = [];
  for (let tokenIndex = 0; tokenIndex < 4; tokenIndex += 1) {
    for (let profileIndex = 0; profileIndex < 2; profileIndex += 1) {
      profiles.push({
        field_name: "communication_signal",
        token_id: tokenIndex,
        profile_index: profileIndex,
        policy_visible: false,
      });
    }
  }
  return profiles;
}

function agentEncoding() {
  return [
    "agent_id",
    "x",
    "y",
    "energy",
    "hydration",
    "health",
    "health_ratio",
    "injury_load",
    "age",
    "energy_modifier",
    "hydration_modifier",
    "trophic_role",
    "last_damage_source",
    "water_access_reason",
    "species_id",
  ];
}
