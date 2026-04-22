# Evolution Sim

Minimal standalone workspace for the artificial-life simulator.

## Layout

- `docs/`: project blueprint and frozen MVP specs
- `python/evolution_sim/`: headless simulator package
- `python/tests/`: deterministic and replay tests

## Run

```bash
cd /Users/njm/Projects/evolution-sim
PYTHONPATH=python python3 -m evolution_sim.cli.run_headless --seed 7 --ticks 2000 --output output/sim-runs/seed7.json
```

## Inspect

```bash
cd /Users/njm/Projects/evolution-sim
PYTHONPATH=python python3 -m evolution_sim.cli.inspect_run output/sim-runs/seed7.json
```

## Test

```bash
cd /Users/njm/Projects/evolution-sim
PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_*.py'
```

## Viewer

Generate a small replay for the browser viewer:

```bash
cd /Users/njm/Projects/evolution-sim
npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json
```

Serve the repo root:

```bash
cd /Users/njm/Projects/evolution-sim
npm run viewer:serve
```

Open:

```text
http://127.0.0.1:4173/viewer/index.html?replay=../output/sim-runs/species-check.json
```

The viewer shows:

- terrain mix across plain, forest, wetland, rocky, and water, plus live agents colored by replay-adjudicated durable species
- current frame season, births, deaths, and live species count
- moving climate fronts and deterministic storm/drought state
- hydrology split into primary hard-water reasons and separate support counts, so shoreline, wetland support, and flooded support cannot be confused with the tile's main drinkable source
- climate-driven hazard layers for exposure and instability, with hazard counts, hazard overlays, and per-tile hazard levels
- carcass fields with visible carrion stock, freshness, mixed-source patching, deposition, consumption, and decay pressure
- trophic-role visibility so herbivore, omnivore, and carnivore occupancy can be inspected directly
- habitat-state dynamics such as bloom, flooded, and parched regions
- vegetation depletion, canopy shelter, and terrain-recovery pressure layered on top of habitat and climate, tracked only for land tiles
- active species leaderboard with current size, peak size, and lineage spread
- per-agent inspector with durable species, species status, transient ecotype, terrain context, water-access reason, adjacent-to-water support, wetland/flooded flags, refuge score, health, injury, trophic role, hazard state, carcass presence, hydration and energy modifiers, habitat state, ecology state, and genome traits relevant to niche pressure
- environmental overlays for fertility, moisture, heat, hydrology, shoreline, refuge, hazard, carcass, trophic role, habitat, and ecology that match the simulation state
- habitat overlay and habitat-pressure time-series chart
- separate hard-water and support/refuge charts so primary adjacent water, primary wetland, primary flooded, shoreline support, and canopy refuge stay semantically distinct
- separate hazard, carcass, and combat charts so damage pressure, carrion stock and flow, and predation activity can be attributed over time
- ecology overlay and vegetation-recovery time-series chart, with water kept visually separate from land ecology
- time-series charts for alive population, species count, births/deaths, and trait drift
- species ecology panels for terrain occupancy, shoreline support exposure, hard water access, refuge exposure, hazard exposure, trophic composition, stress, reproduction pressure, attack outcomes, and carcass use, with refuge averages labeled by denominator
- collapse and extinction event visibility tied to the replay timeline

Smoke-check the viewer:

```bash
cd /Users/njm/Projects/evolution-sim
npm run viewer:smoke
```

## Verification Flow

Run the regression and viewer checks for the current Foundation slice, including biotic pressure:

```bash
cd /Users/njm/Projects/evolution-sim
npm run sim:test
npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json
npm run sim:inspect output/sim-runs/species-check.json
REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke
```

Manual viewer URL:

```text
http://127.0.0.1:4173/viewer/index.html?replay=../output/sim-runs/species-check.json
```
