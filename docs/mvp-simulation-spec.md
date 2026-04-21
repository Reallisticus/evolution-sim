# MVP Simulation Spec

This document freezes the first implementation target. It is intentionally narrow.

It is a baseline document, not the active project queue. After the MVP, execution is governed by the `Foundation -> Mind -> Culture -> Capability` model in [evolution-simulator-blueprint.md](/Users/njm/Projects/evolution-sim/docs/evolution-simulator-blueprint.md).

## Post-MVP Semantic Contract

When Foundation layers extend the MVP world, replay and viewer semantics should stay explicit:

- `primary hydrology reason` means the tile's main hard-water source
- `hydrology support` means additional water-related support conditions and is counted separately from the primary reason
- `global refuge score` is averaged across forest tiles
- `species refuge score` is averaged across tiles occupied by that species in the current frame
- `hazard type` means the dominant damage pressure on the tile at the current frame, while `hazard level` is its current normalized severity
- `trophic role` is a derived phenotype from inherited combat and diet-bias traits, not a hard-coded caste flag
- `carcass stats` refer to current carrion state in the world, while `combat` stats refer to attack and damage activity over the run or frame

## Scope

The MVP exists to prove four things:

- the simulator is deterministic under a fixed seed
- agents survive and die under environmental pressure
- agents reproduce with mutation
- runs can be inspected through structured event logs and replay data

This version does not include:

- neural policies
- sexual reproduction
- explicit species logic
- combat
- signaling
- external-world actions

## World

Use a fixed-size 2D grid with deterministic ticks.

### World Constants

- width: `48`
- height: `32`
- max ticks per run: `2000`
- initial agents: `20`
- max alive agents: `320`

### Tile Schema

Each tile stores:

- `terrain`: `plain | forest | wetland | rocky | water`
- `food`: float in `[0, 1]`
- `water`: float in `[0, 1]`
- `occupant_id`: optional agent id

### Terrain Rules

- `plain`: normal movement cost, medium food growth, no water
- `forest`: higher movement cost, higher food growth, no water
- `water`: impassable for agents, water resource available from adjacent tiles

## Agent State

Each agent stores:

- `agent_id`
- `parent_id`
- `lineage_id`
- `x`, `y`
- `energy`
- `hydration`
- `age`
- `alive`
- `genome`

## Genome Schema

The MVP genome is compact and numeric.

### Fields

- `max_energy`
- `max_hydration`
- `move_cost`
- `food_efficiency`
- `water_efficiency`
- `reproduction_threshold`
- `mutation_scale`

### Mutation Rules

- a child inherits a mutated copy of the parent genome
- each gene is perturbed by seeded Gaussian noise
- each gene is clamped to a safe min/max range

## Tick Order

Every tick executes in this order:

1. resource regrowth
2. agent action selection in ascending `agent_id` order
3. action resolution
4. metabolism drain
5. reproduction checks
6. death checks
7. summary event emission

## Observation Model

The first implementation uses a built-in heuristic policy, not a trainable policy.

The action policy may inspect only:

- current tile food
- adjacent water access
- nearest visible food in a local radius
- agent energy and hydration

No global map access is allowed.

## Action Schema

The MVP action set is:

- `stay`
- `move_north`
- `move_south`
- `move_east`
- `move_west`
- `eat`
- `drink`

Invalid actions become `stay`.

## Resource Rules

### Food

- food regrows on `plain` and `forest`
- forest regrows faster than plain
- `eat` consumes food from the current tile
- consumed food becomes agent energy via `food_efficiency`

### Water

- water is sourced from adjacent `water` tiles
- `drink` restores hydration if at least one adjacent water tile exists
- restored hydration is scaled by `water_efficiency`

## Metabolism

Every tick:

- energy decreases by a base metabolic drain
- hydration decreases by a base metabolic drain
- movement adds extra energy cost based on `move_cost`

## Reproduction

Reproduction is asexual in the MVP.

An agent reproduces when:

- it is alive
- it has reached minimum reproduction age
- it is past a fixed reproduction cooldown
- its energy is above `reproduction_threshold`
- its hydration is above a minimum fraction of its capacity
- there is an empty neighboring tile

On reproduction:

- the parent loses a fixed reproduction energy cost
- the child receives a mutated genome
- the child starts with partial energy and hydration
- the child inherits the parent lineage id

## Death

An agent dies when:

- energy is `<= 0`
- hydration is `<= 0`
- age exceeds max age

Dead agents are removed from occupancy immediately.

## Event Types

The MVP log format must support:

- `run_started`
- `tick_started`
- `agent_moved`
- `agent_ate`
- `agent_drank`
- `agent_reproduced`
- `agent_died`
- `tick_completed`
- `run_completed`

## Replay Format

Each run should produce a single JSON replay artifact containing:

- run metadata
- config
- final summary
- append-only event list

This is not the final storage format. Parquet can replace or complement it later.

## Success Criteria

The MVP is successful when:

- two runs with the same seed produce identical summaries
- runs show births and deaths under resource pressure
- long runs can produce more total historical agents than the alive-population cap
- genome values drift over generations
- the replay file is sufficient to inspect what happened without rerunning the simulation
