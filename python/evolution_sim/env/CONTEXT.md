# Foundation Runtime

Foundation Runtime is the ecological source of truth for the simulator. It
defines the world pressures, deterministic tick semantics, policy-facing
surfaces, and runtime state that later exports and controllers observe.

## Language

**Foundation**:
The world plus observability layer that must be strong enough to measure later
controller work. Foundation includes ecology, runtime contracts, summary
metrics, replay semantics, and gate evidence.
_Avoid_: world-only work, visuals-only work

**World substrate**:
The terrain, climate, resources, hazards, hydrology, agents, signals, and
deterministic tick progression that define a run.
_Avoid_: map, board

**Observation contract**:
The policy-facing encoded view of self state, local ecology, local patch fields,
and allowed diagnostics available at action-selection time.
_Avoid_: raw world read, hidden state dump

**Action affordance**:
A Foundation-provided mask or availability signal that says whether an action is
currently resolvable or useful enough to consider. It is not always pure physics
legality.
_Avoid_: legality oracle

**Policy surface**:
The explicit boundary through which controllers receive observations, action
masks, and post-action feedback. A controller should not read private world
state outside this surface.
_Avoid_: world helper access, private world reads

**Summary-only mode**:
A lightweight run mode that computes shared summary metrics and optional
trajectory records without building replay, viewer, or taxonomy payloads.
_Avoid_: compact replay, headless replay

**Full replay**:
The compatibility artifact for viewer playback and replay contract checks. It
contains events, viewer data, frames, maps, catalogs, taxonomy, and analytics.
_Avoid_: training dataset, summary report

**Hydrology reason**:
The primary hard-water access reason used for drinking semantics on a tile.
_Avoid_: water score

**Hydrology support**:
Additional water-related support conditions on a tile, such as shoreline,
wetland substrate, or flooded support.
_Avoid_: water reason

**Refuge score**:
The local protection signal used to explain shelter, recovery, and species
occupancy patterns.
_Avoid_: safety score

**Hazard type**:
The dominant current land-tile damage pressure. Hazard type must stay distinct
from generic danger or death cause.
_Avoid_: danger type

**Trophic role**:
A derived phenotype from inherited combat and diet-bias traits. It is not a
hard-coded caste.
_Avoid_: class, species type

**Meat mode**:
The derived animal-resource strategy exposed in ecology, replay, and viewer
surfaces.
_Avoid_: carnivore flag

**Carrion**:
Animal-resource state produced by death and decay. It is a world resource and a
diagnostic pressure surface, not just a visual marker.
_Avoid_: corpse marker

**Reproductive signal**:
Runtime signal state used around mate search, reproductive readiness, and
communication substrate work.
_Avoid_: message, chat

**Foundation gate**:
The readiness check for viability, ecology pressure, replay compatibility,
summary contracts, and release boundary behavior.
_Avoid_: unit test suite

## Example Dialogue

Developer: "Can Mind read the nearest plant and nearest carrion from the world?"

Domain expert: "Only through the observation contract or an explicit policy
surface. If it needs more context, add an observable field and gate it; do not
let the controller read private world state."

Developer: "Can I compute a new chart from summary-only by reusing replay
frames?"

Domain expert: "No. Summary-only must stay lightweight. Add a shared summary
metric directly or use full replay when the viewer payload is required."
