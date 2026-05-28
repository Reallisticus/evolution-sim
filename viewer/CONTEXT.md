# Viewer Contracts

Viewer Contracts cover browser and Node-side inspection of full replay output.
The viewer observes replay data; it does not own simulation truth.

## Language

**Viewer**:
The browser-facing inspection surface for full replay outputs, map layers,
frames, species/catalog views, and dashboard models.
_Avoid_: simulator frontend

**Replay payload**:
The full replay data consumed by the viewer, including frames, map, agent
encoding, catalogs, analytics, and events.
_Avoid_: summary-only output

**Frame**:
One captured replay state used for playback and visual inspection.
_Avoid_: tick unless referring to simulator time directly

**Map layer**:
A viewer-rendered surface for terrain, hydrology, refuge, hazards, trophic
roles, meat modes, ecology, or other replay map state.
_Avoid_: overlay when the contract is data-backed

**Agent encoding**:
The ordered per-agent field contract used to decode frame agent data.
_Avoid_: row shape

**Catalog**:
Replay metadata for agents, species, ecotypes, or legends that lets the viewer
interpret encoded IDs and frame data.
_Avoid_: lookup table

**Generated contract**:
A checked viewer contract artifact generated from Python-side expectations and
validated by viewer tests.
_Avoid_: hand-maintained schema

**Viewer smoke**:
An executable check that opens or validates replay behavior across core viewer
paths.
_Avoid_: screenshot test unless visual inspection is specifically included

## Example Dialogue

Developer: "Can the viewer infer missing species labels from the latest frame?"

Domain expert: "No. Full replay owns catalog semantics. If a label or ID is
needed, add it to the replay contract and generated viewer contracts."

Developer: "Can summary-only feed a compact viewer?"

Domain expert: "No. Summary-only is not a replay payload. Use full replay for
viewer playback."
