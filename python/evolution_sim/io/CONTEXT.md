# Replay And Trajectory Exports

Replay And Trajectory Exports are the durable outputs used to inspect,
validate, train from, and compare simulator runs. They define compatibility and
training data boundaries, not the live world itself.

## Language

**Export contract**:
A stable shape, ordering, and semantic promise for a persisted simulator output.
Changing one requires matching tests, docs, and compatibility review.
_Avoid_: serialization detail

**Full replay contract**:
The contract for full JSON replay output consumed by viewer and golden checks.
Key order, event order, frame order, and viewer payload shape are compatibility
surfaces.
_Avoid_: arbitrary JSON

**Shared summary**:
The summary fields available across run modes. These fields must remain
computable without replay frames or viewer payloads.
_Avoid_: replay summary

**Replay taxonomy**:
The species, ecotype, lineage, and catalog semantics attached to full replay
outputs.
_Avoid_: labels-only metadata

**Trajectory record**:
One policy-decision row containing observation, action mask, requested/resolved
action, outcome, reward components, and related provenance needed for training
or diagnostics.
_Avoid_: replay event

**Trajectory stream**:
A gzip JSONL artifact containing a header, one or more trajectory records, and a
footer with run and trajectory summary metadata.
_Avoid_: replay log

**Artifact**:
A serialized controller, model, report, or training output that can be reused in
evaluation or diagnostics.
_Avoid_: checkpoint, blob

**Golden**:
A fixed replay expectation used to detect unintended compatibility drift.
_Avoid_: snapshot unless referring to test mechanics

**Gate report**:
A JSON report whose blockers, warnings, timing, and readiness fields determine
whether a boundary passes, needs review, or fails.
_Avoid_: benchmark

**Benchmark report**:
A repeatable performance report with scenario metrics, completion status,
runtime counters, and machine profile.
_Avoid_: gate report

## Example Dialogue

Developer: "Can I add a field to the full replay map?"

Domain expert: "Yes, but treat it as a full replay contract change. Update the
viewer, generated contracts, golden expectations, and replay invariant docs."

Developer: "Can the Mind trainer learn from full replay events?"

Domain expert: "Use trajectory records instead. Full replay is for viewer and
compatibility; trajectory streams are the training-data boundary."
