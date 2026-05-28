# Project Docs

Project Docs are the durable written boundary for roadmap intent, readiness
audits, invariants, experiment ledgers, and architecture decisions.

## Language

**Blueprint**:
The long-horizon project direction and layer model: Foundation, Mind, Culture,
and Capability.
_Avoid_: backlog

**Readiness audit**:
A dated assessment of whether a layer or boundary is mature enough for the next
phase of work.
_Avoid_: status note

**Invariant doc**:
A document that records behavior or output semantics that should not drift
without explicit contract updates.
_Avoid_: design idea

**Experiment ledger**:
A durable account of candidate runs, results, failures, and next research
boundaries.
_Avoid_: scratch notes

**Stage plan**:
A staged implementation and validation path with explicit acceptance
boundaries.
_Avoid_: roadmap unless it lacks gate criteria

**ADR**:
A short architecture decision record for a hard-to-reverse, non-obvious
trade-off. ADRs explain why a decision was made.
_Avoid_: changelog entry

**Spec**:
A normative description of expected behavior or interfaces that implementation
and tests should preserve.
_Avoid_: aspiration

## Example Dialogue

Developer: "This failed candidate taught us the scalar tuning family is
exhausted. Where does that go?"

Domain expert: "Put it in the Mind v3 experiment ledger, not an ADR. It is a
research result and next-boundary note."

Developer: "We are choosing to make full replay the compatibility contract
instead of summary-only. Is that an ADR?"

Domain expert: "Yes, if the trade-off was real and future readers may try to
reverse it. Otherwise keep the invariant doc current."
