# Capability Layer Spec

This document defines how agents may eventually act outside the simulator.

The goal is not unrestricted desktop escape. The goal is mediated, inspectable external action.

## Design Rules

1. No direct host access from agent logic.
All external actions must go through a capability gateway.

2. Every capability is explicit.
If a capability is not registered, agents cannot invoke it.

3. Every invocation is logged.
All requests, approvals, denials, inputs, outputs, and side effects must be auditable.

4. Start with read-mostly capabilities.
The first external actions should be low-risk and easy to inspect.

5. Human approval remains available.
Some capabilities may require explicit approval even after the system matures.

## Capability Model

Each capability must define:

- `capability_id`
- `name`
- `description`
- `risk_level`
- `input_schema`
- `output_schema`
- `rate_limit`
- `approval_policy`
- `sandbox_policy`

## Capability Tiers

### Tier 0: Disabled

- no external actions allowed

### Tier 1: Read-Only

Examples:

- read a small allowed file
- inspect a sandbox directory
- read a note board exposed by the simulator owner

### Tier 2: Sandboxed Write

Examples:

- write a note into a dedicated sandbox directory
- append to a simulator-owned journal

### Tier 3: Tool Use

Examples:

- invoke a tightly scoped helper tool
- request a browser-like lookup through an approval layer

### Tier 4: High-Risk

Examples:

- unrestricted filesystem access
- arbitrary shell execution
- uncontrolled network access

Tier 4 is out of scope for this project and should not be implemented.

## First Allowed External Capability

The recommended first capability is:

- `sandbox_write_note`

Why:

- visible side effect
- easy to audit
- low risk
- enough to test whether agents learn to externalize information

## Invocation Flow

1. Agent emits an external action intent.
2. Simulator validates the intent against the capability registry.
3. Policy layer checks rate limits and approval rules.
4. Sandbox executor performs the action if allowed.
5. Result is returned to the agent as an observation event.
6. Full audit log entry is appended.

## Audit Log Requirements

Each external action log entry stores:

- timestamp
- run id
- tick
- agent id
- lineage id
- capability id
- request payload
- decision: `allowed | denied`
- reason
- execution result summary

## MVP Position

The initial simulator implementation should expose:

- no live external capabilities
- only the interfaces and event types needed to add them later

This keeps the early ecology work clean while preserving the long-term objective.
