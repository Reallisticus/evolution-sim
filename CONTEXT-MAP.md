# Context Map

This repo uses multiple domain contexts. Read the context that matches the work
area, then read any related contexts named below.

## Contexts

- [Foundation Runtime](./python/evolution_sim/env/CONTEXT.md) - world ecology,
  deterministic ticks, policy-facing observations, action affordances, and
  summary/replay-producing runtime state.
- [Replay And Trajectory Exports](./python/evolution_sim/io/CONTEXT.md) - full
  replay output, summary-only output, trajectory JSONL, and durable artifact
  boundaries.
- [Mind Controllers](./python/evolution_sim/mind/CONTEXT.md) - guarded Mind v1
  and v2 learned-controller work plus current Mind v3 autonomous-controller
  experimentation.
- [Viewer Contracts](./viewer/CONTEXT.md) - browser replay inspection, generated
  contracts, map layers, dashboard models, smoke tests, and visual validation.
- [Trainer Operations](./scripts/CONTEXT.md) - remote RTX trainer workflows,
  long-running experiment execution, artifact transfer, and local-vs-remote
  responsibilities.
- [Project Docs](./docs/CONTEXT.md) - roadmap, readiness audits, specs,
  invariant docs, experiment ledgers, and ADR placement.

## Relationships

- **Foundation Runtime -> Replay And Trajectory Exports**: runtime state becomes
  full replay, summary-only reports, and trajectory records; export contracts
  must not pull in extra runtime work casually.
- **Replay And Trajectory Exports -> Viewer Contracts**: full replay is the
  viewer compatibility source; summary-only is not a viewer payload.
- **Foundation Runtime -> Mind Controllers**: Foundation defines observations,
  action affordances, and post-action feedback; Mind consumes only the allowed
  policy surface.
- **Mind Controllers -> Replay And Trajectory Exports**: controller candidates
  must be serialized, replayable, diagnosable, and evaluated through held-out
  artifacts.
- **Trainer Operations -> Mind Controllers**: remote trainer runs heavy CUDA,
  long gate, and sweep work; source changes and review remain local/GitHub
  driven.
- **Project Docs -> All Contexts**: docs record boundaries, ledgers, readiness
  criteria, and decisions that future changes must preserve or explicitly
  update.

## Shared Rules

- Preserve deterministic replay semantics unless a contract update is explicit.
- Keep summary-only lightweight.
- Keep long sweeps opt-in.
- Keep Mind v3 autonomous work opt-in, serialized, deterministic, replayable,
  and held to strict per-seed gates.
- Do not relax action-collapse, heuristic-action, or per-seed gates to make a
  candidate pass.
