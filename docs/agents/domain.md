# Domain Docs

This repo uses a multi-context domain-doc layout. Engineering skills should use
`CONTEXT-MAP.md` at the repo root, when present, to find the relevant
per-context `CONTEXT.md` files for the task.

## Before Exploring, Read These

- `CONTEXT-MAP.md` at the repo root, if present.
- Relevant per-context `CONTEXT.md` files named by `CONTEXT-MAP.md`.
- `docs/adr/` for system-wide decisions, if present.
- Context-scoped `docs/adr/` directories, if present.
- Existing repo docs under `docs/` that are relevant to the task.

If any of these files or directories do not exist, proceed silently. Do not flag
their absence or suggest creating them upfront; domain docs can be produced
later when terms or decisions need to be captured.

## Expected Layout

```text
/
├── CONTEXT-MAP.md
├── docs/
│   └── adr/
└── <context>/
    ├── CONTEXT.md
    └── docs/
        └── adr/
```

For this project, likely contexts include simulator/foundation/replay,
viewer/contracts, Mind v3 controller experiments, trainer workflows, and docs or
spec work. Use the context map as the source of truth once it exists.

## Vocabulary

When output names a domain concept in an issue title, refactor proposal,
hypothesis, or test name, use the term as defined in the relevant `CONTEXT.md`.
If the concept is not defined yet, either use the project's existing docs and
code vocabulary or note the gap for future domain-doc work.

## ADR Conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly
instead of silently overriding the decision.
