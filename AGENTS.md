# AGENTS.md

This repo is a deterministic artificial-life simulator.

Current phase: Mind v3 autonomous-controller experimentation. Foundation,
replay, viewer, trajectory, and Mind v1/v2 guarded safety-floor contracts are
the measurement boundary, not the active research goal. Learned-controller work
must reduce heuristic action selection while preserving deterministic replay,
held-out diagnostics, strict per-seed gates, and Foundation release behavior.

The v61-v63 IQL validation loop closed the current scalar coefficient,
prior-blend, extraction-only, and global actor-bias family as non-promotable.
The v137-v176 audit trail then closed the tiny support-archive /
nearest-neighbor scorer loop as strategically saturated: the central carrion
survival blocker did not move and strict seeds were consumed as support
provenance. v177 restored transition-row data support through exact branch
replay. The next useful branch is v178 transition-row dataset audit before
rollout-context policy capacity, world-model/transition-value capacity, or a
distinct controller/data path.

2026-06-10 repository-audit checkpoint: the v137-v176 source backlog was
preserved, committed, pushed, and paired with a documented `output/mind/`
backup. Keep that durability bar for new Mind v3 work: do not start a new
experiment while its source, package entrypoints, tests, ledger entries, or
digest-referenced artifacts are local-only unless the user explicitly asks for a
docs-only or negative-control slice. See
`docs/repository-audit-2026-06-10-remediation.md`.

## Environment

- Python: 3.14.3 from `.python-version`
- Node: npm scripts are the command entrypoints
- Python package path: `PYTHONPATH=python`
- Determinism: use `PYTHONHASHSEED=0` for simulator commands

Prefer npm scripts over raw Python commands when a script exists.

## Validation Ladder

Fast local checks:
- `npm run sim:test`
- `npm run sim:golden:quick`
- `npm run sim:gate:quick`
- `npm run sim:bench:quick`
- `npm run viewer:contracts:check`
- `npm run viewer:model:test`
- `npm run viewer:map:test`
- `npm run viewer:validate`
- `git diff --check`

Viewer check:
- `npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json`
- `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`

Gate boundary checks:
- `npm run sim:golden`
- `npm run sim:test:full`
- `npm run sim:gate:release`
- `npm run sim:mind:gate -- --reuse-trajectories --fail-on-blockers`
- `npm run sim:mind:gate:extended`
- `npm run sim:bench`

## Project Rules

- Full replay is the compatibility contract.
- Summary-only mode must stay lightweight and must not build viewer/replay payloads.
- Long sweeps are opt-in; use compact contract equivalents in the fast loop.
- Every world mechanic needs matching observability, metrics, tests, and viewer/replay semantics where applicable.
- Mind contracts may evolve only with matching artifact/runtime diagnostics and held-out gates.
- Do not change golden/replay semantics casually.
- Do not make tests poke cache internals or call `_invalidate_biotic_state()` directly.
- Mind v3 promotion requires strict broad held-out seed behavior and controlled fixture evidence; aggregate gains do not excuse per-seed alive or birth regressions.
- Do not relax action-collapse, heuristic-action, or per-seed gates to make a candidate pass. Record the first failing seed, fixture, and action distribution instead.
- Keep novel controller work opt-in, serialized, deterministic, and replayable. Do not add hidden heuristic action selection or fixture-specific runtime identity.
- Do not treat seeds `5,13,19,29,37,41` as clean promotion-heldout evidence for scorers trained or selected using recent support/provenance artifacts that consumed those seeds. Mint and document a fresh promotion-heldout matrix before promotion-style claims.
- Do not hardcode digests of gitignored Mind artifacts without a durable artifact backup path and retrieval note.
- Do not make `python/evolution_sim/cli/` a shared library for new Mind experiment code. Shared fixture, gate, digest, and report helpers belong under `python/evolution_sim/mind/` and CLIs should be thin wrappers.
- The habitat water encoding bug was fixed as a dedicated replay/viewer contract change. Future replay/viewer contract changes need the same tests, docs, viewer handling, and golden regeneration.
- Push-time CI Mind gates are compact contract coverage, not promotion evidence. Strict Mind claims require the explicit strict commands and recorded per-seed blockers.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `Reallisticus/evolution-sim`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default Matt Pocock skills triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Use a multi-context domain-doc layout rooted at `CONTEXT-MAP.md`. See `docs/agents/domain.md`.
