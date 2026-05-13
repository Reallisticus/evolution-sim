# AGENTS.md

This repo is a deterministic artificial-life simulator.

Current phase: Mind v3 autonomous-controller experimentation. Foundation,
replay, viewer, trajectory, and Mind v1/v2 guarded safety-floor contracts are
the measurement boundary, not the active research goal. Learned-controller work
must reduce heuristic action selection while preserving deterministic replay,
held-out diagnostics, strict per-seed gates, and Foundation release behavior.

The v61-v63 IQL validation loop closed the current scalar coefficient,
prior-blend, extraction-only, and global actor-bias family as non-promotable.
The next useful Mind v3 branch should add rollout-context policy capacity or a
distinct controller/data path, not another scalar tuning pass on the same
single-observation representation.

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
