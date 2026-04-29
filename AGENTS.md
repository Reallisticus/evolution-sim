# AGENTS.md

This repo is a deterministic artificial-life simulator.

Current phase: Foundation hardening before Mind v1. Do not start learned-controller work until Foundation ecology, observation/action contracts, trajectory data, reward components, release gates, and benchmark metrics are stable.

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
- `git diff --check`

Viewer check:
- `npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json`
- `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`

Gate boundary checks:
- `npm run sim:golden`
- `npm run sim:test:full`
- `npm run sim:gate:release`
- `npm run sim:bench`

## Project Rules

- Full replay is the compatibility contract.
- Summary-only mode must stay lightweight and must not build viewer/replay payloads.
- Long sweeps are opt-in; use compact contract equivalents in the fast loop.
- Every world mechanic needs matching observability, metrics, tests, and viewer/replay semantics where applicable.
- Do not freeze Mind contracts around unstable Foundation ecology.
- Do not change golden/replay semantics casually.
- Do not make tests poke cache internals or call `_invalidate_biotic_state()` directly.
