# Two-Gate Recovery Plan

This plan protects full replay compatibility while adding a separate execution
path for long-running production checks and future training workloads.

## Gate A: Contract And Modularization

- Full replay output is the compatibility contract.
- Replay key order, viewer map order, agent encoding order, and legend ordering
  are frozen in runtime contract tests.
- Golden seed discovery is not allowed; `GOLDEN_SPECIATION_SEED` stays fixed at
  `3`.
- Tests must not poke `cached_*` internals or call `_invalidate_biotic_state()`
  directly.
- Reporting code should move out of `SimulationWorld` into
  `evolution_sim.env.runtime.reporting`.
- `SimulationWorld` may keep private compatibility wrappers, but wrappers should
  delegate to runtime modules instead of owning shared reporting logic.

Gate A closes only when goldens, runtime contracts, replay taxonomy tests, the
viewer smoke check, and the full suite pass.

## Gate B: Runtime Modes

- Keep the existing `RunMode` surface: `FULL_REPLAY` and `SUMMARY_ONLY`.
- CLI replay generation remains full-replay only.
- `SUMMARY_ONLY` must not create replay payloads, viewer payloads, frame
  captures, or replay-taxonomy rewrites.
- `SUMMARY_ONLY` may compute only terminal snapshots needed by
  `SHARED_SUMMARY_FIELDS`.
- Full-only taxonomy/species assertions belong in compact full-replay tests.
- Long multi-seed viability checks belong in `SUMMARY_ONLY`.

## Scaling Addendum

The current benchmark shows that replay removal alone is not enough. On the Mac
baseline, seed 3 at 320 ticks is dominated by core simulation cost, not JSON or
viewer serialization. Before adding expensive agent-mind or intelligence loops,
new work must follow these rules:

- Long sweeps must be explicit, opt-in, and outside the fast suite.
- Every new long check needs a compact contract equivalent for regular runs.
- Each expensive subsystem needs a summary-only bypass or reduced surface when
  it is not part of the assertion.
- Full-suite runs fail fast by default so an early failure does not continue into
  multi-hour tail tests.
- Benchmark scenarios can be filtered for quick local before/after checks.
- Shared expensive topology or geometry should be precomputed once per world and
  reused.
- Per-agent caches must be semantic-preserving and invalidated through runtime
  helpers.
- Any benchmark claim must use one machine profile for before and after.
- Future mind/intelligence execution should run behind its own mode/config gate,
  with deterministic cheap stubs available for contract tests.

## Current Performance Priorities

1. Keep full replay byte-compatible.
2. Make `SUMMARY_ONLY` skip full replay event and tick-detail bookkeeping.
3. Cache static biotic diffusion targets by terrain map and radius.
4. Cache tick-scoped effective environment fields and route terrain lookups
   through static topology.
5. Use sparse biotic source maps so empty terrain does not pay diffusion cost.
6. Add quick benchmark, quick golden, and fail-fast test-runner controls.
7. Keep moving reporting assembly out of `SimulationWorld`.
8. Add process-level parallelism only after single-world semantics are stable.

## Verification Policy

Low-cost checks during implementation:

- `PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_runtime_contracts.py'`
- `PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_replay_taxonomy.py'`
- Targeted compact headless tests with `-k`
- `npm run sim:bench:quick`
- `npm run sim:golden:quick`
- `python3 -m py_compile` for touched modules
- `git diff --check`

Gate boundary checks:

- `npm run sim:golden`
- `npm run sim:test`
- `npm run sim:test:full`
- `npm run viewer:smoke`
- `npm run sim:bench`
