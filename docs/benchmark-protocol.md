# Benchmark Protocol

Use `npm run sim:bench` for repeatable performance measurements.

Use scenario filters for quick local checks:

```bash
npm run sim:gate:quick
npm run sim:bench -- --warmup 0 --runs 1 --scenario summary_seed7_ticks20 --skip-multiprocess
npm run sim:golden:quick
```

## Fixed Protocol

- Warmup runs: `1`
- Measured runs per scenario: `5`
- Reported metrics:
  - median wall time
  - p95 wall time
  - median peak RSS in KiB
  - p95 peak RSS in KiB
  - replay size for full-replay scenarios

Each measured scenario repetition runs in a fresh worker process. Wall time is
measured inside that worker so process startup overhead is excluded, while peak
RSS is isolated to that repetition instead of inherited from earlier scenarios.
RSS is normalized to KiB across platforms.

## Required Machine Profile

Each benchmark report records:

- CPU model
- RAM
- OS version
- Python version

Before/after comparisons are valid only when collected on the same machine profile.

## Scenarios

- Full replay: `seed=7,ticks=20`
- Full replay: `seed=7,ticks=100`
- Full replay: `seed=GOLDEN_SPECIATION_SEED,ticks=320`
- Summary only: the same three scenarios
- Multi-process summary-only rollout batch for future multi-instance training work

## Long-Run Policy

- Benchmarks are opt-in and are not part of the fast development loop.
- Long viability sweeps should run in `SUMMARY_ONLY` unless the assertion is
  specifically about replay taxonomy or viewer payload compatibility.
- Keep a compact full-replay compatibility test for every long summary-only
  production check.
- Stop and inspect any run that has already failed before waiting for an
  expensive tail test to complete.
- Full test runs fail fast by default. Use `--no-failfast` only when collecting
  a complete failure list is worth the extra runtime.
- Store before/after benchmark logs under `output/logs/` and compare only runs
  collected on the same machine profile.

## Foundation Gate

- `npm run sim:gate:quick` is the regular local/CI readiness gate.
- `npm run sim:gate:release` is the opt-in release boundary gate. It includes
  the long summary-only viability sweep and compact full-replay probes for
  species/taxonomy surfaces.
- A `pass` report means the selected profile passed. A `review` report means no
  hard blocker was found, but Foundation warnings must be judged before moving
  toward `Mind v1`. A `fail` report blocks Mind work.

## Scaling Targets

- Prefer semantic-preserving code-path reductions before hardware scaling.
- Precompute static terrain/topology lookups once per world.
- Cache tick-scoped derived environment fields; they are climate-derived and
  must be reset by `reset_derived_caches()`.
- Rebuild biotic fields from sparse source maps so empty terrain is not scanned
  through the diffusion path.
- Keep future mind/intelligence subsystems behind explicit runtime gates so
  contract tests can use deterministic cheap paths.
- Use process-level parallelism across worlds/seeds after single-world behavior
  is stable; do not introduce in-world threading until determinism is protected.
