# Benchmark Protocol

Use `npm run sim:bench` for repeatable performance measurements.

Use scenario filters for quick local checks:

```bash
npm run sim:gate:quick
npm run sim:bench -- --warmup 0 --runs 1 --scenario summary_seed7_ticks20 --skip-multiprocess
npm run sim:golden:quick
```

## CLI Argument Constraints

The benchmark CLI validates protocol arguments before running any scenario:

- `--warmup` must be an integer greater than or equal to `0`.
- `--runs` must be an integer greater than `0`.
- `--scenario-timeout-seconds` must be a finite positive number.
- `--multiprocess-timeout-seconds` must be a finite positive number.
- `--scenario` may be repeated, but each value must be one of the named
  protocol scenarios exposed by `python/evolution_sim/cli/bench.py`.

Invalid protocol values exit with argparse status `2`, write no JSON report to
stdout, and put the validation message on stderr.

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

## Report Contract

Benchmark reports are JSON documents with a top-level completion flag:

- `complete: true` means every selected scenario and the optional multi-process
  rollout completed.
- `complete: false` means the command caught a timeout or runtime failure and
  wrote a partial report.

Every report contains:

- `protocol`: warmup count, measured run count, machine profile, RSS units, and
  scenario repetition isolation mode.
- `complete`: boolean completion flag.
- `scenarios`: completed scenario metric objects. This list is empty if the
  first selected scenario times out before a measured result is available.
- `multi_process_summary_rollout`: rollout metrics or `null` when skipped or
  not reached.

Each completed scenario records its scenario identity plus median/p95 wall time,
median/p95 peak RSS, replay size when applicable, trajectory record/output
counts, and median runtime cost counters.

Runtime cost counters are part of the benchmark regression surface. Summary-only
scenarios currently assert:

- action-mask builds stay near two mask builds per observation;
- `resource_pressure_accounting_updates` remains within the grid-plus-agent
  tick budget, so plant-budget and energy-spend accounting cannot silently grow
  into a replay-sized side path.

Partial reports keep any completed `scenarios` plus the selected machine
profile and include an `error` object with:

- `phase`: scenario/run phase or multi-process rollout phase that failed;
- `type`: exception class name;
- `message`: human-readable failure reason;
- `completed_scenarios`: number of scenarios with completed measurements;
- `requested_scenarios`: total selected scenario count.

For multi-process rollout timeouts, the `message` includes completed worker
count and pending seeds.

Timeouts should therefore be treated as usable diagnostic reports, not as
silent benchmark stalls. Automation should fail the build or gate when
`complete` is false, but still archive the partial JSON for analysis.

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
- Release reports consume shared summary analytics for carrying-capacity
  pressure, resource-pressure budgets, and initial-to-terminal trait
  distributions. Sustained max-agent saturation is warning/fail gated through
  `carrying_capacity.at_cap_tick_share`. Pathological plant-budget collapse is
  review-gated through ending plant energy per land tile, and missing terminal
  selection signal is review-gated through
  `selection_heredity.terminal_minus_initial_mean`.

## Scaling Targets

- Prefer semantic-preserving code-path reductions before hardware scaling.
- Treat summary-only action-mask builds as a benchmarked cost surface. The
  current expected shape is roughly two mask builds per observation: one mask in
  observation/action selection and one live resolution mask. A third per-agent
  mask phase should be treated as a regression unless a new contract explicitly
  requires it.
- Treat resource-pressure accounting as a watched cost surface. Updating plant
  and energy-spend totals is allowed; building replay/viewer payloads or
  scanning extra replay surfaces to compute those totals is not.
- Precompute static terrain/topology lookups once per world.
- Cache tick-scoped derived environment fields; they are climate-derived and
  must be reset by `reset_derived_caches()`.
- Rebuild biotic fields from sparse source maps so empty terrain is not scanned
  through the diffusion path.
- Keep future mind/intelligence subsystems behind explicit runtime gates so
  contract tests can use deterministic cheap paths.
- Use process-level parallelism across worlds/seeds after single-world behavior
  is stable; do not introduce in-world threading until determinism is protected.
