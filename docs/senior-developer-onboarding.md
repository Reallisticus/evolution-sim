# Senior Developer Onboarding

Date: 2026-05-02
Audience: senior engineers joining the Foundation hardening work before Mind v1.

This repository is a deterministic artificial-life simulator. The current goal
is not to start learned-controller work. The current goal is to stabilize the
Foundation: ecology, reproduction, signals, action/observation/reward contracts,
trajectory data, replay semantics, gates, and benchmark protocol.

## First Hour

Read in this order:

1. `AGENTS.md` for repo-specific operating rules, determinism requirements, and
   validation ladder.
2. `README.md` for command entrypoints and the current Foundation feature set.
3. `docs/pre-mind-reproductive-and-signal-readiness-plan.md` for the staged
   reproduction/signal plan that blocks Mind v1.
4. `docs/mind-readiness-audit-2026-04-27.md` for historical findings, fixed
   issues, and current remaining risk.
5. `docs/replay-invariants.md` and `docs/benchmark-protocol.md` for durable
   output contracts.

Do not start by editing `world.py`. Read it as the runtime spine, then move to
the runtime modules that now own specific behavior.

## Runtime Map

- `python/evolution_sim/config/schema.py`: dataclass config objects and
  validation boundary. Invalid worlds should fail here or at world construction,
  not halfway through a run.
- `python/evolution_sim/env/world.py`: compatibility spine for world state,
  environment generation, and remaining orchestration. It is intentionally being
  reduced through behavior-preserving extractions.
- `python/evolution_sim/env/runtime/ticks.py`: tick-level orchestration boundary.
- `python/evolution_sim/env/runtime/actions.py` and
  `python/evolution_sim/env/runtime/action_space.py`: action selection and
  resolution surfaces, including resolution-mask invalid action outcomes.
- `python/evolution_sim/env/runtime/action_contract.py`: policy-facing action
  contract, reserved slots, masks, and versioned metadata.
- `python/evolution_sim/env/runtime/feeding.py`: diet totals, feeding events,
  meat intake side effects, and animal-resource opportunity counters.
- `python/evolution_sim/env/runtime/resources.py`: vegetation, fresh-kill, and
  carcass resource mechanics, including death resource emission.
- `python/evolution_sim/env/runtime/lifecycle.py`: metabolism, health/hazard
  accounting, damage telemetry, and death-cause accounting.
- `python/evolution_sim/env/runtime/observations.py`: policy observation
  contract and canonical observation encoding.
- `python/evolution_sim/env/runtime/policy.py`: default observation-only
  heuristic boundary. Future learned policies should enter through this shape,
  not through private world reads.
- `python/evolution_sim/env/runtime/reproduction.py`: reproductive readiness,
  mate search orchestration, child-birth planning, single-child and
  multi-offspring birth mutation, parent costs, and event payloads.
- `python/evolution_sim/env/runtime/mating.py`: mate-search diagnostics and
  compatibility constraints.
- `python/evolution_sim/env/runtime/signals.py`: reproductive and communication
  signal substrate, decay, source provenance, and event capture.
- `python/evolution_sim/env/runtime/frames.py`: full-replay frame assembly.
- `python/evolution_sim/env/runtime/summary.py`: summary assembly boundary.
- `python/evolution_sim/env/runtime/reporting.py`: shared summary/frame
  reporting finalizers.
- `python/evolution_sim/env/runtime/surfaces.py`: full-replay frame surface
  materialization.
- `python/evolution_sim/env/runtime/trajectory.py`: trajectory finalization and
  action-outcome data contract.
- `python/evolution_sim/mind/`: disabled-by-default Mind v1 data tooling,
  strict trajectory JSONL loading, deterministic seed splits, behavior-cloning
  baseline artifacts, learned-policy adapter, and summary-only policy
  evaluation gates.

## Output And Gate Map

- `python/evolution_sim/io/replay_writer.py`: full-replay JSON writer. Replay
  key order is intentionally preserved.
- `python/evolution_sim/io/trajectory_writer.py`: gzip JSONL trajectory writer.
  Trajectory rows are sorted for stable archival output.
- `viewer/replay_validator.mjs`: browser/Node replay validator. It duplicates
  Python contract expectations today; generated shared schemas remain future
  work.
- `python/evolution_sim/cli/foundation_gate.py`: quick, ecology, and release
  readiness profiles. The release profile is the real boundary before Mind
  work.
- `python/evolution_sim/cli/bench.py`: benchmark protocol runner. Successful
  reports set `complete: true`; timeout/error reports set `complete: false` and
  include partial scenario data plus an `error` object.

## Stabilized Export Contracts

Treat these as separate contracts. A fix in one must not silently change the
others.

- Full replay JSON is the compatibility contract for viewer playback. It owns
  events, viewer frames, replay taxonomy, species catalogs, and full trajectory
  retention when enabled.
- Summary-only mode is lightweight. It may compute summary metrics and optional
  streamed trajectory records, but it must not call frame capture, viewer
  builders, or replay taxonomy rewriting. Shared summary exports include
  `summary_schema_version=foundation_summary_v1`, carrying-capacity pressure,
  resource-pressure budgets, and selection/heredity trait distributions; these
  metrics must remain available without replay payloads.
- Trajectory JSONL is independent of replay. Its header carries the trajectory,
  observation, action, reward, reproductive-group, recombination, and signal
  contracts. Its footer carries a trajectory summary plus the run summary. The
  Mind v1 loader validates the header, every record, and the footer before a
  dataset can be used for offline experiments.
- Foundation gate JSON reports always include `complete`, `readiness`,
  `summary_gate_flags`, and `timings`. `readiness.blockers` contains error
  flags, `readiness.warnings` contains warning flags, and partial/incremental
  reports use `status: running`. The release profile also reviews sustained
  max-agent saturation through `carrying_capacity.at_cap_tick_share`,
  pathological plant-budget collapse through resource-pressure budgets, and
  missing terminal selection signal through selection/heredity deltas.
- Benchmark JSON reports always include `protocol`, `complete`, `scenarios`,
  and `multi_process_summary_rollout`. Timeout or runtime failures set
  `complete: false` and include an `error` object while preserving completed
  scenario results. Summary-only benchmark regressions also watch action-mask
  build ratio and `resource_pressure_accounting_updates`.

## Normal Validation Ladder

Use npm scripts unless you are doing a focused local probe:

```bash
npm run sim:test
npm run sim:golden:quick
npm run sim:gate:quick
npm run sim:bench:quick
git diff --check
```

At release or contract boundaries, add:

```bash
npm run sim:golden
npm run sim:test:full
npm run sim:gate:release
npm run sim:bench
npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json
REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke
```

Latest validated Foundation boundary state:

- 2026-05-02 full boundary pass: `npm run sim:test:full` passed with 242
  tests, `npm run sim:golden` verified all current replay hashes, `npm run
  sim:bench` completed with seven scenarios, seed-7 300-tick replay plus
  `viewer:smoke` passed, and `git diff --check` was clean.
- `npm run sim:gate:release -- --output
  output/evaluations/foundation-release-current.json` passed with no blockers
  or warnings in `881.0835s`. The previous seed-17 timeout blocker is fixed;
  seed 17 completed in `235.198s` under the `600s` scenario timeout.
- Summary-only action-mask work is now a watched runtime-cost invariant:
  benchmark reports should stay near two action-mask builds per observation
  unless a later contract intentionally adds another mask phase.

## Boundary Audit Snapshot

The main risk is hidden coupling through the live `SimulationWorld` object.
Recent extractions created useful module boundaries, but many runtime modules
still call private `world._*` helpers. Treat those calls as migration debt. When
extracting, preserve behavior first, then make the boundary explicit.

Highest-priority remaining leaks after the current Foundation hardening slice:

- `runtime/reproduction.py` still owns birth placement through several private
  world mutation helpers and remains the largest runtime boundary leak.
- `runtime/actions.py` has been reduced to three compatibility fallback reads;
  keep future action scoring/resolution work on explicit contexts.
- `runtime/observations.py`, `runtime/surfaces.py`, and `runtime/action_space.py`
  each have a single compatibility adapter read left.
- `runtime/reporting.py` still gathers frame/summary snapshots through private
  world wrappers.
- `runtime/feeding.py` still reaches into private source-species, matched-diet,
  and tile-summary helpers.

Latest mechanical private-call audit after the resource/lifecycle/tick context
extractions:

- `runtime/reproduction.py`: 21 private world reads, 10 unique helpers.
- `runtime/actions.py`: 3 private world reads, 3 unique helpers.
- `runtime/observations.py`: 1 private world read, 1 unique helper.
- `runtime/surfaces.py`: 1 private world read, 1 unique helper.
- `runtime/action_space.py`: 1 private world read, 1 unique helper.
- `runtime/resources.py`: 0 private world reads.
- `runtime/ticks.py`: 0 private world reads.
- `runtime/lifecycle.py`: 0 private world reads.

Do not paper over these with new world wrappers. Convert the next boundary leak
by passing explicit runtime arguments or moving the authority into the runtime
module that owns the behavior.

## Working Rules

- Do not expose human-readable action names, species labels, reproductive-group
  labels, or signal meanings to policy/Mind input.
- Preserve `SUMMARY_ONLY` as lightweight. It must not build viewer/replay
  payloads.
- Preserve full replay as the compatibility contract.
- Keep every new world mechanic paired with metrics, replay/summary output,
  validation, and tests.
- Prefer narrow, behavior-preserving extractions before semantic changes.
- Learned-policy runtime remains disabled by default. Use `sim:mind:split` and
  `sim:mind:evaluate -- --enable-mind` only after Foundation export contracts
  are stable for the scenario being evaluated.
