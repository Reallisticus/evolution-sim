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

- 2026-05-04 Foundation-to-Mind hardening validation:
  `npm run sim:test:full` passed with 281 tests in 1857 seconds,
  `npm run sim:golden` verified all current replay hashes,
  `npm run sim:gate:release` passed with no blockers or warnings in 1082
  seconds, `npm run sim:bench` passed, seed-7 300-tick replay plus
  `viewer:smoke`, `viewer:smoke:visual`, and `viewer:smoke:session` passed,
  `viewer:contracts:check` passed, and `git diff --check` was clean.
- Production-readiness follow-up: keep the heavy release checks intact, but make
  their wall time easier to schedule and inspect. The full suite and release
  gate are now functionally green yet operationally expensive enough to deserve
  explicit CI/runtime-budget treatment.
- Summary-only action-mask work is now a watched runtime-cost invariant:
  benchmark reports should stay near two action-mask builds per observation
  unless a later contract intentionally adds another mask phase.

## Boundary Audit Snapshot

The main risk is hidden coupling through the live `SimulationWorld` object.
Recent extractions created useful module boundaries, but many runtime modules
still call private `world._*` helpers. Treat those calls as migration debt. When
extracting, preserve behavior first, then make the boundary explicit.

Highest-priority remaining leaks after the current Foundation hardening slice:

- `runtime/reproduction.py` now routes private world authority through the
  `ReproductionContext` adapter, including reproductive signal context access.
  Keep future birth planning and biological readiness work context-only.
- `runtime/derived.py`, `runtime/biotic.py`, `runtime/signals.py`,
  `runtime/actions.py`, `runtime/observations.py`, `runtime/surfaces.py`, and
  `runtime/action_space.py` are clean of direct private world reads. Do not
  reintroduce compatibility fallbacks there.
- `runtime/frames.py`, `runtime/summary.py`, `runtime/lifecycle_summary.py`,
  and `runtime/collectors.py` are the next reporting/export authority cluster.
- `runtime/surface_snapshots.py` gathers frame/summary snapshots through a
  single snapshot adapter; reporting should stay on the extracted authority
  modules.
- `runtime/feeding.py` now routes private world authority through the
  `FeedingContext` adapter. Keep telemetry and opportunity accounting split.

Latest mechanical private-call audit after the resource/lifecycle/tick context
and derived/biotic/signal context extractions:

- `runtime/reproduction.py`: 10 private world reads, all in context adapters.
- `runtime/feeding.py`: 10 private world reads, all in the context adapter.
- `runtime/surface_snapshots.py`: 9 private world reads, all in the snapshot
  adapter.
- `runtime/frames.py`: 10 private world reads across frame assembly/reporting
  authority.
- `runtime/summary.py`: 7 private world reads across summary finalization.
- `runtime/lifecycle_summary.py`: 8 private world reads across trophic lifecycle
  aggregation.
- `runtime/collectors.py`: 5 private world reads in collector orchestration
  adapters.
- `runtime/derived.py`: 0 private world reads.
- `runtime/biotic.py`: 0 private world reads.
- `runtime/signals.py`: 0 private world reads.
- `runtime/action_space.py`: 0 private world reads.
- `runtime/actions.py`: 0 private world reads.
- `runtime/observations.py`: 0 private world reads.
- `runtime/surfaces.py`: 0 private world reads.
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
