# Senior Developer Onboarding

Date: 2026-06-10
Audience: senior engineers joining Foundation-boundary and Mind v3 autonomous-controller work.

This repository is a deterministic artificial-life simulator. Foundation,
replay, trajectory, viewer, and guarded Mind v1/v2 contracts are now the
measurement boundary for Mind v3 autonomous-controller experiments. Mind v3 work
is allowed, but it must stay opt-in, deterministic, replayable, and measured
against strict held-out seed, fixture, action-source, and action-collapse gates.

## First Hour

Read in this order:

1. `AGENTS.md` for repo-specific operating rules, determinism requirements, and
   validation ladder.
2. `README.md` for command entrypoints and the current Foundation feature set.
3. `docs/mind-v3-autonomous-evolution.md` for the current autonomous-controller
   ledger, latest non-promotable candidates, and next research boundary.
4. `docs/repository-audit-2026-06-10-remediation.md` for the current audit
   remediation order, source/evidence durability status, and next-coder
   prompt.
5. `docs/pre-mind-reproductive-and-signal-readiness-plan.md`,
   `docs/mind-readiness-audit-2026-04-27.md`, and
   `output/audits/deep-system-audit-2026-05-11.md` for historical findings,
   fixed issues, and remaining measurement risk.
6. `docs/replay-invariants.md` and `docs/benchmark-protocol.md` for durable
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
- `python/evolution_sim/mind/`: Mind data tooling, strict trajectory JSONL
  loading, deterministic seed splits, guarded Mind v1/v2 safety-floor
  artifacts, Mind v3 autonomous policy/runtime adapters, neural/IQL trainers,
  diagnostics, and recovery/archive tools.

## Output And Gate Map

- `python/evolution_sim/io/replay_writer.py`: full-replay JSON writer. Replay
  key order is intentionally preserved.
- `python/evolution_sim/io/trajectory_writer.py`: gzip JSONL trajectory writer.
  Trajectory rows are sorted for stable archival output.
- `viewer/replay_validator.mjs`: browser/Node replay validator. It duplicates
  Python contract expectations today; generated shared schemas remain future
  work.
- `python/evolution_sim/cli/foundation_gate.py`: quick, ecology, and release
  readiness profiles. The release profile is the compatibility boundary for
  Mind work.
- `python/evolution_sim/cli/mind_v3_labeled_iql_slice.py`: labeled Mind v3
  train/evaluation slice with strict broad seed and controlled fixture
  acceptance surfaces.
- `python/evolution_sim/cli/mind_gate.py`: Mind gate runner for guarded
  learned-policy evaluation and blocker reporting.
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
  Mind loader validates the header, every record, and the footer before a
  dataset can be used for offline experiments. Policy-owned rollout context must
  be replayable from public trajectory rows, policy actions, finalized outcomes,
  or explicit update records.
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

Latest validated boundary state:

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
- 2026-05-13 Mind v3 v61-v63 boundary: v61, v62, and v63 are not promotable.
  The current single-observation IQL coefficient/prior/extraction/global-bias
  family is exhausted for promotion because strict deployment slices still show
  action collapse, per-seed alive/birth regressions, and controlled carrion
  movement failures. The next useful branch should add rollout-context policy
  capacity or a distinct sequence, flow, world-model, or archive-replay path,
  with the existing strict gates kept hard.
- 2026-06-10 full-repository audit and remediation: the Foundation measurement
  boundary remains sound. The v137-v176 source backlog has been committed and
  pushed, and digest-referenced `output/mind/` evidence has a documented backup
  path. Keep that durability bar for new work. The next Mind v3 slice should
  follow the v176 exact-branch-replay/transition-row route, not another tiny
  support-scorer probe. Seeds `5,13,19,29,37,41` are no longer clean
  promotion-heldout evidence for scorers trained or selected using recent
  support/provenance artifacts.

## Boundary Audit Snapshot

Do not rely on stale private-call count tables from old onboarding notes. When a
boundary claim matters, regenerate it mechanically for the exact slice under
review and document the command. Treat private `SimulationWorld` access from
policy or trainable Mind inputs as a hard error; controlled diagnostics may
inspect world internals only when the report marks those fields metadata-only
and keeps them out of trainable/runtime payloads.

The 2026-06-10 harness leak has been remediated: shared report/metric helpers
live in `python/evolution_sim/mind/evaluation_helpers.py`, and shared fixture,
gate, digest, report, and leakage-scan helpers live in
`python/evolution_sim/mind/evaluation_harness.py`. Keep new Mind experiment
helpers in the Mind layer; `python/evolution_sim/cli/` should stay parser and
orchestration code, not a shared library.

## Working Rules

- Do not expose human-readable action names, species labels, reproductive-group
  labels, or signal meanings to policy/Mind input.
- Preserve `SUMMARY_ONLY` as lightweight. It must not build viewer/replay
  payloads.
- Preserve full replay as the compatibility contract.
- Keep every new world mechanic paired with metrics, replay/summary output,
  validation, and tests.
- Prefer narrow, behavior-preserving extractions before semantic changes.
- Learned-policy runtime remains disabled by default unless explicitly enabled.
  Mind v3 experiments must stay opt-in, serialized, deterministic, and measured
  against strict broad-seed, controlled-fixture, heuristic-action, action-share,
  and per-seed no-regression gates.
