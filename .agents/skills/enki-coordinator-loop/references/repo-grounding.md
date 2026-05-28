# ENKI Repo Grounding

This reference captures the source-grounded audit used to create the ENKI
coordinator loop on 2026-05-24. Refresh facts from source before making claims;
this file is a map, not authority.

## Inventory

Tracked source inventory from `git ls-files`:

- 266 tracked files total.
- 162 Python source files under `python/evolution_sim/`.
- 50 Python test/support files under `python/tests/`.
- 23 viewer files under `viewer/`.
- 18 tracked docs under `docs/`.
- 4 project skills under `.agents/skills/` before ENKI was added.

Generated or heavy local state exists and is not the source of truth:

- `output/` is large run/artifact history.
- `.venv/` and `node_modules/` are dependencies.
- `python/tests/goldens/` contains tracked replay/golden artifacts. Treat these
  as compatibility artifacts, not normal source to rewrite casually.

## Foundation Spine

Key files:

- `python/evolution_sim/config/schema.py`: dataclass-backed world configuration
  with validation at config boundaries.
- `python/evolution_sim/env/world.py`: `SimulationWorld`, the large one-shot
  orchestration object. `run()` enforces one-shot semantics, selects
  `RunMode.FULL_REPLAY` or `RunMode.SUMMARY_ONLY`, and delegates tick execution
  and collection.
- `python/evolution_sim/env/runtime/state.py`: shared runtime data types,
  including `RunMode`, `Agent`, `Tile`, and `SimulationWorldResult`.
- `python/evolution_sim/env/runtime/ticks.py`: deterministic tick loop. It
  resets tick state, invalidates biotic state, captures observations and masks,
  uses deterministic turn ordering, resolves actions, applies metabolism,
  hazards, reproduction, death, trajectory records, and tick completion.
- `python/evolution_sim/env/runtime/collectors.py`: separates
  `FullReplayCollector` from `SummaryCollector`.
- `python/evolution_sim/env/runtime/actions.py`: action scoring, decision
  context, resolution context, invalid-action reasons, and outcome resolution.
- `python/evolution_sim/env/runtime/policy.py`: `Policy` protocol and
  `ObservationHeuristicPolicy`. This is the heuristic source that Mind v3
  autonomous candidates must not secretly depend on.
- `python/evolution_sim/env/runtime/observations.py`: observation contract and
  encoder. Audited constants include `OBSERVATION_SCHEMA_VERSION =
  mind_observation_v3`, `OBSERVATION_ENCODER_VERSION =
  mind_observation_encoder_v2`, and a generated viewer input vector size of
  `542`.
- `python/evolution_sim/env/runtime/trajectory.py`: trajectory contract and
  per-agent record schema. Audited constants include
  `TRAJECTORY_SCHEMA_VERSION = mind_trajectory_v1`.
- `python/evolution_sim/io/replay_writer.py`: full-replay payload construction
  and atomic replay writes.
- `python/evolution_sim/io/trajectory_writer.py`: JSONL/JSONL.gz trajectory
  streaming with header, records, footer, provenance digest, summaries, and
  atomic replace.

Foundation constraints:

- Full replay is the compatibility contract.
- Summary-only must stay lightweight and avoid viewer/replay payload work.
- Public contracts matter more than private caches.
- Fixed-seed determinism matters across simulator, trajectory, and viewer
  surfaces.

## Gates And Evaluation

Key files:

- `python/evolution_sim/cli/test_runner.py`: custom test runner. It filters slow
  tests for the fast suite, enforces runtime budgets, and rejects direct
  `_invalidate_biotic_state` or `cached_*` test poking.
- `python/evolution_sim/cli/foundation_gate.py`: quick/ecology/release profiles,
  full-replay probes, summary seed workers, replay size checks, and blocker
  reporting.
- `python/evolution_sim/cli/foundation_gate_validators/trajectory.py`: validates
  summary/viewer Mind contract metadata, trajectory versions, observation
  contract, signal/action capacity, reproductive group catalog, and trajectory
  records.
- `python/evolution_sim/cli/evaluate.py`: multi-seed evaluation aggregation.
- `python/evolution_sim/cli/bench.py`: benchmark scenarios and process isolation.
- `python/evolution_sim/cli/golden_harness.py`: golden scenario verification and
  update path.

Preferred command entrypoints come from `package.json`, not ad-hoc raw commands:

- `npm run sim:test`
- `npm run sim:test:full`
- `npm run sim:golden:quick`
- `npm run sim:golden`
- `npm run sim:gate:quick`
- `npm run sim:gate:release`
- `npm run sim:bench:quick`
- `npm run sim:bench`

CI in `.github/workflows/ci.yml` uses Python 3.14.3, Node 20, `npm ci`, viewer
contract/model/map/validate checks, golden verification, the fast simulator
suite, quick foundation gate with blockers, a small Mind gate, and replay/viewer
smoke. Expensive release checks are scheduled/manual.

## Mind Spine

Key files:

- `python/evolution_sim/mind/contracts.py`: Mind contract versions and
  `MIND_RUNTIME_ENABLED_DEFAULT = False`.
- `python/evolution_sim/mind/artifacts.py`: artifact loading and version checks.
  Loading requires `enable_mind=True`; artifact/trajectory/observation/policy/
  action/provenance versions are validated.
- `python/evolution_sim/mind/dataset.py`: trajectory JSONL loading, record
  validation, transition segmentation, discounted returns, and dataset
  provenance.
- `python/evolution_sim/mind/baseline.py`: baseline trainers and guarded model
  metadata.
- `python/evolution_sim/mind/learned_policy.py`: guarded, autonomous, and
  autonomous-online learned policy modes. Guard/delegate routes can use the
  heuristic and must be diagnosed.
- `python/evolution_sim/mind/evaluation.py` and `python/evolution_sim/mind/gates.py`:
  learned-vs-heuristic comparison and Mind gate blockers/warnings.
- `python/evolution_sim/mind/torch_trainer.py`: torch actor/critic/IQL training,
  device selection, calibration, support filtering, counterfactual labels, and
  diagnostics.

Mind v3 files:

- `python/evolution_sim/mind/evolution.py`: controller metadata, architecture
  eligibility, policy-visible architectures, rollout/recovery hidden units, and
  promotion eligibility checks.
- `python/evolution_sim/mind/v3_policy.py`: heuristic-free autonomous policy
  runtime. It supports linear controllers, frozen neural residuals,
  planner-distilled runtime, support residual runtime, rollout context, recovery
  context, diagnostics, and policy-owned update traces.
- `python/evolution_sim/mind/rollout_context.py`: policy-owned public rollout
  context derived from previous trajectory rows and finalized outcomes.
- `python/evolution_sim/mind/recovery_context.py`: recovery context feature
  contract and update trace.
- `python/evolution_sim/cli/mind_v3_evaluate.py`: broad and controlled fixture
  evaluation. Audited promotion-style constants include broad seeds
  `5,13,19,29,37,41`, current fixture seeds `13,19,29,37,41,43`, `120` ticks,
  maximum dominant requested-action share `0.50`, and zero heuristic
  action-source expectation for autonomous candidates.
- `python/evolution_sim/cli/mind_gate.py`: train/evaluate gate. Audited strict
  control targets include hard guard rate `0.1190` and total fallback rate
  `0.4780`; promotion-review targets include hard guard rate `0.0568` and total
  fallback rate `0.4740`.

Current Mind direction from source and project instructions:

- The existing single-observation IQL coefficient/prior/extraction/global
  actor-bias family is closed as non-promotable unless explicitly requested as a
  negative control.
- Useful next work is contract-first rollout-context capacity, exact branch
  replay/archive paths, sequence/world-model style branches, or diagnostics that
  falsify state aliasing and support gaps before heavy training.
- Train-gate success alone is not promotion evidence.

## Viewer Spine

Key files:

- `python/evolution_sim/env/viewer_contracts.py`: Python source of viewer
  contract constants.
- `python/evolution_sim/cli/viewer_contracts.py`: generates/checks
  `viewer/contracts.generated.mjs`.
- `viewer/contracts.generated.mjs`: generated module; do not edit by hand.
- `viewer/replay_validator.mjs`: payload, summary, map, agent encoding,
  trajectory, catalog, frame, and version validation.
- `viewer/replay_model.mjs`: decoded frames, events by tick, significant event
  ticks, episodes, agent event index, positions, and trajectory records by
  agent.
- `viewer/dashboard_model.mjs`: overlay contracts, comparison baselines,
  preferences, URL state, and episode filtering.
- `viewer/episode_model.mjs`: event lens categories, episode scope, role
  summaries, ledger entries, storyboard export model, and tile explanations.
- `viewer/map_layers.mjs`: terrain, agent, trail, decision overlay, event map,
  species halos, selected life paths, deltas, and marker rendering.
- `viewer/app.js`: large browser UI orchestration over Pixi and model helpers.
- `viewer/smoke.mjs`, `viewer/visual_smoke.mjs`,
  `viewer/malformed_smoke.mjs`, `viewer/session_smoke.mjs`,
  `viewer/validator_smoke.mjs`: replay validation and browser smoke surfaces.

Viewer commands:

- `npm run viewer:contracts:check`
- `npm run viewer:model:test`
- `npm run viewer:map:test`
- `npm run viewer:validate`
- `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`
- `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke:visual`

## Test Distribution

The Python test surface is broad. Audited high-count modules include:

- `python/tests/test_mind_v1.py`: 306 tests.
- `python/tests/test_headless_sim.py`: 52 tests.
- `python/tests/test_runtime_action_contracts.py`: 46 tests.
- `python/tests/test_foundation_gate_cli.py`: 41 tests.
- `python/tests/test_runtime_reproduction_contracts.py`: 40 tests.
- `python/tests/test_runtime_feeding_contracts.py`: 26 tests.
- `python/tests/test_mind_v3_carrion_recovery_distill.py`: 23 tests.
- `python/tests/test_runtime_trajectory_contracts.py`: 22 tests.
- `python/tests/test_runtime_signal_contracts.py`: 21 tests.
- `python/tests/test_mind_v3_rollout_context_controller.py`: 16 tests.

Treat `python/tests/test_foundation_gate_cli.py` as a key integration contract
surface: it exercises stale Mind contracts, stale observation/action/signal
versions, reproductive catalogs, replay size, gate flags, timeout handling, and
summary-only behavior.

## Existing Project Skills

- `.agents/skills/evolution-sim-work/SKILL.md`: default repo implementation
  workflow.
- `.agents/skills/foundation-gate-loop/SKILL.md`: Foundation measurement and
  gate loop.
- `.agents/skills/mind-contract-loop/SKILL.md`: Mind interface and trajectory
  contract loop.
- `.agents/skills/mind-v3-autonomous-loop/SKILL.md`: Mind v3 autonomous
  controller loop.

ENKI should coordinate those skills. It should not replace their narrower
contract rules.

## Practical Validation Ladder

Targeted first:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.<module>
node viewer/<model_or_map_test>.mjs
```

Then broader:

```bash
npm run viewer:contracts:check
npm run viewer:model:test
npm run viewer:map:test
npm run sim:test
npm run sim:golden:quick
npm run sim:gate:quick
npm run sim:bench:quick
git diff --check
```

Release or promotion work:

```bash
npm run sim:golden
npm run sim:test:full
npm run sim:gate:release
npm run sim:mind:gate -- --reuse-trajectories --fail-on-blockers
npm run sim:mind:gate:extended
npm run sim:bench
```

Mind v3 promotion-style slice:

```bash
npm run sim:mind:v3:labeled-iql-slice -- \
  --candidate-artifact <artifact.json> \
  --enable-mind \
  --mind-runtime-mode autonomous \
  --seeds 5,13,19,29,37,41 \
  --ticks 120 \
  --fixture-names carrion_only \
  --fixture-seeds 13,19,29,37,41,43 \
  --fixture-ticks 120 \
  --output <slice-report.json>
```

## Repo-Specific Failure Modes

Watch for:

- docs saying a contract exists when source/gates do not enforce it;
- changes that make `SUMMARY_ONLY` build full replay/viewer state;
- tests that pass only by mutating private caches;
- viewer generated contracts drifting from Python contract constants;
- Mind artifacts that load without version/provenance checks;
- policy features that include agent ids, fixture ids, lineage/debug metadata,
  or private world state;
- autonomous Mind v3 reports with nonzero heuristic action-source count;
- dominant action collapse hidden behind average metrics;
- broad seed success without controlled fixture evidence;
- artifact/report paths missing from final reports;
- long trainer work without a log command and fetch path.
