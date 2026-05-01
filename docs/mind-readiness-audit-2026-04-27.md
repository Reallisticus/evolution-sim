# Mind Readiness Audit

Date: 2026-04-27
Latest update: 2026-05-01
Scope: `python/evolution_sim/`, `python/tests/`, `viewer/`, `docs/`, CI, and npm entrypoints in the current working tree.

This audit cross-checks the 2026-04-25 pre-Mind report against the current code. Several findings from that report are already fixed on this branch. The important remaining conclusion is narrower now: Foundation is not yet ready for Mind v1 because quick-profile ecology, the 120-tick ecology seed bank, release-horizon validation, and baseline Mind data contracts are in place, but the new reproduction and signal substrate still needs additional staged semantics, release-scale pressure, and contract hardening before learned-controller work starts.

## 2026-04-30 Update: Release Gate Closed, Reproductive/Signal Cleanup Blocks Mind

The latest audit pass independently confirmed the current validation claims:

- `npm run sim:test:full`: pass, 166 tests in 1454.967s.
- `npm run sim:golden`: pass, including compact full-replay fixture sizes of `3904126`, `19095623`, `191055154`, and `29893799` bytes.
- `npm run sim:gate:release -- --output output/evaluations/foundation-release-audit.json`: pass in 828.135s with no blockers or warnings. The summary sweep took 763.3447s.
- `npm run sim:bench`: pass. Full speciation replay remains the heavy case at `65.9557s`, `618624` KiB RSS, `191055154` replay bytes, and `40915` trajectory records. Summary-only speciation remains simulation-heavy at `45.5399s` and `99728` KiB RSS. Streaming trajectory overhead remains modest: seed 7, 100 ticks reported `3.1063s` and `1301917` output bytes versus `2.2445s` for summary-only.
- Viewer smoke: pass, `viewer_smoke_ok selected_agent=1 final_frame=299`.
- `git diff --check`: pass.

The older release-gate operability warning below is now historical evidence, not
the current gate status. Release validation has completed successfully.

The next blocker is no longer "can the release gate finish?" It is now the
pre-Mind reproductive and signal semantics captured in
[`pre-mind-reproductive-and-signal-readiness-plan.md`](pre-mind-reproductive-and-signal-readiness-plan.md).
Sexed reproduction, reproductive groups, reproductive-readiness pheromones,
opaque communication tokens, reserved mate/communication action slots, and
learned-state inheritance metadata are deliberate Foundation work. The first
reproduction and signal slices are now implemented; the remaining work is to
keep adding staged semantics, measuring release-scale behavior, and representing
each exposed surface in observation/action/reward/replay contracts before Mind
v1.

## Verification Run

Latest end-of-batch validation after the animal-resource floor, hunter/carrion acquisition, and carrion-navigation slice:

- `python3 -m compileall -q python/evolution_sim python/tests`: pass.
- Focused runtime/headless/gate regressions for hunter/scavenger feeding, carrion navigation, and the animal-resource floor: pass.
- `npm run sim:test`: pass, 113 tests in 110.127s.
- `PYTHONHASHSEED=0 PYTHONPATH=python python3 -m evolution_sim.cli.golden_harness --update --all`: refreshed all replay goldens after deterministic policy/feeding changes.
- `npm run sim:golden`: pass. Verified `seed7_ticks20=2c52e3790af852968fa13a1fbc12bc3d309f89f251b19d192bd603fc6229a259`, `seed7_ticks100=663f5936b673e722c36648f623293d29840ec923a429ea1bb408d6ba587e1da6`, `speciation_seed_ticks320=78f038c27cefc7c9ed15820e8ae66a1c76cee2c7eaaca145cb78c4b3669f914e`, and `seed5_ticks120_provenance=a46af00fcb3301c8069caf609d8100af41c9ba47492cab79164dfd62ee94e881`.
- `npm run sim:gate:quick -- --fail-on-blockers`: pass in 3.641s.
- `npm run sim:gate:ecology -- --fail-on-blockers`: pass in 59.150s with no blockers or warnings. The ecology profile now enforces `min_animal_resource_consumption_run_share_by_mode=0.5`; release profile is configured at `0.75`.
- The previously tracked run-level no-animal-resource-consumption seeds are now cleared: no ecology seed has zero animal-resource consumption across all animal modes. By mode, remaining no-consumption cases are hunter `1/20` (`15`), mixed `4/13` (`9`, `14`, `15`, `16`), and scavenger `7/20` (`1`, `2`, `4`, `5`, `6`, `7`, `8`). These clear the ecology floor with consuming-run shares of hunter `19/20`, mixed `9/13`, and scavenger `13/20`.
- The remaining no-consumption cases are not global resource-absence cases. Present-but-never-reachable cases are mixed seed `14` and scavenger seeds `2` and `6`; reachable-but-unconsumed cases are hunter seed `15`, mixed seeds `9`, `15`, `16`, and scavenger seeds `1`, `4`, `5`, `7`, `8`.
- The same ecology report still shows animal-specialist reproduction weakness. Parent births by meat mode were `915 none`, `16 mixed`, `6 scavenger`, and `1 hunter`; terminal biologically ready agents were still `0` for hunter, mixed, and scavenger. Late-window presence by run improved to hunter `17/20`, mixed `12/20`, scavenger `20/20`, and none `20/20`.
- `npm run sim:bench:quick`: pass. `summary_seed7_ticks20` reported `median_wall_seconds=0.548`, `median_peak_rss_kib=91648`, and runtime cost counters.
- `git diff --check`: pass.
- `git diff --check`: pass.

Previous end-of-batch validation after the reproduction-readiness diagnostic, meat-specialist diet-match, animal-mode reproduction-energy, and diet-aggregate reporting slices:

- `python3 -m compileall -q python/evolution_sim python/tests`: pass.
- `PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_headless_sim.HeadlessSimulationTests.test_animal_mode_reproduction_energy_requirement_uses_configured_multiplier python.tests.test_headless_sim.HeadlessSimulationTests.test_meat_specialist_diet_match_discounts_plant_fallback python.tests.test_runtime_contracts.RuntimeContractTests.test_world_config_rejects_invalid_ranges_early python.tests.test_runtime_contracts.RuntimeContractTests.test_reproduction_biological_blockers_are_grouped_by_role_and_mode python.tests.test_evaluate_cli python.tests.test_foundation_gate_cli`: pass, 20 tests in 3.406s.
- `PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_evaluate_cli python.tests.test_foundation_gate_cli`: pass, 16 tests in 3.086s after the reporting-only diet aggregate slice.
- `npm run sim:test`: pass, 92 tests in 110.815s.
- `PYTHONHASHSEED=0 PYTHONPATH=python python3 -m evolution_sim.cli.golden_harness --update --all`: refreshed all replay goldens after deterministic summary/replay schema changes.
- `npm run sim:golden`: pass. Verified `seed7_ticks20=5df2734be42365becc402ec3a76962c469442875517b1100fc0a9a4d8a0189db`, `seed7_ticks100=7d8bca815a53fe8ce123a5886b8f9e6ee8715258f87b887b480b979be33de7c5`, `speciation_seed_ticks320=d86316f5640e539bf17e663f5f383953fcf238c09c77ade3d12b31816fcb3b86`, and `seed5_ticks120_provenance=cf8d98383040d403e32d8cb5ddbdca504759f115719ae85d69dfd0fe8c775d2d`.
- `npm run sim:gate:quick -- --fail-on-blockers`: pass in 3.5483s.
- `npm run sim:gate:ecology -- --fail-on-blockers`: pass in 56.4069s. The profile ran seeds `1-20` for `120` summary-only ticks and wrote `output/evaluations/foundation-ecology-current.json` with status `pass`, no blockers, and no warnings.
- Current ecology gate criteria require at least two trophic roles and one animal-resource meat mode per run using the late-window population floor, plus at least two animal-resource meat modes across the aggregate seed bank. Terminal aggregate counts were `7 carnivore`, `1020 herbivore`, `67 omnivore`, with meat modes `19 hunter`, `24 mixed`, `1020 none`, and `31 scavenger`.
- The same ecology report still shows animal-specialist reproduction weakness, but the animal-mode reproduction energy multiplier improved parent births. Across 20 runs, parent births were `5 carnivore`, `916 herbivore`, `21 omnivore`; parent meat-mode births were `3 hunter`, `20 mixed`, `916 none`, `3 scavenger`. Four seeds (`4`, `10`, `15`, `20`) still had no animal-resource consumption, and animal modes still had `0` terminal biologically ready agents.
- Reproduction-readiness diagnostics show energy remains the primary terminal bottleneck for animal modes even after the multiplier. At terminal state, hunters had `8.5250` total energy against `15.9915` required with `7.8512` shortfall; scavengers had `22.9998` against `26.1674` with `4.6926` shortfall; mixed agents had `14.3513` against `20.3648` with `6.1871` shortfall.
- Diet-by-mode aggregates show animal modes still rely heavily on plant fallback. Hunters consumed `9.7305` animal energy across `55` animal events but `70.7610` plant energy; scavengers consumed `4.0722` animal energy across `15` carcass events but `148.3860` plant energy; mixed agents consumed `1.9696` animal energy across `15` animal events but `77.0325` plant energy. This makes animal-resource access and feeding opportunity the next concrete repair target.
- The meat-specialist matched-diet fix reduced false diet blockage for hunters and carnivores by discounting low-yield plant fallback in the specialist diet denominator. The animal-mode reproduction-energy multiplier improved animal-mode births, especially mixed-mode births, but it did not fix no-consumption seeds or terminal animal-mode readiness. Feeding opportunity and animal-resource access remain the next concrete tuning targets.
- `npm run sim:bench:quick`: pass. `summary_seed7_ticks20` reported `median_wall_seconds=0.5627`, `median_peak_rss_kib=91696`, and runtime cost counters.

Earlier ecology-gate failures in this audit were useful diagnostics and are now stale as pass/fail evidence. They identified the failure mechanism that this slice targeted: animal-resource specialists existed initially, but did not conserve enough energy, navigate to carrion/water reliably enough, or reproduce enough to survive the late window.

Earlier commands completed during this audit:

- `python3 -m compileall -q python/evolution_sim python/tests`: pass.
- `npm run sim:test`: pass, 55 tests in 173.660s.
- `npm run sim:gate:quick -- --fail-on-blockers`: pass.
- `npm run sim:bench:quick`: pass as a command, but exposed an RSS metric defect. On this macOS run, `summary_seed7_ticks20` reported `median_peak_rss_kib=91815936`; that value is bytes on macOS while the field says KiB.
- `npm run sim:gate:release -- --output output/evaluations/foundation-release-current.json`: stopped after more than 30 minutes with no progress output and no report file. Treat the current release profile as operationally unverified until it has progress reporting, timeout/checkpoint behavior, or enough performance work to complete predictably on a developer machine.

## Implementation Progress

These updates track work begun from this audit in the current working tree:

- 2026-04-27: Added a `WorldConfig.validate()` boundary, nested config validation, and a `SimulationWorld.__init__()` validation call so invalid dimensions, tick counts, season lengths, terrain ratios, spawn capacity, cross-field age/reproduction settings, and negative or out-of-range rates fail before terrain generation or runtime setup.
- 2026-04-27: Added a one-shot lifecycle guard to `SimulationWorld.run()` so reusing an episode object raises a clear error instead of appending frames, events, trajectory rows, counters, and derived state across runs.
- 2026-04-27: Added runtime contract tests for invalid ranges, invalid cross-field relationships, validation of mutated config objects before runtime setup, and one-shot run behavior.
- 2026-04-27: Removed the dead normalized-vector fast path from trophic profile caching; cache semantics are now intentionally raw-genome-keyed, with a focused regression that repeated agent/genome profile calls share the cached profile.
- 2026-04-27: Updated the benchmark runner so each scenario repetition executes in a fresh worker process and normalizes `ru_maxrss` to KiB on macOS/Linux before reporting `*_peak_rss_kib`.
- 2026-04-27: Tightened `_effective_tile_fields()` to be explicitly current-tick/current-climate only by removing the misleading `season` argument and updating callers.
- 2026-04-27: Added explicit `avg_historical_*_gene` and `avg_alive_*_gene` summary fields while preserving the existing `avg_*_gene` fields as historical compatibility aliases.
- 2026-04-27: Added Foundation gate progress callbacks, per-summary-seed and per-full-replay-probe timing, incremental JSON report writes, and per-scenario timeout/error blocker reporting. The npm gate entrypoint now emits progress to stderr and writes a usable partial report while long release runs are still in flight.
- 2026-04-27: Split biological reproduction readiness from population/spatial availability, added `agent_reproduction_blocked` events, and exposed max-agent saturation plus blocked-reproduction counts in summaries, frames, evaluation reports, and full-replay gate probes.
- 2026-04-27: Added versioned `mind_action_outcome_v1` trajectory outcomes, resolution action masks, resolution-valid flags, attack target/damage/success/kill/immediate-feed details, feeding source tile/source type/consumed/gained details, and invalid-resolution reasons while preserving legacy action helper return semantics for direct callers.
- 2026-04-27: Extended action outcomes with passive damage/death fields and explicit passive trajectory rows for agents killed before their turn. Rows now distinguish agents killed before action from agents that acted and then died later in the same tick.
- 2026-04-27 validation for this slice: targeted runtime contract tests, `python3 -m compileall -q python/evolution_sim python/tests`, `npm run sim:test` (65 tests in 171.597s), `npm run sim:golden:quick`, `npm run sim:gate:quick`, `npm run sim:bench:quick`, and `git diff --check` passed. After the benchmark RSS fix, `npm run sim:bench:quick` reports isolated KiB RSS (`median_peak_rss_kib=90992` on this macOS run) instead of macOS byte values.
- 2026-04-27 gate operability validation: `python3 -m compileall -q python/evolution_sim/cli python/tests/test_foundation_gate_cli.py python/tests/test_evaluate_cli.py`, `PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_foundation_gate_cli python.tests.test_evaluate_cli`, and `npm run sim:gate:quick -- --output output/evaluations/foundation-quick-current.json --scenario-timeout-seconds 600` passed. The quick gate wrote per-seed/probe timings and completed with status `pass`.
- 2026-04-27 reproduction analytics validation: `python3 -m compileall -q python/evolution_sim python/tests/test_runtime_contracts.py python/tests/test_evaluate_cli.py python/tests/test_foundation_gate_cli.py`, focused runtime/evaluation/gate unit tests, `npm run sim:test` (69 tests in 158.079s), `npm run sim:golden:quick`, `npm run sim:gate:quick`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=e3d260860709b3efd821524f45a0d195fa321c7bc651e93648fbad6d1360c1ef`, `seed7_ticks100=86890161b92fb9ab89db56861fa032260b0b0fba07c910a98a9371269ae28a96`, `speciation_seed_ticks320=38c6f781a11210691d91c8022936805833fba07a8e87600d770df662dd1cafce`, `seed5_ticks120_provenance=a9d475f9f8045fc54ba94f464d1dada3a6ea0ddce97a09220641d86074034b5f`.
- 2026-04-27 action-outcome schema validation: focused trajectory outcome tests, `npm run sim:test` (72 tests in 177.945s), `npm run sim:golden:quick`, `npm run sim:gate:quick`, and `npm run sim:bench:quick` passed. Updated golden hashes: `seed7_ticks20=f2c11efc4d7ed1b0c9ddce918fa3b503d60f5905de1921b29df7b1e04ef7fd94`, `seed7_ticks100=eb864b16867f6b95aaaced38f4805cc361a9ac56c1752e3e6c57394e5e76a6ce`, `speciation_seed_ticks320=534b04241db0f03dd0dfda63fa996017b6adb876cfd40948a02fcac7874a5300`, `seed5_ticks120_provenance=174ff3ae819d68a17a0bcae7a29c0abe77d1c1d02169a0f21615c77f72ecf3e7`.
- 2026-04-27 passive-outcome validation: focused passive/action-outcome tests, `npm run sim:test` (74 tests in 174.691s), `npm run sim:golden:quick`, `npm run sim:gate:quick`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=3b92a762455c6cb724070d1e0857c1cdc14ba1140e37f721ee73a63bf78bfe9c`, `seed7_ticks100=e7c006be71a4d2d28b2d599976671fe04f0dd179f7b952fa534167f5d66b080a`, `speciation_seed_ticks320=d1587a10ebb07d7e166fbc64cf77dcf6dbfc10bb08c1a3a3b56d0bbdc6e404d2`, `seed5_ticks120_provenance=71980a71a0ae7fcc13b3725fc40c37f5f2d342f003137bd4f6f3839d4faff65b`.
- 2026-04-27: Added `mind_observation_v2` with policy-excluded `metadata`, moved `agent_id` out of the policy-visible observation surface, and added a canonical `mind_observation_encoder_v1` encoder/decoder. The encoder declares shape, dtype, value range, enum vocabularies, patch order, quantization scale, and compressed int16 storage for compact replay payloads.
- 2026-04-27: Added encoded `observation_input` and `observation_metadata` to every trajectory record while preserving `observation_digest` for integrity checks. Full replay now contains trainable policy inputs instead of only hashes.
- 2026-04-27: Added a versioned reward contract with component and total bounds, clamped resource-acquisition reward to a stable scale, and added tests covering invalid-action, movement, resource, terminal, and reproduction reward components.
- 2026-04-27: Added a real sequential tick-conflict trajectory regression where an action is valid in the observation mask but invalid in the resolution mask after an earlier agent moves into the target tile.
- 2026-04-27: Extended Foundation gate Mind checks to require the canonical observation encoder metadata, validate encoded observation payloads, require policy-excluded observation metadata, and require versioned reward component bounds.
- 2026-04-27 observation/reward contract validation: focused observation, trajectory, reward, and gate tests passed; `npm run sim:test` passed (77 tests in 186.544s); `python3 -m evolution_sim.cli.golden_harness --update --all` updated all replay goldens; `npm run sim:golden:quick`, `npm run sim:gate:quick`, and `npm run sim:bench:quick` passed. Updated golden hashes: `seed7_ticks20=aa102f94b785ff4f733f63ea139c5cc8db3652fcd42a88a7d9434995c5e5791d`, `seed7_ticks100=b7eb4694158deeed6fbf883b6287fdbcb3f76e998f8470b9b5aea030bfcb7dfa`, `speciation_seed_ticks320=51df760de728ef0d1641511cca43b0ac32f2fef7f7757a9084d6ba61065e91ee`, `seed5_ticks120_provenance=2b28c3c96f06f29155392d4fd48d61c967585cc98e15490ceb14f817f1edde1e`.
- 2026-04-27: Decoupled trajectory capture from full replay by adding explicit `SimulationWorld.run(..., record_trajectory=True)` and `trajectory_sink` support. Summary-only trajectory capture now records trainable rows without events, frames, replay taxonomy, or viewer payloads.
- 2026-04-27: Added `JsonlTrajectoryWriter` and `npm run sim:trajectory` for gzip JSONL trajectory streaming. Streams include a header with the trajectory/observation/reward contract, one record line per decision, and a footer with run summary plus streaming trajectory statistics, without retaining all trajectory records in memory.
- 2026-04-27 trajectory streaming validation: focused runtime contract tests passed, `npm run sim:trajectory -- --seed 7 --ticks 4 --output /tmp/evolution-sim-trajectory-test.jsonl.gz` wrote 80 records, `npm run sim:test` passed (79 tests in 188.892s), `npm run sim:golden:quick`, `npm run sim:gate:quick`, and `npm run sim:bench:quick` passed.
- 2026-04-27: Added a production policy boundary (`Policy.decide(observation, action_mask) -> ActionDecision`) and moved the default heuristic behind that boundary. The default policy now consumes only frozen `mind_observation_v2` data plus the action mask; a runtime contract regression prevents it from calling the legacy live-world heuristic.
- 2026-04-27: Expanded observation v2 with bounded navigation targets for water, plant, carrion, and prey so the observation-only policy has long-range cues without receiving `SimulationWorld`.
- 2026-04-27: Added policy id/version metadata to trajectory records and Foundation gate checks. Trajectory summaries now include `policy_interface_version`, and first-record gate checks require `policy_id` and `policy_version`.
- 2026-04-27: Added runtime-cost counters for observation/action-mask builds, biotic state builds/cache hits/invalidations, and biotic diffusion target cache hits/misses. Bench reports now include median runtime-cost counters and a streaming trajectory scenario.
- 2026-04-27: Made replay/runtime/ecotype centroid units explicit by emitting raw gene centroids separately from normalized centroids, with identity-mode metadata for runtime lineage, frame-local ecotype, and replay taxonomy catalogs.
- 2026-04-27: Repaired the quick-gate animal-resource collapse by separating focused specialist classification from breadth-diet consumption drive, then making omnivore hunter/scavenger policy conserve and forage locally before long prey pursuit. Quick gate terminal populations now retain both hunter and scavenger modes on seeds `7` and `8`.
- 2026-04-27 policy/ecology validation: focused runtime/bench/gate/headless ecology tests passed (56 tests in 51.900s), `npm run sim:test` passed (82 tests in 103.350s), all replay goldens were refreshed, `npm run sim:golden:quick`, `npm run sim:gate:quick`, `npm run sim:bench:quick`, and the `trajectory_stream_seed7_ticks100` bench scenario passed. Updated golden hashes: `seed7_ticks20=928ee586bfa7d5103e93b1194cc40c29bde720babd96cf245e2bccc4f3ab6a91`, `seed7_ticks100=d91e14f0cb77f8a6cf5b3d3c61b9aa9bd76dab15cf164febd008f5eb23c8b11d`, `speciation_seed_ticks320=15db662ee00699b114990f6cb816ef2c02abccf3b7be68fc230d175b193bfa5b`, `seed5_ticks120_provenance=3a7b66398edb0f0281e723848f8aa86730cfd2d019c722afab1db40ec0002e5a`.
- 2026-04-27: Added a summary-only `ecology` Foundation gate profile and `npm run sim:gate:ecology` as the post-repair seed-bank probe. It runs seeds `1-20` for `120` ticks, emits per-seed progress, writes incremental output to `output/evaluations/foundation-ecology-current.json`, and requires terminal trophic and animal-resource mode diversity without running heavy full-replay probes.
- 2026-04-28: Added compact trophic lifecycle diagnostics to summary-only runs and Foundation gate reports: initial role/mode counts, births by parent and child role/mode, deaths by role/mode, initial-cohort survival, late-window checkpoint presence, and an ecology failure rollup. These fields make the current seed-bank failure concrete: animal-resource specialists are present initially, but they are not reproducing enough to survive the late window.
- 2026-04-28: Repaired the 120-tick ecology seed bank by teaching the observation-only heuristic to prioritize urgent hydration, carrion, and close prey before long-distance hunts; expanding bounded navigation cues; modestly increasing meat/carcass nutrition; adding a low-yield carnivore plant fallback; and reducing low-energy stationary drain for meat-mode specialists.
- 2026-04-28: Tightened the ecology gate around the repaired behavior. The profile now uses a late-window per-run animal-mode floor, requires at least one late-window animal-resource mode per seed, and requires at least two animal-resource modes across the aggregate seed bank.
- 2026-04-28 ecology repair validation: focused runtime/gate/headless tests passed, `npm run sim:test` passed (89 tests in 111.206s), all replay goldens were refreshed and verified with `npm run sim:golden`, `npm run sim:gate:quick -- --fail-on-blockers` passed, `npm run sim:gate:ecology -- --fail-on-blockers` passed, and `npm run sim:bench:quick` passed. Updated golden hashes: `seed7_ticks20=7623ad8511a990c57204ee1cdc93a850acb2934d4f5a318ec9c75a744c64fa53`, `seed7_ticks100=99c627a68302807596193d04066b38d81c3cd2ce1caf945b79df3c7056ca3f2f`, `speciation_seed_ticks320=5ada5daa259f5b11c5ef1d621b5c172ae37feaedbb09304f09fa17846274a335`, `seed5_ticks120_provenance=19071428df836699b1e6df0504ed99af5af2933b4c71a3cc541c52a944e35186`.
- 2026-04-28: Added reproduction biological-blocker diagnostics by trophic role and meat mode. Evaluation reports and Foundation gate ecology rollups now expose terminal biological blocker counts and run-level blocked-reproduction counts split by role/mode, making animal-mode reproduction failure visible without reading per-seed logs.
- 2026-04-28: Adjusted meat-specialist diet matching so low-yield plant fallback keeps hunter/scavenger agents alive without drowning out sparse animal-resource meals in the reproduction matched-diet denominator. Plant-only fallback still produces no specialist diet match.
- 2026-04-28: Added reproduction energy-readiness diagnostics by trophic role and meat mode. Summaries, evaluation aggregates, and gate rollups now report alive agents, energy-shortfall agents, total energy, required reproduction energy, and total energy gap per role/mode.
- 2026-04-28 reproduction diagnostic validation: `python3 -m compileall -q python/evolution_sim python/tests`, focused runtime/evaluate/gate tests (17 tests in 3.382s), `npm run sim:test` (91 tests in 105.260s), `npm run sim:golden`, `npm run sim:gate:quick -- --fail-on-blockers`, `npm run sim:gate:ecology -- --fail-on-blockers`, and `npm run sim:bench:quick` passed. Updated golden hashes: `seed7_ticks20=a745840721008801cbb8b04a8a20f1d3c006809cc5159b3d49162ced36aaba31`, `seed7_ticks100=23b06a03d497ab9df4c3c405e7c2a1ddfbeb4317adbe28fb637a3937b8c03a56`, `speciation_seed_ticks320=23515db9f996c21d4fb920a0dec7f9751fbd224eebc83fa9e1700c43520d3838`, `seed5_ticks120_provenance=2e8c7dc00a0b3bab7ed5306bf3cde46fbb3f9efd93fb6a920500c939f04fffe7`.
- 2026-04-28: Added `reproduction.animal_mode_energy_requirement_multiplier` with validation and applied it to non-`none` meat modes in the reproduction energy requirement helper. The default `0.82` multiplier is conservative: it improves animal-mode parent births in the 20-seed ecology bank without treating it as a complete fix for animal-resource access.
- 2026-04-28 animal-mode reproduction-energy validation: `python3 -m compileall -q python/evolution_sim python/tests`, focused runtime/headless/evaluate/gate tests (20 tests in 3.406s), `npm run sim:test` (92 tests in 106.029s), `npm run sim:golden`, `npm run sim:gate:quick -- --fail-on-blockers`, `npm run sim:gate:ecology -- --fail-on-blockers`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=5df2734be42365becc402ec3a76962c469442875517b1100fc0a9a4d8a0189db`, `seed7_ticks100=7d8bca815a53fe8ce123a5886b8f9e6ee8715258f87b887b480b979be33de7c5`, `speciation_seed_ticks320=d86316f5640e539bf17e663f5f383953fcf238c09c77ade3d12b31816fcb3b86`, `seed5_ticks120_provenance=cf8d98383040d403e32d8cb5ddbdca504759f115719ae85d69dfd0fe8c775d2d`.
- 2026-04-28: Added diet-by-trophic-role and diet-by-meat-mode aggregation to evaluation reports and Foundation ecology rollups. The aggregate intentionally totals event and energy fields only, not ratio/share fields, so cross-run reporting does not sum meaningless percentages.
- 2026-04-28 diet aggregate validation: focused evaluate/gate tests (16 tests in 3.086s), `npm run sim:test` (92 tests in 110.815s), `npm run sim:gate:ecology -- --fail-on-blockers`, and `git diff --check` passed. Goldens were not refreshed for this reporting-only slice because simulator summaries and replay payloads did not change.
- 2026-04-28: Added animal-resource opportunity diagnostics by meat mode to shared summaries, evaluation aggregates, and Foundation ecology rollups. The counters split alive ticks/agent-ticks, resource-present ticks, resource-absent ticks, present-but-unconsumed ticks, and source-specific fresh-kill/carcass consumption. Gate rollups now list no-consumption seeds by animal mode and separate absent-resource seeds from present-unconsumed seeds.
- 2026-04-28 animal-resource opportunity validation: compileall, focused headless/evaluate/gate tests, `npm run sim:test` (95 tests in 117.612s), refreshed and verified all replay goldens, `npm run sim:gate:quick -- --fail-on-blockers`, `npm run sim:gate:ecology -- --fail-on-blockers`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=e0d580bcc3c289f2107af4ec0d508930f5e2911c843e680b4040ee8cff73214f`, `seed7_ticks100=acd0a2ce722f421060b27a3ec032db7f8b87ea934bd295a0ccd471f5a0894265`, `speciation_seed_ticks320=090155f3d7ed75c1d5b1e7ece0f3a72acf5cfb5dbf02a96be10fa9be6ac916fa`, `seed5_ticks120_provenance=08a901ee7cb2806ca74c77a31888e74b1c5cf210bb855088b31aeee59e110079`. The ecology rollup now shows no animal-resource-absent runs for hunter, mixed, or scavenger; the remaining no-consumption cases are present-unconsumed resources.
- 2026-04-28: Repaired scavenger carrion acquisition by making carrion navigation prefer actual carcass/fresh-kill resources over diffuse scent, letting declared scavenger modes consume carcasses even when their absolute drive is below the generic animal-use threshold, and making scavenger eat resolution prefer useful carcass intake over same-tile plant fallback. The observation-only heuristic now seeks nearby carrion before omnivore plant fallback, but avoids moderate-energy long-distance carrion chases that collapse late-window animal-mode presence.
- 2026-04-28 scavenger acquisition validation: compileall, focused scavenger policy/headless tests, `npm run sim:test` (100 tests in 110.883s), refreshed and verified all replay goldens, `npm run sim:gate:quick -- --fail-on-blockers`, `npm run sim:gate:ecology -- --fail-on-blockers`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=61137e6452704c2b2cadf6f7007b4d1b17df119f5f44445f3fa01fd958692502`, `seed7_ticks100=d46349313933fb56c75bef6047cfedd079899ea0f9e2f89b9bfc9b084ee92e97`, `speciation_seed_ticks320=98170f5ba8eda438d73bcaea1b3318873ef7148b416e9aba43e4abed0ca04b4e`, `seed5_ticks120_provenance=1243bdfa0ae841692b780efc131418e8c864251586a751eeb0e2c6be38c494be`. Known seeds `10`, `15`, and `20` now have scavenger carcass consumption; seed `4` remains present-but-unconsumed.
- 2026-04-28: Repaired part of hunter prey/fresh-kill acquisition by making hunter-mode omnivores attack adjacent prey before plant fallback, making prey navigation prefer actual vulnerable prey over diffuse predator/animal biomass, and adding configurable hunter-mode attack damage modifiers for wounded prey. A follow-up detour rule lets desperate scavengers step around a blocked direct carrion vector instead of conserving in place when the first step toward visible carrion is water-blocked.
- 2026-04-28 hunter/scavenger acquisition validation: compileall, focused hunter/scavenger policy and headless tests, `npm run sim:test` (104 tests in 124.034s), refreshed and verified all replay goldens, `npm run sim:gate:quick -- --fail-on-blockers`, `npm run sim:gate:ecology -- --fail-on-blockers`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=3d00be3ef8cd3d91d0bf0c784c59d8677998c0966e2106d9e061082ad903e10f`, `seed7_ticks100=cd03d3b0a8447310ccb75effd592043e9bbfd6f8060cdbea61369c139d1b6bde`, `speciation_seed_ticks320=59aef4721f57e1cb58c42a60d4a5ed9a47930a3df69ce463a723aae2be76cd75`, `seed5_ticks120_provenance=f691e6a6f9bee4e88ca6ac832417a0b2b6276758fab1249de0a0ff8c7497dc0d`. In the 20-seed ecology rollup, hunter no-consumption improved to `2/20`, scavenger remained at `3/20`, mixed improved to `7/13`, and animal-resource-absent runs remained `0` for all animal modes.
- 2026-04-28: Repaired same-tile hunter/mixed animal-resource consumption so low-yield fresh-kill/carcass intake beats plant fallback when the animal resource is already under the agent, allowed hunters to opportunistically consume carcasses, and fixed carrion navigation so actual resources under another occupant are still visible to the observation-only policy.
- 2026-04-28: Added a critical local-food escape hatch for meat modes so starving agents can eat adequate same-tile food instead of chasing distant carrion into late-window collapse. The heuristic policy version is now `observation_heuristic_v4`.
- 2026-04-28: Promoted animal-resource consumption floors into Foundation gate profiles. Ecology now requires each live animal mode to consume animal resources in at least `50%` of its alive runs; release is configured at `75%`.
- 2026-04-28 animal-resource floor validation: compileall, focused hunter/scavenger/carrion-navigation/gate tests, `npm run sim:test` (113 tests in 110.127s), refreshed and verified all replay goldens, `npm run sim:gate:quick -- --fail-on-blockers`, `npm run sim:gate:ecology -- --fail-on-blockers`, `npm run sim:bench:quick`, and `git diff --check` passed. Updated golden hashes: `seed7_ticks20=2c52e3790af852968fa13a1fbc12bc3d309f89f251b19d192bd603fc6229a259`, `seed7_ticks100=663f5936b673e722c36648f623293d29840ec923a429ea1bb408d6ba587e1da6`, `speciation_seed_ticks320=78f038c27cefc7c9ed15820e8ae66a1c76cee2c7eaaca145cb78c4b3669f914e`, `seed5_ticks120_provenance=a46af00fcb3301c8069caf609d8100af41c9ba47492cab79164dfd62ee94e881`. The 20-seed ecology rollup now clears run-level no-consumption seeds entirely; remaining by-mode no-consumption cases are hunter `1/20`, mixed `4/13`, and scavenger `7/20`, with animal modes still at `0` terminal biologically ready agents.
- 2026-05-01: Added the reproductive runtime cleanup as an explicit pre-signal plan slice, then extracted reproductive eligibility, readiness reporting, child genome stabilization, asexual/sexual birth orchestration, cost/cooldown accounting, and reproduction event emission from `world.py` into the runtime reproduction boundary. `SimulationWorld` keeps thin compatibility wrappers for existing tests and local callers, while the new runtime helper is directly covered by the Stage 1 sexual reproduction regression.
- 2026-05-01 reproductive runtime cleanup validation: `python3 -m compileall -q python/evolution_sim python/tests`, focused reproductive regressions, `npm run sim:test` (165 tests in 105.878s), `npm run sim:golden:quick`, `npm run sim:golden`, `npm run sim:gate:quick`, `npm run sim:bench:quick`, and `git diff --check` passed. Full golden verification matched `seed7_ticks20=c4c4d59c05b563d49c86003180c98ca34fff494784ad049d549c95d4f80398c9`, `seed7_ticks100=b2413ee89de8f905b477113c91615318dcaf639bf6c95460d2513a2a4b9f7c86`, `speciation_seed_ticks320=a01483ec7e79a00b67b50677559e21c9dd3bee19816df546a1d0f739fe7875e3`, and `seed5_ticks120_provenance=d1b0373e3b49dca66021612e88e13dbf4903d9a6d72e9f16ed56907c966b27ac`.
- 2026-05-01: Implemented the first live signal substrate slice on the default Foundation path. Biologically ready agents now emit deterministic reproductive-readiness fields with configured intensity, radius, duration, decay, and energy-cost hooks; observation self/local patch fields, replay frames, summary metrics, analytics, and the viewer expose the numeric field. Communication signal tokens remain reserved and opaque, with no scripted movement or unmasked policy action.
- 2026-05-01 reproductive-readiness signal validation: `python3 -m compileall -q python/evolution_sim python/tests`, focused signal/replay regressions, `npm run sim:test` (167 tests in 104.767s), refreshed and verified all replay goldens, `npm run sim:gate:quick`, `npm run sim:bench:quick`, seed-7 replay generation plus `viewer:smoke`, and `npm run sim:gate:release -- --output output/evaluations/foundation-release-signal.json` passed. Refreshed golden hashes: `seed7_ticks20=ea28840ec2aa92eeae95f2acba942c6896d9a9fe77f80703ad7cf3e2100e7df6`, `seed7_ticks100=3cc29658bb4baf3060904802a1bf9bfe28016388974a1080ee84bc49a7cb38f8`, `speciation_seed_ticks320=2d451c51a046bdc8a78a9228e3d2fea9c9f4c43762bbaaa01ce6c128163804ef`, and `seed5_ticks120_provenance=e0e2abce6e0d9f259d11b1918f5b9f4872e0b3911d1236e2859fd505a91bfa73`. The release gate had no blockers or warnings; wall time was 857.2421s, with dense summary seeds at 185.1913s, 169.3533s, and 238.0486s. A 200-tick seed-3 enabled/disabled timing probe showed signal fields were not the driver at that horizon (`11.5285s` enabled vs `11.5381s` disabled), so release-scale performance remains a separate watched item before adding communication emissions.
- 2026-05-01: Hardened the signal contract to `foundation_signal_contract_v2`. Full replay frames now include debug-only per-tick signal emission provenance with source agent ID, profile ID, token/profile index, source position, intensity, radius, duration, decay, energy cost, and emitted tick. The Foundation gate checks that the signal contract and frame metadata remain versioned and policy-hidden; communication profiles remain reserved and inert.
- 2026-05-01 signal contract hardening validation: `python3 -m compileall -q python/evolution_sim python/tests`, focused signal/gate/replay tests, `npm run sim:test` (170 tests in 106.642s), refreshed and verified all replay goldens, `npm run sim:gate:quick`, `npm run sim:bench:quick`, seed-7 replay generation plus `viewer:smoke`, and `git diff --check` passed. Refreshed golden hashes: `seed7_ticks20=d558c81a2b7a0a73dd2cea8ab75bf7127bd81e73511dc80e6f5c7180995993fb`, `seed7_ticks100=0f7d508917d95f00f5bc735f1e01761719c27e3938e9f35eab9d54ca037afd43`, `speciation_seed_ticks320=aebce1b7510795e2c7b9713fbe7bfac41cc9986a5620580753c463c4c79b62c6`, and `seed5_ticks120_provenance=efb76ec62b0ff59fddbb7bb19d16b9272fac6833146d73da945eb637293eef84`.
- 2026-05-01: Added a shared browser/Node replay validator for the viewer. The validator now rejects malformed full replay payloads before rendering when map grids are ragged, frame ticks are nonmonotonic, agent rows do not match `agent_encoding`, signal emission metadata is missing or stale, required frame matrices are absent, or trajectory records are missing required Mind fields.
- 2026-05-01 viewer validator validation: `npm run viewer:validate`, browser-level `npm run viewer:smoke:malformed`, seed-7 replay load through `viewer:smoke`, and `git diff --check` passed. The Node validator smoke covers seven malformed replay cases, and the browser malformed smoke covers the real failed-load UI path for empty frames, missing signal emission metadata, ragged terrain, and mismatched action/signal communication capacity.
- 2026-05-01: Threaded `SignalConfig` through observation and trajectory contract generation so full replay and streaming trajectory headers declare the actual configured communication token count, profiles per token, signal radius/duration limits, and communication decay rate instead of static defaults.
- 2026-05-01 config-aware signal contract validation: focused runtime contract regressions, `python3 -m compileall -q python/evolution_sim python/tests`, `npm run sim:test` (171 tests in 110.765s), refreshed and verified all replay goldens, `npm run sim:gate:quick`, `npm run sim:bench:quick`, `npm run viewer:validate`, seed-7 replay generation plus `viewer:smoke`, browser-level `npm run viewer:smoke:malformed`, and `git diff --check` passed. Refreshed golden hashes: `seed7_ticks20=cf42554c951735f25c551c204c3a8647e1ff296cd10ec529420bad34cc0b687f`, `seed7_ticks100=137f6fc2bcaf7d193e40069e0f0ab65ddef85d8292f9e07babad884f5f06fbe3`, `speciation_seed_ticks320=c145b5c2378ebb6658aaaa475de4df1e128c5d6b36c7ea487a7a4a89e12ad276`, and `seed5_ticks120_provenance=42365f9e76a2b5ced4907f236a10534f148bc870d1e150c7a98c8bd56f239454`.
- 2026-05-01: Closed the follow-on communication-capacity mismatch: `ActionContract`, observation action names, and runtime action masks now derive reserved signal action slots from the active `SignalConfig`, matching the signal profile catalog used by replay and trajectory metadata. Focused runtime contract checks for the default reservation path and a custom `2 x 3` communication profile configuration passed.
- 2026-05-01: Added opt-in opaque communication emission plumbing behind `SignalConfig.communication_signal_emission_enabled`. Communication actions remain masked by default and require both config enablement and agent signal-emission trait bias; when enabled, they emit numeric communication fields with token/profile IDs, deterministic radius/duration/decay/cost, and debug-only replay provenance. The default heuristic still has no communication-emission behavior.
- 2026-05-01 opt-in communication validation: `python3 -m compileall -q python/evolution_sim python/tests`, focused runtime/gate communication regressions, `npm run sim:test` (175 tests in 109.898s), refreshed and verified all replay goldens, `npm run sim:gate:quick`, `npm run sim:bench:quick`, `npm run viewer:validate`, browser-level malformed replay smoke, seed-7 replay generation plus `viewer:smoke`, and `git diff --check` passed. Refreshed golden hashes: `seed7_ticks20=32765e5d9c62c3b7508789dce1d11ca1b07f9e73476f8aa23ad7d8055bc793f4`, `seed7_ticks100=17f22b40f56215de7d6e81743caab46a3c2855ed66043b0a51cad44c2e081f5f`, `speciation_seed_ticks320=43981369f28b347ef25aeeda55ad3ac359f00d5f34d39c6781b3193c23cdf582`, and `seed5_ticks120_provenance=2ff325d27e2250267f8b058d2ea53415d8f68c3697e65db4fd0e28b30ac0c23e`.
- 2026-05-01: Tightened the reproductive runtime boundary after the communication slice. `SimulationWorld._run_tick()` now delegates the tick-level reproductive readiness signal pass, biological/blocking checks, and birth loop to `runtime_reproduction.run_reproduction_phase()`, keeping the same signal-before-birth ordering while reducing direct orchestration in `world.py`.
- 2026-05-01 reproductive phase boundary validation: focused phase and Stage 1 reproductive regressions, compileall for the touched modules, `npm run sim:test` (176 tests in 104.016s), `npm run sim:golden:quick`, `npm run sim:golden`, `npm run sim:gate:quick`, `npm run sim:bench:quick`, and `git diff --check` passed. Golden hashes remained `seed7_ticks20=32765e5d9c62c3b7508789dce1d11ca1b07f9e73476f8aa23ad7d8055bc793f4`, `seed7_ticks100=17f22b40f56215de7d6e81743caab46a3c2855ed66043b0a51cad44c2e081f5f`, `speciation_seed_ticks320=43981369f28b347ef25aeeda55ad3ac359f00d5f34d39c6781b3193c23cdf582`, and `seed5_ticks120_provenance=2ff325d27e2250267f8b058d2ea53415d8f68c3697e65db4fd0e28b30ac0c23e`.
- 2026-05-01: Fixed viewer smoke reproducibility. The browser smoke scripts now start a local viewer server on port `4173` when one is not already serving the viewer, reuse an existing server when present, and stop only the server they started. This makes `npm run viewer:smoke` and `npm run viewer:smoke:malformed` valid standalone entrypoints.
- 2026-05-01 viewer smoke reproducibility validation: `npm run viewer:smoke:malformed`, `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`, `npm run viewer:validate`, port `4173` listener checks, and `git diff --check` passed.
- 2026-05-01: Added the first proto-role/X/Y/Z reproductive-stage scaffold. `ReproductionConfig` now has explicit thresholds for role differentiation, X/Y/Z expression, Z plasticity, role-complementarity bonus, and Z/Z pairing penalty. Reproductive genomes classify into Stage 2 proto-role or Stage 3 X/Y/Z expression at high thresholds, same fixed-role mating is blocked, fixed/plastic and complementary pairings are allowed, and reproductive group records promote to the highest observed member stage.
- 2026-05-01 proto-role/X/Y/Z scaffold validation: compileall for touched modules, focused runtime contract regressions for config validation/stage classification/compatibility/group-stage promotion, `npm run sim:test` (180 tests in 107.314s), refreshed and verified all replay goldens, `npm run sim:gate:quick`, `npm run sim:bench:quick`, `npm run viewer:validate`, seed-7 replay generation plus `viewer:smoke`, and `git diff --check` passed. Refreshed golden hashes: `seed7_ticks20=6cda5a03ac4ad4e5b663f1b10bcc58f3b5cb7fa4bb471f85faab7f7ded3755eb`, `seed7_ticks100=f1c40206bd681fc038c91157d484c6889f3d06370c967506243cefd999671a25`, `speciation_seed_ticks320=c565b18a20cf8dc0691799df9265ef1d19cbd2e4242bbd86ed10209c3940035e`, and `seed5_ticks120_provenance=8f4b11b41725ec7ba2c121557fb2c98d4622db80cd3f588bcb24edaded9b289f`.
- 2026-05-01: Versioned reproductive birth event payloads as `reproduction_event_v1`. Asexual and same-group sexual births now include parent lineage IDs, parent reproductive group IDs, child lineage/group/stage/expression, parent energy costs, compatibility/inbreeding placeholders or values, mate distance, offspring count, hybrid flag, and inert Mind inheritance metadata while preserving legacy keys such as `child_id`, `lineage_id`, `reproduction_mode`, and `parent_ids`.
- 2026-05-01 reproductive event schema validation: compileall for `runtime/reproduction.py`, focused Stage 1 sexual/asexual fallback event regressions, `npm run sim:test` (180 tests in 108.934s), refreshed and verified all replay goldens, `npm run sim:gate:quick`, `npm run sim:bench:quick`, `npm run viewer:validate`, seed-7 replay generation plus `viewer:smoke`, and `git diff --check` passed. Refreshed golden hashes: `seed7_ticks20=6cda5a03ac4ad4e5b663f1b10bcc58f3b5cb7fa4bb471f85faab7f7ded3755eb`, `seed7_ticks100=3ee0a8faade29c29a1efb0bdd655e325dc43d5852c7581eccb0da70791c4601e`, `speciation_seed_ticks320=df595f8d0d1c8133ecf8d2d5c1b757d4bc7ff1274e27c4bbd4419910487c4f64`, and `seed5_ticks120_provenance=4621c37dd702ee2d67de83012ae224b26ab9254d00da55d7aafcf8fb5f7f5e52`.
- 2026-05-01: Hardened reproductive observability before adding stronger role behavior. Mate search now produces a diagnostic report for sex-capable parents, including selected partner, group scarcity, partner sexual lock, distance, partner readiness, expression incompatibility, and sex-capable asexual fallback reasons. Reproductive summary/frame stats now expose stage counts, expression counts, capability counts, ready group counts, mate-search run/tick counters, and per-group alive stage/expression composition.
- 2026-05-01 reproductive observability validation: compileall for touched runtime/tests, focused mate-search/stage/group regressions, `npm run sim:test` (182 tests in 111.409s), refreshed and verified all replay goldens, `npm run sim:gate:quick` (pass, no blockers or warnings), `npm run sim:bench:quick` (`summary_seed7_ticks20` median wall `0.6125s`, median RSS `92704 KiB`), `npm run viewer:validate`, seed-7 300-tick replay generation plus `viewer:smoke`, `git diff --check`, and port `4173` listener check passed. Refreshed golden hashes: `seed7_ticks20=218839d702a5875550eea19c5ca7df18ea99909b0412f869ffa98c890e3ac6ad`, `seed7_ticks100=e60c38ba3a75e29e9acdc4c14737476e9cd15ea479e7cfc8d909dfd124d5b3ac`, `speciation_seed_ticks320=9de17eea66ccfb37d330ec88b1e343383010c402f54c500071fc64377f887dfc`, and `seed5_ticks120_provenance=c705579c29ab70f0b4e5a5d2ab24680ee829dcb646127ad030397b204faa141f`.
- 2026-05-01: Re-hardened mate-search observability to keep diagnostics lossless. The runtime now preserves primary fallback counts for stable summaries, while also recording all overlapping candidate constraints and partner-pool counts: scanned agents, same-group candidates, same-group sex-capable candidates, in-radius candidates, biologically ready candidates, and per-constraint blockers. Tick-detail event rows are skipped when tick details are disabled, preserving lightweight summary-only behavior.
- 2026-05-01 mate-search diagnostic hardening validation: compileall for touched runtime/tests, focused overlapping-constraint and lightweight tick-detail regressions, `npm run sim:test` (184 tests in 110.973s), refreshed/verified full replay goldens, `npm run sim:gate:quick` (pass, no blockers or warnings), `npm run sim:bench:quick` (`summary_seed7_ticks20` median wall `0.6107s`, median RSS `92768 KiB`), `npm run viewer:validate`, and seed-7 300-tick replay generation plus `viewer:smoke` passed. Refreshed golden hashes: `seed7_ticks20=9bdda95d1425f4203c64c14f120cc2ba957222635c5f748fd0f434e0263f7beb`, `seed7_ticks100=879bc83beab696c521b8142c5adfc15f5222cce05d320c8c05b9c53593431895`, `speciation_seed_ticks320=2121506a986f71147a876b03ca7a595e859862ec84e94c6a33225afe3c28df7b`, and `seed5_ticks120_provenance=7609e94cb801a77b19f3e16f5cbee26594085c8ba16f6b5201447d4c0e95d7d3`.

Earlier targeted probes from the deeper Foundation pass, retained as audit evidence. Several are now addressed by the implementation progress above, while release-horizon ecology, animal-specialist reproductive strength, and release-scale performance remain open:

- `WorldConfig(climate=ClimateConfig(season_length=0))` crashes with `ZeroDivisionError` in `_season_state()`, so config validity is not enforced at the boundary.
- `WorldConfig(water_tile_ratio=-0.1)` initializes successfully but produces `1383` water tiles out of `1536`, because negative target counts flow into Python slicing. `WorldConfig(water_tile_ratio=1.2)` fails later as an impossible spawn world instead of failing config validation.
- `WorldConfig(max_ticks=0)` runs zero ticks but reports `ticks_executed=1`, because summaries use `self.tick + 1` even when the loop never executed.
- Calling `_effective_tile_fields(x, y, "wet")` and `_effective_tile_fields(x, y, "dry")` at tick `90` returned identical values while the active season was dry. The explicit `season` argument does not compute an alternate season; current-tick climate dominates.
- A seed-1 lethal attack at tick `4` emitted `agent_damaged`, `agent_died`, immediate `agent_ate`, then `agent_attacked`. The attacker's trajectory row only recorded `resource_gain`, `reproduced`, `died`, and `reproduction_ready_after`; it did not record target id, damage, kill, or immediate kill-feed as action outcome fields.
- Reusing the same `SimulationWorld` instance for two full replay runs appends replay surfaces instead of resetting them: a `2`-tick world produced `2` frames on the first run and `4` accumulated frames on the second. The world is effectively one-shot but does not enforce that contract.
- `genome_profile_key(agent.genome) == genome_vector(agent.genome)` is `False` for normal agents, confirming the trophic cache fast path is dead.
- Pre-repair default summary-only runs at 120 ticks for seeds `1,2,3,7,11` all ended herbivore-only with all meat modes `none`, so animal-resource collapse was visible well before the old 800-tick release artifact.
- A 20-tick seed-7 summary run rebuilt biotic state 170 times. That is not automatically a bug, but it makes per-tick derived-state cost a Foundation metric before Mind rollout scale increases.
- A pre-repair broader default 120-tick sweep over seeds `1-20` ended `18/20` runs herbivore-only with no terminal animal-resource mode. The maximum terminal animal-energy share across the 20 runs was `0.0123`; seed `11` reached `0.0`.
- Sampled default 800-tick summary-only release seeds `3`, `7`, and `11` each exceeded a `45s` per-run timeout. Sampled compact 1000-tick worlds at `32x24`, `16` initial agents, and `160` max agents exceeded a `35s` timeout for seeds `1-5`.
- Parameter-combination probes showed additional contract gaps: `initial_agents > max_agents` is accepted and starts above the cap, `max_age < reproduction.min_age` causes extinction before reproduction, very large reproduction cost permits births that immediately push parents toward death, negative hazard and attack-cost rates are accepted, and a huge biotic diffusion radius caused a 20-tick run to take `3.7446s`.
- Interaction stress probes combining hydrology, hazards, reproduction, crowding, and predation still tended toward terminal herbivore dominance. A crowded high-hazard 80-tick run generated `232` attacks and `68` kills but ended `97` herbivores, `1` carnivore, and no hunters. A cheap-attack crowded run generated `313` attacks and `92` kills but still ended `90` herbivores out of `94` alive.
- A full replay schema probe found no actual frame-shape mismatch in a seed-7 40-tick replay, but found `20` species centroids whose `max_energy` value was outside raw gene limits because replay taxonomy stores normalized centroid values under raw gene keys. Before the action/observation contract slices, the same trajectory probe confirmed records contained no observation payload and action outcomes only contained `resource_gain`, `reproduced`, `died`, and `reproduction_ready_after`.

## Current Verdict

Foundation is materially closer but still not closed for Mind v1. The current tree now has an enforceable observation-only default policy boundary, trainable trajectory payloads, versioned rewards/action outcomes, low-memory trajectory streaming, explicit runtime-cost counters, explicit centroid units, quick plus 120-tick ecology gates, and a passing release gate.

The strongest remaining negative evidence is now semantic instability, not gate operability: the desired world still includes deeper role-based sex-expression behavior, optional mate-attempt semantics, emergent/default communication use, hybridization, and later learned-state inheritance. Those semantics would change state, actions, observations, rewards, replay, metrics, gates, and viewer surfaces if added after Mind v1. They must be Foundation work.

Sexed reproduction status: Stage 1 is implemented as a rare facultative live path. The current default still begins with automatic single-parent asexual reproduction, but reproductive traits now mutate and can unlock `stage1_facultative_sex`. When a local same-group partner is also biologically ready, both parents pay cost/cooldown and the child is created through grouped recombination plus mutation, with dual-parent replay metadata and inbreeding viability pressure. If no valid partner exists, the old asexual path remains the fallback and the fallback reason is now counted. The first reproductive-readiness pheromone field is live and replay-auditable, communication emission is available only through an explicit opt-in config and trait gate, and high-threshold proto-role/X/Y/Z expression is scaffolded with compatibility diagnostics. Full Z imbalance behavior, hybridization, multi-offspring strategy, emergent/default communication policy, and policy-visible mate attempt outcome contracts remain future Foundation work.

So the order is now: preserve the validated release gate as the regression boundary, implement the reproductive/signal plan in staged Foundation slices, keep pre-Mind observability aligned with that stable environment, then start Mind. The observation/trajectory scaffolding is real enough to build on, but reproduction and signal semantics still need a deeper pass before learned-controller work starts.

Mind v1 should not start until these are true:

- Long-horizon release gates continue to preserve terminal trophic and animal-resource diversity after reproductive/signal changes.
- Reward components remain explicitly defined as an experiment contract, with tests for scale, terminal events, invalid actions, and reproduction.
- The release gate measures the full Mind data contract and the Foundation ecology contract directly.
- Benchmark memory metrics are trustworthy per scenario and cross-platform.
- Sexed reproduction, reproductive groups, pheromones, communication slots, and any future mate action are already represented in state, observation, action masks, rewards, replay, gates, and baseline policies without exposing human-readable semantics to Mind.

## Critical Findings

### 1. The heuristic policy boundary was not enforced

Files:

- `python/evolution_sim/env/runtime/policy.py`
- `python/evolution_sim/env/runtime/observations.py`
- `python/evolution_sim/env/world.py`

Before the policy-boundary slice, the active heuristic took `(world, agent)` and read live world state through private helpers. That made the observation contract descriptive rather than authoritative.

Status: addressed for the default policy path. `SimulationWorld` now accepts a `Policy`, the default `ObservationHeuristicPolicy` consumes only the frozen observation plus action mask, and trajectory rows record the policy id/version used for the decision. A runtime contract regression prevents the default policy from calling the legacy live-world action chooser. Observation v2 now also includes bounded navigation cues so the policy does not need private world reads for longer-range water, plant, carrion, or prey signals.

Remaining work: keep future policies behind the same interface, add richer policy diagnostics only as explicit trajectory fields, and avoid reintroducing direct `SimulationWorld` reads in learned-controller adapters.

### 2. Trajectory records were not trainable because observations were not stored

Files:

- `python/evolution_sim/env/runtime/trajectory.py:14`
- `python/evolution_sim/env/world.py:2820`

`TRAJECTORY_RECORD_FIELDS` includes `observation_digest` and `action_mask`, but not the actual observation. `_begin_trajectory_decision()` stores only the digest at `world.py:2835`.

This is useful for integrity checks, but not sufficient training data. A learner cannot train from a replay without reconstructing observations by replaying the simulator exactly, which couples training to implementation details and makes dataset archival weak.

Consequence for Mind: the first real dataset will either be untrainable or require a separate side-channel. That side-channel would become the real contract, bypassing the one in replay.

Required fix: store a compact, versioned observation payload or tensor per decision. Keep the digest as an integrity field.

Status: addressed for full replay trajectory rows in the 2026-04-27 observation contract slice, then extended to low-memory collection in the trajectory streaming slice. Records include compressed, versioned `observation_input` payloads plus policy-excluded `observation_metadata`, and `npm run sim:trajectory` streams the same contract without full replay payloads.

### 3. Trajectory logging is tied to full replay mode

File: `python/evolution_sim/env/world.py:196`

Before the 2026-04-27 trajectory streaming slice, `run()` set `record_trajectory = mode == RunMode.FULL_REPLAY` at `world.py:202`, and tests asserted that summary-only mode omitted trajectory bookkeeping.

Full replay is intentionally heavy. Mind training needs long rollouts and many seeds, so trajectory collection must be available in a mode that does not retain viewer frames and event-heavy replay surfaces.

Consequence for Mind: the practical training path will either explode output size or skip the current trajectory path entirely.

Required fix: add an explicit trajectory mode or output sink, independent of viewer replay mode. Examples: JSONL, parquet, chunked gzip JSONL, or callback-based streaming.

Status: addressed for gzip JSONL. `SimulationWorld.run()` now accepts explicit trajectory recording and a `trajectory_sink`; `JsonlTrajectoryWriter` streams records independently of full replay, and `npm run sim:trajectory` exposes the path.

### 4. Pre-tick masks and live action resolution can diverge

Files:

- `python/evolution_sim/env/world.py:267`
- `python/evolution_sim/env/world.py:287`
- `python/evolution_sim/env/runtime/trajectory.py:79`

The world captures observation snapshots before the agent loop, including the pre-tick action mask. It then runs agents sequentially. For each agent, it recomputes a live action mask at `world.py:288` and resolves invalid actions to stay at `world.py:289`. `trajectory.py:79` computes `action_valid` from the pre-tick mask, not from the live mask that actually governed resolution.

The sequential simulator may allow an earlier agent to change occupancy or resources before a later agent acts. That can make `action_valid` disagree with the actual resolution path.

Consequence for Mind: invalid-action penalties and imitation labels can be wrong in exactly the multi-agent conflicts Mind will need to learn around.

Required fix: log both `observation_action_mask` and `resolution_action_mask`, or move to a true two-phase action model where all actions are selected from the same pre-tick state and conflicts are resolved explicitly.

### 5. Observation schema is not frozen tightly enough

File: `python/evolution_sim/env/runtime/observations.py:53`

`observation_contract()` lists field names and `privileged_world_state: False`. It does not define dtypes, ranges, enum vocabularies, tensor order, shape, missing-value policy, or what fields are metadata versus policy input.

`build_observation()` includes `agent_id` at `observations.py:72`. That may be useful metadata, but it should not be part of policy input because it creates identity leakage and weak generalization.

Consequence for Mind: two training implementations can both claim to use `mind_observation_v1` while encoding different tensors.

Required fix: split observation metadata from model input, publish a canonical encoder with shape/range tests, and make the gate validate it.

Status: addressed for the data contract and default policy path. `mind_observation_v2` separates `metadata` from policy input, declares a canonical encoder and enum vocabularies, includes bounded navigation cues, and the gate validates encoded payloads. The default heuristic now consumes this observation boundary through `Policy.decide(observation, action_mask)`.

## High Findings

### 6. Foundation gate is improved but still too shallow for Mind readiness

File: `python/evolution_sim/cli/foundation_gate.py:302`

The current gate now checks Mind contract metadata and first trajectory records. This fixes the older claim that the gate has no Mind checks at all. The 2026-04-27 observation/reward slice also made `_mind_contract_flags()` validate canonical encoded observation payloads, policy-excluded observation metadata, and versioned reward bounds.

Remaining gate gaps:

- release profile must prove the same ecological guarantees at longer horizons and more seeds;
- long-run release profile preserves trophic/carrion signals across terminal state, not only aggregate sweep totals;
- animal-specialist reproduction and consumption strength must be promoted from diagnostics into explicit release criteria once the desired thresholds are chosen.

Consequence for Mind: the quick gate can pass while release-horizon ecology remains under-proven.

### 7. Benchmark RSS is process-lifetime and platform-dependent

File: `python/evolution_sim/cli/bench.py:53`

`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` is a lifetime process peak. Sequential scenarios inherit peaks from earlier scenarios. On macOS, `ru_maxrss` is bytes; on Linux, it is KiB. The code labels the metric `peak_rss_kib` unconditionally.

The quick benchmark reported `91701248` as `median_peak_rss_kib` for a 20 tick summary run on macOS, which is actually about 91.7 MB if interpreted as bytes, not about 91 GB.

Consequence for Mind: memory regressions and per-scenario comparisons are currently unreliable.

Required fix: run every scenario repetition in a fresh worker process, normalize units by platform, and report both peak RSS and output size separately.

Status: addressed in the 2026-04-27 benchmark hardening slice for scenario repetition isolation and KiB normalization. Replay output size remains reported separately as `median_replay_size_bytes`.

### 8. Trophic profile cache key fast path is dead

File: `python/evolution_sim/env/runtime/lifecycle.py:33`

`genome_profile_key()` returns raw genome attribute values. `Agent.genome_vector` is produced by `genome_vector()` and normalizes genes. `cached_trophic_profile()` compares the normalized vector to the raw tuple at `lifecycle.py:44`, so the optional fast path never activates under normal genomes.

Consequence: this is probably not a correctness bug because the raw key is recomputed, but it is misleading and wastes the intended cache reuse path.

Required fix: either store the raw profile key on the agent, or change the cache to use the normalized vector consistently and deliberately.

Status: addressed in the 2026-04-27 hardening slice by making the cache deliberately raw-genome-keyed and removing the unused normalized-vector fast path.

### 9. Summary trait averages mix dead and alive agents

File: `python/evolution_sim/env/world.py:4626`

`_build_summary()` creates `genomes = [agent.genome for agent in self.agents.values()]`, then reports fields like `avg_max_energy_gene` from that historical population. The names do not indicate that dead agents are included.

Consequence for Foundation and Mind: selection pressure at terminal state can be masked by extinct lineages. This matters if gates or reward calibration use trait averages as evidence of adaptation.

Required fix: rename current fields to `avg_historical_*_gene` or add explicit `avg_alive_*_gene` fields and use alive fields in readiness checks.

Status: addressed in the 2026-04-27 hardening slice by adding explicit historical and alive summary fields. Existing `avg_*_gene` fields remain historical aliases for compatibility.

### 10. Reproduction blocking is mostly invisible

Files:

- `python/evolution_sim/env/world.py:3635`
- `python/evolution_sim/env/world.py:3662`

`_is_reproduction_ready()` includes `_has_empty_neighbor()`, so spatially blocked reproduction is not distinguishable from biological unreadiness. `_reproduce()` can still return `False` if a destination disappears before reproduction, but emits no event.

Consequence for Mind: reproduction pressure and failed opportunity are lost as analytics and reward signals.

Required fix: split biological readiness from spatial availability, emit `REPRODUCTION_BLOCKED`, and test both crowded-neighborhood and max-population cases.

Status: addressed in the 2026-04-27 reproduction analytics slice. Biological readiness is now evaluated separately from max-population and local-crowding availability, blocked opportunities emit `agent_reproduction_blocked`, and summary/evaluation/gate reports expose max-agent saturation plus blocked-reproduction counts.

### 11. Release gate has no progress reporting or timeout

File: `python/evolution_sim/cli/foundation_gate.py:573`

The quick gate passed. The current release gate was started with the release profile and output path, but after more than 30 minutes it had emitted no progress and had not written a report file. The process was CPU-bound, so this was not an idle wait.

Consequence for Foundation: a short quick gate is useful for CI but cannot prove sustained emergent behavior. The release gate is now operable, and that operability should be preserved as the ecological closure check.

Required fix: add per-seed/per-probe progress output, scenario timing, incremental report writing, and a timeout. Release readiness should require terminal and aggregate survival of resource pressure, trophic diversity, and meat modes across seeds, but that gate also has to complete predictably.

Status: addressed for gate operability in the 2026-04-27 gate slice and verified at release scale on 2026-04-30. `foundation_gate` now records per-seed/probe timings, emits progress to stderr through the CLI, writes incremental reports to `--output`, converts scenario timeouts/exceptions into gate blockers, and the latest release profile completed with no blockers or warnings.

### 12. Release durability still needs to be treated as the real ecological gate

The older release report in `output/evaluations/foundation-release.json` was only `review`, with terminal trophic/carrion weakness after 800 ticks. That finding is now historical. The 2026-04-30 release gate completed successfully after the ecology and replay-size repairs.

Consequence for Foundation: quick-gate pass alone does not close Foundation. Release readiness should continue to require terminal and aggregate survival of resource pressure, trophic diversity, and meat modes across seeds, especially after reproductive/signal behavior changes.

### 13. Invalid config can silently reshape the world

Files:

- `python/evolution_sim/config/schema.py:142`
- `python/evolution_sim/env/world.py:404`
- `python/evolution_sim/env/world.py:448`
- `python/evolution_sim/env/world.py:1591`
- `python/evolution_sim/env/world.py:4699`

`WorldConfig` is still a set of plain dataclasses. The runtime assumes positive dimensions, positive tick counts, positive season lengths, meaningful terrain ratios, bounded resource rates, and feasible spawn geometry.

Verified failures:

- `water_tile_ratio=-0.1` initializes successfully but creates a mostly-water world because `candidates[:negative_count]` assigns almost every candidate except the tail;
- `water_tile_ratio=1.2` fails as "No spawnable land tiles" during population spawn, not as a config error;
- zero width or height fails later with `ValueError: min() iterable argument is empty`;
- `season_length=0` crashes with `ZeroDivisionError`;
- `max_ticks=0` executes no tick but reports one tick.
- `initial_agents=30, max_agents=20` starts with 30 agents and only blocks future births after the population falls below cap;
- `max_age=20` with default reproduction `min_age=30` produced extinction before reproduction;
- negative hazard rates and negative attack costs are accepted;
- `fresh_kill_conversion_rate=2.0` is accepted even though conversion is treated defensively elsewhere.

Consequence for Foundation: tuning, ablations, and future Mind environments can accidentally run a different world than the config appears to request. This is a domino-class issue because every gate and benchmark assumes the environment dimensions and ratios mean what they say.

Required fix: add a real config validation boundary with clear error messages, make `SimulationWorld` call it before building fields or spawning agents, and add parameter-combination tests that cover both invalid individual values and invalid cross-field relationships.

Status: addressed in the 2026-04-27 hardening slice for the listed invalid ranges and cross-field relationships. Remaining config-contract work should focus on explicit policy decisions for performance limits, carrying-capacity semantics, and any additional tuning parameters that are not yet validated.

### 14. `SimulationWorld.run()` is one-shot but not enforced

File: `python/evolution_sim/env/world.py:196`

Running the same `SimulationWorld` instance twice does not reset agents, events, viewer frames, trajectory records, counters, or caches. A 2-tick full replay produced 2 frames on the first call and 4 accumulated frames after the second call.

Consequence for Mind: training environments usually have explicit episode reset semantics. If a wrapper reuses a world object, it can silently mix episodes and corrupt replay/trajectory data.

Required fix: either enforce one-shot execution with a clear error on second `run()`, or add a real `reset(seed/config)` path and tests proving all episode state is cleared.

Status: addressed in the 2026-04-27 hardening slice by enforcing one-shot execution with a clear `RuntimeError` on the second `run()`.

### 15. Action outcomes are not causal enough for learning

Files:

- `python/evolution_sim/env/world.py:3386`
- `python/evolution_sim/env/runtime/trajectory.py:60`

The attack path applies damage, may kill the target, may immediately consume the fresh kill, and only then emits `AGENT_ATTACKED`. The trajectory record for the attacker currently sees this mostly as `resource_gain`.

The missing action-outcome fields are exactly the fields a learned policy will need:

- target id and target species/ecotype;
- damage dealt and success;
- kill;
- immediate kill-feed;
- resource source tile;
- invalid reason when resolution becomes stay;
- passive outcomes such as being killed before acting.

Consequence for Mind: reward and imitation data will conflate "attacked successfully", "ate fresh kill", and "resource gain happened" unless action results become first-class records.

Required fix: introduce a versioned `ActionOutcome` schema and have trajectory rows attach the resolved outcome produced by `_resolve_action()`, not infer the outcome later from coarse tick events.

Status: addressed for active and passive action outcomes in the 2026-04-27 action-outcome slices. Trajectory rows now carry `mind_action_outcome_v1`, resolution masks, explicit attack and feeding outcome details, invalid-resolution reasons, passive damage/death fields, and passive rows for agents killed before their turn.

### 16. `_effective_tile_fields()` has a misleading season parameter

File: `python/evolution_sim/env/world.py:2594`

The function accepts `season`, but always starts from current tick climate state. It also caches by tick only. Passing `"wet"` or `"dry"` at the same tick returns the same effective fields in the verified dry-season probe.

Consequence for Foundation: any ablation or fixture that thinks it can score alternate season conditions through this helper is not doing what it says. The parameter is also misleading inside `DerivedTileMemo`.

Required fix: either remove the parameter and declare the helper current-season only, or make the helper accept a real climate state/season key and cache by that full input.

Status: addressed in the 2026-04-27 hardening slice by removing the `season` parameter and making the helper explicitly current-tick/current-climate only.

### 17. Species catalog centroids mix raw and normalized units

Files:

- `python/evolution_sim/genome/species.py:67`
- `python/evolution_sim/env/world.py:3915`
- `python/evolution_sim/env/taxonomy.py:900`

Runtime species centroids are raw gene values from `centroid_from_members()`. Replay-taxonomy species centroids are averages of `ReplayAgentRecord.genome_vector`, which is normalized 0-1 gene space, but they are emitted under the same gene names in `species_catalog`.

Consequence for analysis: a downstream consumer can read `species_catalog["centroid"]["max_energy"]` and get raw energy units before replay taxonomy, but normalized units after replay taxonomy. That can break gates, viewer interpretation, and future Mind dataset analysis.

Required fix: make centroid units explicit. Either store `centroid_raw` and `centroid_normalized`, or keep one canonical unit with a schema/version bump.

### 18. Viewer validation is still shallow for malformed replay files

File: `viewer/app.js:1822`

The viewer now rejects empty `frames`, which fixes the older empty-replay crash. But validation still does not check map width/height consistency, rectangular terrain/surface grids, required frame surface arrays, agent row length/types, monotonic frame ticks, catalog shape, or trajectory row shape.

Consequence for debugging: the viewer is the main inspection tool for Mind iteration. It should fail early with a replay-schema error instead of crashing deep in rendering or silently showing misleading surfaces.

Required fix: add a strict replay validator or generated schema check, and extend the smoke test to load intentionally malformed replays.

Status: partially addressed. The browser now uses a shared `viewer/replay_validator.mjs` module that validates map dimensions, rectangular terrain/base/frame grids, required frame surfaces, agent row length and in-bounds positions, monotonic frame ticks, signal emission schema/provenance, action/signal communication capacity consistency, catalogs, and first trajectory record shape. `npm run viewer:validate` covers malformed replay cases in Node, and `npm run viewer:smoke:malformed` covers the browser failed-load path. Remaining work is a generated schema shared with Python.

## Medium Findings

### 19. Fresh-kill feed event still has redundant `consumed` and `energy`

File: `python/evolution_sim/env/world.py`

The earlier report flagged that a fresh-kill event records the same value under `consumed` and `energy`. This should be simplified before freezing trajectory/replay consumers.

Consequence: not dangerous today, but duplicate authority in a schema becomes expensive once external tools depend on it.

### 20. `_kill_agent()` fresh-kill state remains brittle for future death semantics

File: `python/evolution_sim/env/world.py`

The old report's concern is still conceptually valid: death-event fresh-kill state is coupled to the current assumption that non-attack deaths do not create fresh kills. If that changes, event fields can become stale unless the state is refreshed after deposits.

Consequence: future carcass/fresh-kill changes can silently misreport event state.

### 21. Large world wrapper layer still widens the private API

File: `python/evolution_sim/env/world.py`

The delegating wrapper block remains. It is not a direct bug, but it makes `SimulationWorld` look like the owner of every runtime concern. The current heuristic uses that broad private surface directly, which is exactly the boundary Mind needs to remove.

Consequence: future policy code will be tempted to keep reading `world._*`.

### 22. Heuristic constants need a named config

File: `python/evolution_sim/env/runtime/actions.py`

The heuristic remains full of embedded thresholds such as `0.92`, `1.04`, `0.72`, `0.84`, and `1.12`.

Consequence: baseline comparison against Mind will be hard to reason about, tune, or freeze.

Required fix: introduce `HeuristicPolicyConfig`, version it, and include the config version in trajectory metadata.

### 23. Private API enforcement is still narrow

File: `python/evolution_sim/cli/test_runner.py`

The cache-poking guard has improved beyond the old regex-only complaint, but tests still rely on many private helpers. Some are justified. The repo needs an explicit testing namespace for sanctioned internals.

Consequence: future tests can keep normalizing direct `world._*` use, making the Mind boundary harder to enforce.

## Cross-Check Against The 2026-04-25 Plan

Accepted and still important:

- Controller boundary is the central blocker.
- Foundation gate must enforce the Mind data contract, not only ecology.
- Trajectory and reward contracts need more rigor before training.
- Viewer and replay are important because they become the debugging surface for Mind.
- Heuristic constants should be frozen as baseline configuration.

Fixed or mostly fixed on current branch:

- Taxonomy set iteration now sorts keys.
- npm scripts and CI set `PYTHONHASHSEED=0`.
- `min_meat_modes` no longer appears silently halved.
- Ecology pressure is no longer only a warning in the current gate logic.
- `bench.py` p95 is no longer `max()`.
- `run_headless.py` has safer output handling and help text.
- Summary-only repeated determinism tests exist.
- Action-mask impossible-eat coverage exists.
- Golden harness validates the live Python runtime and has safer update semantics.
- Viewer terrain legend escaping, empty frame validation, and unknown terrain rendering have been addressed.
- `_resolve_action()` now documents that the return value means "moved".
- `_total_variation_distance()` is used and should not be removed.
- Soft refuge is not dead; it is derived by refuge thresholds and has tests.

Rejected or downgraded:

- Fresh-kill conversion `min(x, x * rate)` is defensive, not a critical bug.
- Viewer Pixi state leakage is less severe in the current code than the old report stated, because the app is initialized once and layer children are cleared.
- The repo does have observation, trajectory, reward scaffolding, an enforced heuristic controller boundary, and a low-memory trajectory streaming path now. The remaining issue is release-scale ecological proof on top of those contracts.

Missed by the old plan:

- Trajectory stored only observation digests, not observations, before the 2026-04-27 observation contract slice.
- Trajectory collection is tied to full replay mode.
- `action_valid` can be computed from a different mask than the one used for action resolution; both masks are now recorded and covered by a real conflict regression.
- `agent_id` appeared inside observations and needed to become policy-excluded metadata.
- Benchmark RSS is also cross-platform unit-broken, not only process-cumulative.
- Trophic cache key fast path compares normalized and raw genome vectors.
- Summary gene averages are historical, not terminal alive population.
- Biological reproduction readiness and spatial blockage are conflated.
- Invalid `WorldConfig` values can crash late or silently produce very different worlds.
- `SimulationWorld` has no explicit one-shot/reset contract.
- Attack trajectories do not carry the causal result of the action.
- Effective-field season arguments do not compute alternate seasons.
- Replay-taxonomy species centroids use normalized units under raw-looking gene keys.
- Sexed, intentional reproduction is a Foundation semantics change, not a Mind afterthought.

## Addendum: Gaps Missed In This Audit

The first version of this document still underweighted Foundation quality. It focused heavily on the controller boundary and trajectory plumbing. Those are real blockers, but they are not the whole readiness question.

### A. Terminal Ecology Is Not Proven Ready

The older release report ended seeds `3,7,11,17,29` at 800 ticks with only herbivores alive:

- total terminal trophic counts: `1591 herbivore`, `0 omnivore`, `0 carnivore`;
- total terminal meat modes: `1591 none`, `0 hunter`, `0 scavenger`, `0 mixed`;
- plant energy share was effectively `1.0` in every seed;
- carcass energy was deposited in every seed, but almost none was consumed.

This means Foundation currently proves that plants can sustain a population, not that the environment supports durable multi-trophic ecology. Mind v1 would mostly learn plant foraging unless the animal-resource niche is fixed and validated first.

### B. The Hard Population Cap May Be The Real Carrying Capacity

The same release artifact shows `peak_alive_agents == 320` in every seed, which is exactly `WorldConfig.max_agents`. Alive counts end at `317-319`, and last birth tick is `799` for all five runs.

That suggests the ecosystem may be bouncing against an administrative cap rather than settling into an ecological carrying capacity. For Mind, this matters because reproduction success, crowding pressure, and lineage fitness are partly shaped by an invisible global ceiling.

Needed metrics:

- ticks spent at or near `max_agents`;
- reproduction blocked by max population;
- reproduction blocked by local crowding;
- deaths by cause while population is saturated;
- per-role birth/death rates under saturation.

### C. The Observation Contract Is Weaker Than The Heuristic's Sensorium

`WorldConfig.default_vision_radius` is `4`. The heuristic scans that radius for food and water targets, and carnivore/scavenger paths can extend to radius `6`. The current observation contract uses `LOCAL_PATCH_RADIUS = 2`.

Even if the heuristic were refactored to consume observations, a learned policy using the current observation would not see the same decision surface. That makes heuristic imitation and A/B comparisons unfair by construction.

The observation schema needs an explicit sensor budget:

- local patch radius;
- whether long-range smell/biotic fields are allowed;
- whether carrion/prey signals are local grids or scalar directional cues;
- whether the heuristic baseline is restricted to the same sensor budget.

### D. Sorted Agent Turn Order Is An Ecological Bias

The tick loop processes actions, reproduction, and deaths in `sorted(self.agents)` order. Lower-id agents always act first. In a crowded world, that decides who reaches resources, who occupies tiles, who attacks first, and who gets reproduction space.

This is deterministic, but deterministic is not the same as unbiased. For Mind, a stable ID-order advantage can leak into trajectory labels and lineage success.

Options:

- keep deterministic order but rotate priority by tick;
- use a seeded shuffled order per tick;
- move to a two-phase simultaneous action model with explicit conflict resolution.

Whichever choice is made should become part of the replay contract.

### E. The Gate Proves Surface Presence, Not Causality

Current tests and gates do a good job proving that hydrology, hazards, ecology states, carcass surfaces, and trophic counters exist and are internally consistent. They do not prove that those systems causally shape behavior in the intended way.

Foundation needs an ecological acceptance battery:

- ablate hazards and show hazard damage/shelter/heat tolerance stop mattering;
- ablate carrion and show scavenger niches degrade;
- ablate hydrology and show hydration-driven behavior changes;
- reduce plant regrowth and show survival/reproduction respond;
- increase predation affordances and show animal-resource roles can persist;
- run random, stay-only, heuristic, and simple oracle policies to show the environment is not only viable under the hand-tuned heuristic.

### F. Resource Budgets Are Incomplete

Fresh-kill and carcass flows have conservation-style counters. Plant food, vegetation, hydration, hazard damage, and reproduction energy do not have equally strong budget accounting.

For Mind, reward design and environment tuning need budget clarity:

- food created by regrowth;
- food removed by consumption;
- food lost by parched/flooded states;
- energy spent on metabolism, movement, attack, reproduction;
- healing energy or health deltas by source;
- water access and hydration deficits by terrain/season.

Without this, it is hard to know whether a learned policy improved behavior or merely exploited a hidden inflow.

### G. Automatic Reproduction Limits Mind's Behavioral Surface

Reproduction is automatic after threshold checks. A policy can only influence it indirectly through survival, position, hydration, health, and energy. That may be acceptable for Mind v1, but it should be an explicit design decision.

If the desired emergent behaviors include mate seeking, nesting, dispersal, crowding avoidance, parental timing, or reproductive tradeoffs, reproduction cannot stay entirely outside the action space forever.

### H. Performance Is A Foundation Requirement

The earlier release gate run that exceeded 30 minutes without producing a report was not only tooling friction. It exposed why Mind training needs explicit performance budgets. The 2026-04-30 release gate now completes, but performance remains a Foundation contract because Mind training will need many more environment steps than the release gate.

The likely cost centers include:

- per-agent heuristic scans over radius `4-6`;
- per-agent biotic invalidation after each action;
- biotic diffusion with radius `18`;
- full replay JSON and taxonomy probes;
- repeated recomputation of derived fields during decision-making.

Foundation should define a target such as environment steps per second at `48x32`, `320` agents, summary/trajectory mode, and full replay mode. Without that target, Mind work will hit performance cliffs before policy quality is measurable.

### I. Selection And Heritability Are Not Yet Audited

The simulator has genomes, mutation, species, ecotypes, and trait summaries. The missing proof is that selection is doing useful work:

- initial role/trait distribution versus terminal alive role/trait distribution;
- per-role survival and reproduction rates;
- trait heritability across parent/child pairs;
- whether terminal trait drift is adaptive or just survivor bias;
- whether speciation/ecotype splits correspond to ecological differentiation.

This belongs in Foundation because Mind will learn on top of whatever evolutionary pressure exists. If the evolutionary substrate collapses to one role, Mind starts from a narrow world.

### J. Config Was Not A Validated Contract

Status: addressed for the audited invalid ranges and cross-field relationships on 2026-04-27. `WorldConfig` and nested dataclasses now validate before runtime setup, and runtime contract tests cover invalid ranges, invalid cross-field relationships, mutated config validation, spawn capacity, and one-shot episode behavior.

Before that slice, `WorldConfig` and its nested dataclasses defined defaults but no validation. That mattered because many runtime helpers assumed positive dimensions, positive tick periods, bounded ratios, and nonnegative rates.

Verified concrete bug: `SimulationWorld(WorldConfig(climate=ClimateConfig(season_length=0))).run()` raises `ZeroDivisionError` in `world.py:1592`, because `_season_state()` divides by `self.config.climate.season_length` before the local `max(..., 1)` guard is applied.

Other risky examples from the same pattern:

- terrain ratios are converted directly to target counts in `world.py:409-414`; negative ratios or sums above available plain tiles are not rejected, they are silently interpreted by list slicing and sequential assignment;
- dimensions of zero or negative values flow into environment-field generation and terrain/grid construction;
- resource, hazard, carcass, and reproduction rates can be set outside meaningful ranges;
- `max_agents < initial_agents` and impossible spawn geometry are not rejected as configuration errors.

Foundation now has a `validate()`/`__post_init__` layer and tests that invalid configs fail early with clear messages. Remaining config work should focus on explicit limits for performance-sensitive parameters and on future reproduction semantics.

### K. Default Ecology Lost Animal Niches Early

Status: addressed for the quick profile on 2026-04-27, the 20-seed, 120-tick ecology profile on 2026-04-28, and the release gate on 2026-04-30. Animal-specialist reproduction should remain a watched metric, especially after reproductive/signal changes.

The older release artifact showed terminal animal-resource collapse at 800 ticks. A pre-repair probe found the same direction by 120 ticks:

- seed 1: `11` alive, all herbivore, all meat mode `none`, animal energy share `0.0123`;
- seed 2: `102` alive, all herbivore, all meat mode `none`, animal energy share `0.0002`;
- seed 3: `93` alive, all herbivore, all meat mode `none`, animal energy share `0.0041`;
- seed 7: `76` alive, all herbivore, all meat mode `none`, animal energy share `0.0018`;
- seed 11: `98` alive, all herbivore, all meat mode `none`, animal energy share `0.0`.

The pre-repair expanded 20-seed 120-tick sweep strengthened this finding: `18/20` seeds ended herbivore-only with no terminal animal-resource mode. The only exceptions were seed `6`, with one terminal scavenger/omnivore, and seed `15`, also with one terminal scavenger/omnivore. That pass/fail evidence is now stale for the repaired 120-tick ecology and release profiles, but still explains why release diagnostics and animal-specialist birth/death metrics are required.

This does not contradict the fixture tests. The tests prove the mechanics can favor scavengers, hunters, herbivores, and omnivores in constructed arenas. The default generated ecology now proves a 120-tick seed-bank floor and has a passing release profile, but those gates must continue to catch regressions as reproduction and signal semantics change.

Foundation work therefore needs two tracks:

- keep the fixture arenas as causal unit/integration evidence;
- add default-world release evidence that those same niches survive without hand-built arenas.

### L. Fixture Causality Exists But Is Not Release Readiness

The test suite has important Foundation coverage that the old plan underweighted:

- `test_fixture_plant_only_arena_favors_herbivore_births`;
- `test_fixture_carrion_only_arena_favors_scavenger_over_hunter`;
- `test_fixture_prey_rich_arena_favors_hunter_over_scavenger`;
- `test_fixture_mixed_stable_arena_specialists_outperform_omnivore_in_own_channels`;
- `test_fixture_shocky_seasonal_arena_favors_omnivore_survival_not_reproduction`;
- `test_fixture_low_productivity_cascade_hurts_animal_specialists_more_than_herbivores`;
- `test_production_readiness_mixed_world_sweep`.

Those are strong subsystem tests. The release gate now provides the default-world closure check, but the causal fixtures should continue to inform release thresholds as reproductive and signal behavior changes.

CI currently runs the quick gate in the regular job and the full simulator test/benchmark in the expensive job, but not `sim:gate:release`. Keep release as the explicit ecological closure check; schedule it only if runtime cost is acceptable.

### M. Derived-State Cost Is Already A Foundation Concern

A seed-7, 20-tick summary-only probe rebuilt biotic state 170 times. The mechanism is understandable: the heuristic reads biotic fields while choosing actions, and `_run_tick()` invalidates biotic state after each agent action.

This is not a correctness failure by itself, but it means Mind readiness cannot treat performance as later cleanup. Additional probes showed the cost cliff:

- saturated default-map start with 320 agents took `13.5641s` for 40 summary-only ticks, compared with `0.782s` for the 20-agent baseline;
- setting biotic diffusion radius to `128` took `3.7446s` for only 20 ticks;
- default 800-tick summary-only seeds `3`, `7`, and `11` each exceeded `45s`;
- compact 1000-tick summary-only seeds `1-5` each exceeded `35s`.

Long trajectory rollouts will multiply:

- per-agent heuristic or policy evaluation;
- observation construction;
- action masks;
- biotic diffusion;
- reward construction;
- trajectory serialization.

Foundation needs cost counters and profiles before the observation/trajectory design is frozen, because a schema that is technically correct but too expensive will not support training.

### N. Reproduction Economics Need Explicit Design

Reproduction is currently automatic and indirectly controlled by state. That is acceptable only if it is declared as a Mind v1 design constraint.

Status: partially addressed for observability and Stage 1 biology, not for the full design. Biological readiness is now separated from max-population and local-crowding blockage, blocked reproduction emits events, summaries/evaluations expose blocked counts, and rare same-group two-parent reproduction exists as a biology-gated path. Policy-visible mate attempts, reproductive roles, and hybridization remain future work.

The remaining mechanics still need explicit decisions:

- `_reproduction_block_reason()` still calls `len(self.alive_agents())` for each reproduction candidate, which is avoidable work under large populations;
- parent reproduction cost is a fixed absolute energy amount, while thresholds and child energy are genome-scaled.
- policy-directed mate seeking, reproductive roles, and hybridization are not implemented and would require state, action, observation, reward, replay, and gate changes before Mind schemas are frozen.

None of these alone blocks the simulator, but together they make reproductive pressure hard to measure, reward, or tune.

### O. Episode Lifecycle Is Not A Contract Yet

`SimulationWorld.run()` works as used by the CLIs and tests because they instantiate a fresh world for each run. It is unsafe as an environment API. A second call on the same object continues from mutated agents while resetting `self.tick` to zero and accumulating replay artifacts.

Before Mind wrappers are built, choose one:

- one-shot world objects, with a guard that raises on a second `run()`;
- or reusable environments, with `reset()` clearing agents, grids, counters, events, frames, trajectory records, caches, and species/ecotype registries.

The first option is simpler and probably right for the current architecture. The second option is useful only if training throughput requires object reuse, and it needs tests.

### P. Action Causality Is Still Event-Derived

The simulator records rich events, but trajectory rows do not own the resolved action result. That is fine for replay viewing and weak for learning. An attack can cause damage, death, immediate feeding, hazard damage after the action, and later reproduction/death logic in the same tick. The trajectory row needs a causal `ActionOutcome` object rather than a post-hoc summary.

This is especially important if actions become simultaneous or if reproduction becomes an action. Without explicit outcome objects, the reward layer will keep inferring causality from event order.

### Q. Seasonal Field Semantics Need Tightening

The effective-field helper should either be "current tick only" or "evaluate this requested season/climate." It currently looks like the second but behaves like the first. This matters for ecological ablations, fixtures, and any future curriculum that varies season conditions.

### R. Replay Analytics Need Unit Discipline

The species centroid unit mismatch is a symptom of a broader schema discipline issue: replay analytics mix runtime lineage species, replay-taxonomy species, raw genes, normalized vectors, and viewer-friendly labels. That is workable only if each field declares its identity mode and units.

Before Mind datasets are generated, every numeric replay field that could become a feature, label, reward, or gate input should be classified as raw unit, normalized unit, code/enum, counter, or rate.

### S. Sexed Intentional Reproduction Is A Foundation Redesign

Adding sexed recombination, reproductive groups, mate attempts, pheromones, and later X/Y/Z-like reproductive expression is not a small feature. It changes the environment's core selection pressure and the data contract Mind will learn from.

This is now explicitly in scope for pre-Mind Foundation cleanup. The detailed architecture is in [`pre-mind-reproductive-and-signal-readiness-plan.md`](pre-mind-reproductive-and-signal-readiness-plan.md). The important direction is not "add fixed male/female from tick zero"; it is staged biology:

- start with current asexual clone-and-mutate behavior;
- add a reproductive genome module and live reproductive groups;
- introduce rare facultative same-group two-parent recombination;
- reserve mate/communication action slots while keeping policy-facing IDs opaque;
- add pheromone/signal substrates without hardcoded signal meanings;
- add X/Y/Z-like reproductive expression only after group-stage thresholds;
- allow rare gated hybridization later, not at the beginning.

Required design decisions:

- agent state: add reproductive group, reproductive-stage/capability flags, and later reproductive expression to `Agent`, frames, catalogs, observations, trajectory rows, and replay schema;
- initialization: keep Stage 0 asexual and make sexed capability mutation-gated rather than assigning fixed roles to every first organism;
- action space: add `reproduce`, `mate`, or `court` as explicit actions, with masks and invalid reasons;
- partner rule: small-radius search, one-sided attempt versus mutual readiness, same-tick conflict resolution, compatibility scoring, and deterministic tie-breaking;
- costs: decide whether one or both parents pay energy/hydration/cooldown cost;
- genetics: decide grouped genome inheritance, recombination from two parents, reproductive-module mutation, inbreeding penalties, and later sex-linked/plastic traits;
- space: keep child placement local, nest-like, or nearest empty tile, and emit blocked reasons for no partner/no space/max population;
- rewards and analytics: log mate opportunities, attempts, success, blocked attempts, per-expression survival, per-expression reproduction, effective population size, lineage diversity, inbreeding risk, and signal costs.

This also changes the recommended baseline suite. A stay-only, random-valid, heuristic, and later simple mate-attempt baseline should be compared before learned policies are trained. The default heuristic should not emit opaque communication tokens or chase pheromones by human-readable meaning.

## Executable Guarantees To Add

This is the audit-to-test conversion layer. Each item below should become a unit test, integration test, gate criterion, benchmark assertion, or replay/schema validator before Foundation is declared closed.

| Risk class | Current evidence | Executable guarantee |
| --- | --- | --- |
| Parameter combinations | Negative ratios, zero dimensions, zero season length, `initial_agents > max_agents`, `max_age < reproduction.min_age`, negative rates, and conversion rates over 1 were accepted or failed late before the 2026-04-27 hardening slice. | Implemented initial coverage in `test_world_config_rejects_invalid_ranges_early` and `test_world_config_rejects_invalid_cross_field_relationships`; validation now runs before terrain generation and spawn. |
| Terrain generation | High or negative ratio combinations silently produce worlds far from the config's apparent meaning. | `test_terrain_counts_match_validated_ratio_policy`; impossible terrain requests fail with a config error or documented normalization. |
| Episode lifecycle | Reusing one world accumulated events, frames, and trajectory records before the 2026-04-27 hardening slice. | Implemented `test_simulation_world_run_is_one_shot`; training wrappers cannot mix episodes by accidentally reusing a `SimulationWorld`. |
| Long-horizon ecology | Earlier 120-tick and 800-tick probes collapsed to herbivores. The latest quick, ecology, and release gates now pass, but reproductive/signal changes must not regress this. | `test_default_seed_sweep_preserves_late_animal_niches` and release-gate thresholds for terminal role/mode diversity, animal-energy share, carrion consumption, animal-specialist reproduction, and no herbivore-only collapse at release horizons. |
| Stochastic edge cases | Quick seeds, the fixed 20-seed ecology profile, and the release profile now preserve enough animal-resource behavior to pass. Randomized nightly behavior remains future hardening. | Seed-bank tests over fixed rare seeds and randomized nightly seeds report extinctions, one-role collapse, cap saturation, no-animal-mode collapse, no-animal-consumption seeds, and animal-specialist birth/death ratios. |
| Hydrology + hazards + reproduction + predation | Quick and ecology gates now show survival, births, hazards, attacks, kills, fresh-kill consumption, and terminal animal-resource modes in generated-world runs. Stress and release-scale variants still need promotion into gates. | `test_interaction_stress_preserves_cross_system_signals`; gate requires nonzero hydrology pressure, hazard damage, reproduction opportunity, predation, and late-window animal-resource survival in the same generated-world runs. |
| Performance cliffs | Saturated 40-tick run took `13.5641s`; huge diffusion radius 20-tick run took `3.7446s`; long summary runs hit timeouts. | `test_summary_rollout_step_budget`, `test_biotic_diffusion_radius_budget`, and benchmark CI with per-scenario subprocess RSS/time budgets. |
| Schema drift | Full replay shapes matched declared fields, but species centroids were normalized under raw gene keys; trajectory lacked observation payloads and had weak outcomes before the 2026-04-27 action/observation contract slices. | Implemented centroid-unit coverage in `test_species_centroid_units_are_explicit`; observation payload, action outcome, reward-bound, policy metadata, and real resolution-conflict coverage now exist in runtime contract tests. |
| Viewer/replay trust boundary | Viewer validates only shallow shape. | Malformed-replay smoke tests for map dimensions, surface shapes, agent row length/types, frame tick order, catalog units, and trajectory rows. |
| Future sexed reproduction | Current automatic asexual reproduction touches state, actions, rewards, gates, replay, taxonomy, and analytics. | `test_reproductive_group_state_schema`, `test_mate_action_mask_and_invalid_reasons`, `test_partner_conflict_resolution_is_deterministic`, `test_recombination_and_mutation_contract`, `test_inbreeding_penalty_is_recorded`, and `test_x_y_z_stage_does_not_silently_dead_end_population`. |

These guarantees should be implemented as failing tests only when the corresponding fix is in the same change set. Until then, they are acceptance criteria in this plan rather than red CI.

## New Plan Of Action

This order is intentional. Do not start Mind v1 after only wiring observation and trajectory contracts. The release gate is now a passing regression boundary; next implement the major reproductive and signal semantics in Foundation, keep observability aligned with that stable environment, then train Mind.

### Phase 0: Complete The Audit And Fix Measurement

Goal: make claims about readiness trustworthy before tuning or building Mind surfaces.

1. Finish the code-level audit as a checklist, not a one-pass read:
   - `WorldConfig` parameter ranges and implied carrying capacity;
   - terrain generation and hydrology topology;
   - resource regrowth, vegetation, shelter, recovery debt, and food capacity;
   - hazard generation, damage, healing, and refuge effects;
   - trophic profile derivation, diet matching, combat, carcasses, and fresh kills;
   - reproduction, mutation, heredity, lineage, species, and ecotype logic;
   - tick ordering, cache invalidation, and summary/full-replay mode differences;
   - viewer/replay fields used for debugging and analysis.

2. Add config validation before more tuning (initial slice implemented 2026-04-27):
   - positive dimensions, `max_ticks`, season lengths, and drift periods;
   - bounded terrain ratios, no negative target counts, and explicit behavior when requested terrain counts exceed available tiles;
   - nonnegative resource, hazard, carcass, combat, and reproduction rates;
   - `initial_agents <= max_agents`;
   - enough spawnable land for requested initial agents;
   - tests for invalid config failures, including `season_length=0`, zero dimensions, negative ratios, ratios over 1, and impossible spawn worlds.

3. Fix benchmark RSS (scenario isolation and KiB normalization implemented 2026-04-27):
   - execute every scenario repetition in a fresh subprocess;
   - normalize `ru_maxrss` units by platform;
   - rename fields if units change;
   - add a regression test for metric plausibility.

4. Fix trophic profile cache key semantics (raw-key cache semantics implemented 2026-04-27):
   - pick raw or normalized key intentionally;
   - remove the dead optional parameter if not needed;
   - add a cache-hit regression test.

5. Keep taxonomy sorted-key behavior and `PYTHONHASHSEED=0` as non-negotiable invariants.

6. Replace ambiguous summary gene averages (explicit historical/alive fields implemented 2026-04-27):
   - add `avg_alive_*_gene`;
   - preserve old keys only if replay compatibility requires it;
   - document the difference.

7. Add performance baselines:
   - summary-only steps per second;
   - trajectory-mode steps per second;
   - full-replay steps per second;
   - per-tick cost at `20`, `160`, and `320` live agents;
   - profiler output for biotic diffusion, heuristic scan, resource regrowth, and taxonomy.

8. Remove or harden unbounded helper paths:
   - replace `_random_empty_land_tile()`'s unbounded loop with a bounded candidate selection or remove it if truly unused;
   - make impossible spawn/reproduction states fail explicitly rather than by implicit looping or silent `False`.

9. Define episode lifecycle semantics (one-shot guard implemented 2026-04-27):
   - enforce one-shot `SimulationWorld.run()`;
   - or add a tested `reset()` API for training wrappers;
   - ensure events, frames, trajectory records, caches, species/ecotype registries, counters, and tick state cannot leak across episodes.

10. Tighten seasonal field semantics (current-tick helper contract implemented 2026-04-27):
   - remove the unused/misleading `season` argument from `_effective_tile_fields()`;
   - or make it truly compute requested-season values and cache by the full season/climate input.

11. Fix `ticks_executed` semantics for zero or invalid tick counts by validating `max_ticks > 0`, and add an explicit test.

12. Create the executable guarantee suite from this audit:
   - implement config validation tests first;
   - implement lifecycle and schema drift tests second;
   - implement ecological seed-bank and interaction stress tests once the current known failures are fixed;
   - wire pass/fail thresholds into the Foundation gate rather than leaving them as ad hoc audit scripts.

### Phase 1: Repair And Prove Ecological Foundation

Goal: the environment should sustain learnable, nontrivial pressure before Mind-specific observability is treated as the main project.

1. Keep the release gate as the regression boundary. The old herbivore-only release result was disproven by the 2026-04-30 release pass, but every reproductive/signal behavior change must preserve:
   - terminal trophic role diversity;
   - terminal meat-mode diversity;
   - carrion and fresh-kill availability/consumption;
   - predator/prey balance;
   - durable scavenger opportunities;
   - nonzero ecology pressure.

2. Keep the repaired 120-tick ecology seed bank as the cheap regression guard:
   - add a cheap default-world sweep at 120-200 ticks;
   - require at least some terminal or late-window animal-resource survival across seeds;
   - report initial versus terminal role/mode distributions so initial diversity cannot hide terminal collapse.

   Status: the post-repair summary-only seed-bank gate now exists as `npm run sim:gate:ecology`. It runs seeds `1-20` for `120` ticks, writes `output/evaluations/foundation-ecology-current.json` incrementally, and passes with `--fail-on-blockers` as of 2026-04-28. Its current criteria are intentionally modest: one late-window animal-resource mode per run and at least two animal-resource modes across the aggregate seed bank. The report still highlights weak animal-specialist reproduction and should feed the release-horizon criteria.

3. Determine whether `max_agents` is hiding carrying-capacity behavior:
   - ticks spent near `max_agents`;
   - reproduction blocked by max population;
   - reproduction blocked by local crowding;
   - death causes while saturated;
   - per-role birth/death/survival rates under saturation.

4. Add resource and pressure budgets:
   - plant food created by regrowth;
   - plant food removed by consumption;
   - food lost to parched/flooded states;
   - water availability and hydration deficits by terrain/season;
   - energy spent on metabolism, movement, attack, and reproduction;
   - healing and health deltas by source;
   - fresh-kill and carcass conservation.

5. Add selection and heredity analytics:
   - initial versus terminal alive phenotype distribution;
   - per-role survival and reproduction;
   - parent-child trait inheritance and mutation deltas;
   - alive-only trait drift;
   - whether species/ecotype splits correspond to ecological differentiation.

6. Implement the reproductive/signal design before Mind:
   - keep Stage 0 automatic asexual reproduction as the starting behavior;
   - add reproductive groups and a reproductive genome module (Stage 0 registry
     and inert module implemented 2026-04-30);
   - keep grouped recombination helper logic separate from current mutation
     logic (implemented 2026-04-30);
   - add rare facultative same-group two-parent recombination (implemented
     2026-04-30 as a biology-gated path, with the `mate` action still masked);
   - reserve `mate` and communication action slots with opaque policy IDs;
   - add pheromone/signal substrate fields with intensity, range, duration,
     decay, and cost (implemented 2026-05-01 for reproductive-readiness
     emissions; communication remains reserved and opaque);
   - keep human-readable signal meanings out of policy observations;
   - emit blocked-reproduction events for no partner, partner not ready, incompatible partner, local crowding, and global saturation.

7. Normalize or document reproduction economics:
   - decide whether parent reproduction cost should be absolute or genome-scaled;
   - decide whether one or both parents pay costs under sexed reproduction;
   - define recombination and mutation semantics if two parents contribute genomes;
   - report reproduction opportunity, success, and blocked attempts by role and lineage;
   - report reproductive-expression ratio, per-expression survival, per-expression reproduction success, effective population size, and late-stage role-extinction risk;
   - include reproduction pressure in release analytics.

### Phase 2: Prove Foundation Causality And Efficiency

Goal: show the environment is causal, stable, and fast enough, not only richly instrumented.

1. Add causal ablations:
   - hazard ablation changes hazard damage and survival;
   - carrion ablation changes scavenger outcomes;
   - hydrology ablation changes hydration pressure;
   - plant-regrowth stress changes carrying capacity;
   - refuge ablation changes hazard exposure and survival.

2. Promote existing fixture arenas into the readiness story:
   - keep their focused assertions as tests;
   - summarize their outcomes in the release report;
   - add at least one generated-world check for each niche proven by fixtures.

3. Add policy baselines:
   - heuristic;
   - random valid action;
   - stay-only;
   - simple plant-foraging oracle;
   - simple predator/scavenger oracle;
   - if sexed reproduction is added, simple mate-seeking/reproductive-timing oracle.

4. Keep the release gate operable after new behavior lands:
   - progress output before and after each seed/probe;
   - per-scenario timing in the report;
   - incremental JSON writes so partial progress is inspectable;
   - timeout and failure messages for slow scenarios;
   - no silent replay-size or trajectory-memory regression.

5. Make release gate the ecological source of truth:
   - fail if terminal release populations collapse to one trophic role;
   - fail if all terminal meat modes are `none`;
   - fail or review if population spends too much time pinned at `max_agents`;
   - require nontrivial animal-resource consumption, not only carcass deposition;
   - require per-role birth/death/survival metrics;
   - require alive-only trait drift summaries.

6. Keep the release gate available as the opt-in ecological closure check; consider scheduled CI only if cost becomes acceptable.

7. Define explicit release thresholds from distributions, not from a desired narrative.

8. Add action-causality assertions:
   - attack outcome includes target id, damage, success, kill, and immediate kill-feed;
   - feeding outcome identifies source tile and source type;
   - invalid action outcome includes invalid reason;
   - passive death before own turn is represented in the tick/event stream.

9. Fix replay analytic unit contracts:
   - split raw and normalized gene centroids;
   - include identity mode for runtime lineage species versus replay-taxonomy species;
   - add tests that taxonomy rewriting preserves documented units.

### Phase 3: Build Pre-Mind Observability And Contracts

Goal: once Foundation behavior is credible, freeze the interfaces Mind will learn from.

1. Extend Mind contract checks (initial observation, policy, trajectory, and reward checks implemented 2026-04-27):
   - observation payload present;
   - canonical tensor encoder passes shape/range checks;
   - trajectory mode works without full replay;
   - policy boundary does not accept `SimulationWorld`;
   - reward fields are bounded and versioned;
   - invalid action resolution is logged with reasons;
   - action outcomes are first-class and not inferred from coarse tick aggregates.

2. Define:
   - `ActionDecision(requested_action, source, logits_or_scores optional, diagnostics optional)`;
   - `ActionOutcome(kind, valid, invalid_reason, target, damage, kill, resource_source, reproduction_result, diagnostics)`;
   - `Policy.decide(observation)`;
   - `ObservationEncoder.encode(observation)`.

3. Refactor heuristic (default observation-only policy implemented 2026-04-27):
   - it receives only observation plus a versioned `HeuristicPolicyConfig`;
   - all current thresholds move into that config;
   - live-world helper calls are removed from decision code.

4. Keep world-owned encoders:
   - the world may build observations from private state;
   - policies may not see the world.

5. Match the heuristic sensor budget:
   - either shrink the heuristic to `LOCAL_PATCH_RADIUS`;
   - or expand the observation contract to include equivalent long-range cues;
   - include sensor radius and directional signal definitions in the schema.

6. Remove sorted-ID action priority as a hidden policy feature:
   - rotate order deterministically;
   - or seeded-shuffle order per tick;
   - or implement simultaneous action selection and conflict resolution.

7. Add a boundary test:
   - monkeypatch or type-check policy invocation so a policy cannot access `world._*`.

8. Make trajectories trainable:
   - store actual observations or canonical encoded tensors per record;
   - keep `observation_digest` as integrity metadata;
   - store both observation mask and resolution mask, or move to two-phase action selection;
   - add streaming trajectory output independent of viewer replay;
   - include policy version, heuristic config version, observation schema version, reward schema version, and environment config digest;
   - include turn order, conflict resolution, and saturation/crowding context;
   - include target id/species for attack/mate actions when applicable;
   - include passive terminal events for agents killed before their own action if the learning setup needs terminal observations.

9. Define reward as a contract:
   - decide whether Mind v1 optimizes survival, reproduction, lineage persistence, resource acquisition, exploration, or a staged curriculum;
   - normalize resource gains so meat, plants, water, health, and reproduction incentives are comparable;
   - add terminal rewards and penalties with documented horizon semantics;
   - add blocked-reproduction and invalid-action penalties from explicit events;
   - if sexed reproduction is added, define rewards for mate seeking, mate attempt, successful birth, partner cost, and overpopulation pressure without creating degenerate mating loops.

10. Add transition tests:
   - valid move;
   - invalid move;
   - eat plant;
   - eat fresh kill;
   - eat carcass;
   - drink;
   - reproduce;
   - mate/reproduce with partner if sexed reproduction is enabled;
   - blocked reproduction;
   - death.

11. Add gate fixtures that intentionally violate each contract and assert failure.

### Phase 4: Viewer And Debugging Durability

Goal: the viewer stays useful through long Mind iteration.

1. Add trajectory overlays:
   - selected action;
   - invalid action;
   - reward components;
   - policy source;
   - local observation patch.

2. Add schema validation at replay load for all required viewer frame fields:
   - map width/height and rectangular grid shapes;
   - required surface arrays and dimensions;
   - agent row length and field types;
   - monotonic frame ticks;
   - catalog shape and centroid units;
   - trajectory contract shape.
3. Add a long-session smoke test that loads multiple replays in one browser session.
4. Add malformed-replay smoke tests that fail with clear validation errors rather than render-time exceptions.

### Phase 5: Mind v1

Goal: train a learned controller only after the environment and observability contracts are stable.

Entry criteria:

- release ecological gate passes without review;
- ablations and policy baselines demonstrate causal pressure;
- performance budget is met for summary, trajectory, and replay modes;
- heuristic and learned policies consume the same observation schema;
- trajectory output is trainable and deterministic;
- reward contract is versioned and tested;
- reproduction semantics chosen for Mind v1 are already implemented, gated, and represented in observation/action/reward contracts.

## Bottom Line

As of 2026-05-01, the Foundation has moved beyond inert scaffolding: live reproductive groups, same-group facultative sexed recombination, reproductive-readiness signals, config-aware signal/action contracts, opt-in opaque communication emissions, viewer replay validation, reproductive runtime boundary cleanup, and the first proto-role/X/Y/Z expression scaffold are implemented and covered by focused checks.

The active blocker is now semantic completion rather than scaffolding. Mind v1 should still wait until full Z group-imbalance behavior/costs, multi-offspring strategy, hybridization policy, and learned-state inheritance semantics are either implemented or explicitly deferred with stable contracts.
