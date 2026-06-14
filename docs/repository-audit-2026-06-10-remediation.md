# Repository Audit Remediation Plan

Date: 2026-06-10

This document preserves the action plan from the full repository audit covering
Foundation, replay/viewer, Mind v1/v2, Mind v3, CLI, docs, tests, CI, and local
agent skills. The audit found no evidence that existing recorded results are
invalid. The Foundation measurement boundary is mostly sound. The urgent
failures are process and strategy failures around Mind v3 durability,
reproducibility, and experiment direction.

## Audit Intake State

- Last commit: `e7d7e3a` on 2026-05-28.
- Current dirty tree at audit intake: 7 modified tracked paths and 126
  untracked paths, including 45 CLI modules, 40 `mind/` modules, 40 tests, and
  `docs/versioning.md`.
- `output/` is 8.9 GB locally; `output/mind/` is 7.5 GB locally and contains
  digest-referenced evidence that is gitignored.
- CI continuously exercises Foundation and viewer checks, but the push-time
  Mind gate is a two-tick smoke with vacuous thresholds. Strict Mind gates run
  only in the expensive scheduled/manual job.

## Current Remediation State

- P0 source preservation is complete: the v137-v176 backlog was inventoried,
  backed up locally, uploaded to the documented Google Drive archive, committed
  on `codex/p0-durability-backlog`, and merged to `master` as `628ff1e`.
- The habitat water encoding fix is complete in commit `36c1ab7` with
  runtime/viewer handling, focused regressions, generated viewer-contract
  refresh, and regenerated replay goldens.
- The first CI hardening slice is complete: push CI now runs a 20-tick
  non-vacuous Mind gate with default criteria and temporary outputs. This is
  still compact contract coverage, not promotion evidence.
- The shared Mind v3 harness extraction is complete: report/metric helpers live
  in `python/evolution_sim/mind/evaluation_helpers.py`, and fixture
  construction, run aggregation, strict gate helpers, digest validation, and
  leakage scans live in `python/evolution_sim/mind/evaluation_harness.py`.
  The v141/v142 live A/B implementations now live under `mind/`; their CLI
  entrypoints are wrappers.
- The strategy-reset data-support slices have reached the first training slice:
  v177 exact branch replay expansion emits compact transition rows with
  current/next public observations and action masks plus previous same-agent
  public context, v178 audited those rows as source-valid but support-limited,
  v179 expanded exact-branch transition-row support above the default v178
  support thresholds, and a durable default-threshold v178 audit authorized only
  the first same-lane opt-in transition-row training slice. v180 consumed that
  authorization and failed shadow acceptance as slice 1/10. The v180 report and
  policy artifact are durable only after the recorded backup
  `gdrive:evolution-sim-backups/archives/20260611T110135Z-v180-transition-row-policy-training.tar.zst`
  with archive SHA256
  `e9c95f424e0827a7ed25f7de523fc61e9fe38d6062861466a81f1149671aac59`.
  Caller-lowered `--min-*` thresholds remain diagnostic only and cannot
  authorize training routes. v181 has now run as diagnostics-only
  failure-response work, diagnosed sparse imputed transition-value support plus
  broad `eat` overrides and carrion no-coverage, and consumed no second training
  slice. Its report is durable after
  `gdrive:evolution-sim-backups/archives/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`
  with archive SHA256
  `a2b03e862ee095562c21482afcf0b1057a9ebb8f8b0b0d0f08e973dd884b1fd6`.
  v182 has now run as diagnostics-only failure-response design: it added
  imputed-valid-action and observed-support-floor diagnostics, made opt-in
  transition-value overrides abstain on imputed or below-floor currently valid
  action scores, and routed to exact transition-support expansion before any
  slice-2 training. The v182 report is durable after
  `gdrive:evolution-sim-backups/archives/20260611T165440Z-v182-imputed-abstention-design.tar.zst`
  with archive SHA256
  `d53401e5096d4c616a26691faa555ba53ffbdf84b4767061ff09d820340faf54`.
  v183 has now run as diagnostics-only exact transition-support expansion: it
  validated the pinned v182/v181/v180/v179 evidence, wrote a `144`-row expanded
  transition dataset with digest
  `e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, moved
  carrion target coverage from `0` to `18` materialized states, moved broad
  seed `19` support-hole coverage to `6` materialized states, and consumed no
  second training slice. The v183 report, dataset, and source trajectories are
  durable after
  `gdrive:evolution-sim-backups/archives/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`
  with archive SHA256
  `b4e57537142303e622656c8eb51fde114a9452e20641a85cfdb4d742b425eb5c`.
  v184 then ran a fresh v178-style audit of that canonical v183 expanded
  dataset. It passed source validation, schema, leakage, identity, action-mask,
  observation, default support, and digest-pin checks, but failed the target
  audit with `14` failures across `7` attack-forced rows that resolved to
  `stay` with `current_resolution_action_valid=false`. The v184 report exact
  digest is
  `ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`
  with archive SHA256
  `057d34c88f8a5da0c4ad98d87e4451caa043b7ee53c2af1037aa231dc5bcf382`.
  v185 then repaired the target-resolution blocker as diagnostics-only work by
  strict-filtering those `7` rows without backfill. The repaired dataset has
  `137` rows, `24` branches, `6` seeds, and `9` forced actions with digest
  `532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`. The
  v185 repair report exact digest is
  `327586651948e13670cf15285472a934c8cd0a8fc9784d2dcdb092d102c46486`; the
  paired repaired-dataset audit exact digest is
  `abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`. That
  audit passed source, schema, leakage, identity, action-mask, observation,
  target, and default-support checks and authorized only the future explicit
  route `v186_transition_row_policy_training_slice_2_opt_in`. v185 kept
  training/runtime/promotion/slice-2 lifecycle flags closed and is durable after
  `gdrive:evolution-sim-backups/archives/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`
  with archive SHA256
  `c64aa17964462a4bbbc71fc83b98779213c50e26b714aedffe8cb7b42b9b5d2a`.
  v186 then consumed the explicit slice-2 route from the repaired v185 dataset
  after validating the pinned repaired audit exact digest
  `abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`, repaired
  dataset digest
  `532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`, source
  producer `v185_v183_target_resolution_repair`, and route
  `v186_transition_row_policy_training_slice_2_opt_in`. It wrote report digest
  `f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075` and
  artifact digest
  `729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce`.
  Shadow acceptance failed with `0` `carrion_only@120` terminal survivors,
  dominant requested-action share `0.4179`, heuristic action-source count `0`,
  and no broad per-seed alive/birth regressions. Runtime integration,
  runtime action selection, and promotion stayed closed. The v186 report and
  artifact are durable after
  `gdrive:evolution-sim-backups/archives/20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst`
  with archive SHA256
  `9fba7700ec13b23f70f31b0269022c19b74bb7ed2967d2d70a9add05919eb31a`. The
  campaign budget is now 2/10. v187 then ran as a diagnostics-only v186-delta
  blocker review, wrote exact digest
  `4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6`, and
  recommended exactly
  `v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`.
  It consumed no slice 3, trained nothing, changed no runtime behavior,
  authorized no promotion, relaxed no gates, ran no support expansion, and
  reran neither v180 nor v186. The v187 report is durable after
  `gdrive:evolution-sim-backups/archives/20260612T161155Z-v187-v186-delta-blocker-review.tar.zst`
  with archive SHA256
  `c4705d5a53a5f6e5b47dd2125f4cd92d54fc5afb0cf5911044b05fc3f9dab01f`. Do not
  consume slice 3, relax gates, integrate runtime behavior, or promote from
  v187 without a separate explicit task. v188 then found one selected legal
  terminal-survival support trajectory for seed `13` but rejected aggregate
  attempted continuations as support because they had `285` unsupported resolved
  actions. v189 audited that support and blocked slice 3 on `1/6` clean legal
  seed coverage plus dominant requested-action share `0.539244` above the
  `0.50` cap. v190 then ran targeted legal support expansion over seeds
  `13,19,29,37,41,43`, `18` branch points, and `90` replay-verified
  continuations; it found `0/6` clean legal support under the cap, wrote report
  digest `b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`,
  and recommended
  `v191_targeted_legal_support_repair_or_architecture_review_no_training`. The
  v190 report and trajectory evidence are durable after
  `gdrive:evolution-sim-backups/archives/20260613T100008Z-v190-targeted-legal-terminal-survival-support-expansion.tar.zst`
  with archive SHA256
  `e2c1392b066342a43c78274646bfce6589adacc5a61ec3e296f574fed1e7d333` and
  `rclone check` verification of `0` differences and `2` matching files.
  v191 has now diagnosed that blocker without rerunning support expansion or
  training: all `860` v190 unsupported resolved actions were observation-valid
  movement requests that became invalid under the live resolution mask and
  resolved to `stay`, classifying the root cause as same-tick action-mask timing
  / movement-occupancy races. v191 wrote report digest
  `eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5`, kept
  training/runtime/promotion/support-generation flags closed, and routed to
  `v192_action_resolution_contract_repair_no_training`. The v191 report is
  durable after
  `gdrive:evolution-sim-backups/archives/20260613T122723Z-v191-legal-support-repair-architecture-review.tar.zst`
  with archive SHA256
  `bf0184dc2caa6be860723a4b0d6daa35b43264560117dc0794b45c8ce45fdff3` and
  `rclone check` verification of `0` differences and `2` matching files.
  v192 then validated that v191 route and selected exactly a
  support-evidence contract repair, not a runtime/replay behavior change. It
  serialized movement target/blocker audit fields for all `860` unsupported
  resolved actions and classified every one as
  `resolution_invalid_same_tick_occupancy_race`, with no bounds, water, hazard,
  depleted-resource, or stale/illegal-script blockers. v192 wrote report digest
  `17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`, kept
  training/runtime/promotion/support-generation flags closed, and routed to
  `v193_fresh_targeted_legal_support_expansion_after_action_resolution_contract_repair_no_training`.
  The v192 report is durable after
  `gdrive:evolution-sim-backups/archives/20260613T151045Z-v192-action-resolution-contract-repair.tar.zst`
  with archive SHA256
  `e5210b816dddfb6e54ad48fadad417f057fa7de93af6981acbec46991f265fac` and
  `rclone check` verification of `0` differences and `1` matching file.
  v193 then reran targeted legal terminal-survival support expansion under that
  repaired contract. It validated v192 digest
  `17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`, ran `18`
  branch points and `90` replay-verified continuations for seeds
  `13,19,29,37,41,43`, and found `6/6` repaired-contract support under the
  action-share cap. Unsupported requested actions stayed `0`; expected
  same-tick occupancy drift was `860`; unexpected resolution-invalid events
  were `0`; selected-support dominant requested-action share was `0.207558`.
  v193 wrote report digest
  `006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21`, kept
  training/runtime/promotion/slice-3 flags closed, and routed to
  `v194_repaired_contract_terminal_survival_support_dataset_audit_before_slice_3_training_no_training`.
  The v193 report and trajectory evidence are durable after
  `gdrive:evolution-sim-backups/archives/20260613T165955Z-v193-fresh-targeted-legal-support-expansion-after-contract-repair.tar.zst`
  with archive SHA256
  `ceaeb7c2027551393adb1d1374d7f5fd5621aadc3374275992a28b002da0b964` and
  `rclone check` verification of `0` differences and `1` matching file.
  v194 then audited that repaired-contract support as diagnostics-only work
  before any slice-3 training. It validated the v193 digest pin, route, backup
  metadata, `6/6` repaired-contract support, selected trajectory files,
  leakage/action-mask/repaired-resolution/observation/target checks, and wrote
  report digest
  `ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e`.
  It also wrote a gitignored compact support dataset with `3599` rows and
  digest
  `dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc`.
  v194 kept training/runtime/promotion/support-generation flags closed and
  recommended exactly
  `v195_repaired_contract_terminal_survival_support_training_slice_3_opt_in`.
  The v194 report and dataset are durable after
  `gdrive:evolution-sim-backups/archives/20260613T175848Z-v194-repaired-contract-terminal-survival-support-dataset-audit.tar.zst`
  with archive SHA256
  `461295919b0ed81c4d7dc2a4a7ffcb4ceab82c5e4779256205419180129d748e` and
  `rclone check` verification of `0` differences and `1` matching file.
  v195 then consumed the explicit opt-in repaired-contract terminal-survival
  support training slice 3 after validating the v194 report digest
  `ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e`, compact
  dataset digest
  `dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc`, route
  `v195_repaired_contract_terminal_survival_support_training_slice_3_opt_in`,
  v194 backup metadata, v193/v192 digest pins, closed v194 lifecycle flags,
  `3599` rows, leakage-free trainable payloads, unsupported requested actions
  `0`, selected same-tick occupancy drift `77`, and unexpected
  resolution-invalid events `0`. It wrote report digest
  `0d69540a3817b7b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde` and
  artifact digest
  `9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470`.
  Shadow acceptance failed with `0` `carrion_only@120` terminal survivors,
  dominant requested-action share `0.9422`, heuristic action-source count `0`,
  and broad alive/birth regressions on all broad seeds. Runtime integration,
  runtime action selection, promotion, and gate relaxation stayed closed. The
  v195 report and artifact are durable after
  `gdrive:evolution-sim-backups/archives/20260613T185630Z-v195-repaired-contract-terminal-survival-support-training-slice-3.tar.zst`
  with archive SHA256
  `f4d3785619e3937949654cd586d1896fdc20d89fa3d9f3f14d191748ca776049` and
  `rclone check` verification of `0` differences and `1` matching file. The
  campaign budget is now 3/10, and the next same-lane route is
  `v196_repaired_contract_slice_3_failure_response_no_training`, not runtime
  integration, promotion, gate relaxation, or slice-4 training.
  v196 then ran diagnostics-only failure response for that slice-3 failure. It
  validated the pinned v195 report digest
  `0d69540a3817b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde`, artifact
  digest `9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470`,
  v194 report and dataset digests, the required v195 failure-response route,
  v195 backup metadata, slice `3/10` consumption, and exact failure facts. The
  action-collapse mechanism is
  `artifact_low_specificity_feature_coverage_collapse_to_eat_with_miss_abstention`:
  training labels were not dominated (`stay` share `0.207558`), but the frozen
  artifact's diagnostic replay applied overrides on share `0.782944` of
  decisions, the applied override action was `eat` at share `0.939312`,
  mask-only feature-hit share was `0.778513`, exact feature-hit share was
  `0.0`, and misses abstained rather than defaulting to `stay`. v186 had the
  same zero carrion-survivor outcome but not the v195 action collapse or broad
  regression profile. v196 wrote report digest
  `7b3707b60575a14aee806b3989615582369f021f1ceced80e664e4fdd1794d90`, trained
  nothing, consumed no slice 4, changed no runtime behavior, relaxed no gates,
  and authorized no promotion. The v196 report is durable after
  `gdrive:evolution-sim-backups/archives/20260613T213246Z-v196-repaired-contract-slice-3-failure-response.tar.zst`
  with archive SHA256
  `363b4ddb144b8966095c9a1e66f259c418b2a9f4f0c474449fa60d5f06baa45b` and
  `rclone check` verification of `0` differences and `1` matching file. The
  next same-lane route is
  `v197_slice_4_design_requires_coverage_or_abstention_repair_no_training`; the
  campaign budget remains 3/10.
  v197 then completed that no-training coverage/abstention repair. It
  validated the pinned v196 digest
  `7b3707b60575a14aee806b3989615582369f021f1ceced80e664e4fdd1794d90`, route
  `v197_slice_4_design_requires_coverage_or_abstention_repair_no_training`,
  mechanism
  `artifact_low_specificity_feature_coverage_collapse_to_eat_with_miss_abstention`,
  and the v195 report/artifact failure pins. The new opt-in source-key
  specificity gate left default runtime action selection unchanged. With the
  gate enabled for diagnostic replay, override-applied share fell from
  `0.782944` to `0.0`, `7681` low-specificity decisions were rejected,
  heuristic action-source count stayed `0`, broad alive/birth regressions were
  empty, and dominant requested action `eat` stayed under the cap at share
  `0.403329`; exact/high-specificity hit share remained `0.0` and
  `carrion_only@120` terminal survivors remained `0`. Future slice-4 routing
  from v197 requires real replay provenance; mock or override diagnostics fail
  closed to coverage or model-capacity repair. v197 wrote report digest
  `8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a`,
  trained nothing, consumed no slice 4, changed no runtime behavior, relaxed no
  gates, and authorized no promotion. The v197 report is durable after
  `gdrive:evolution-sim-backups/archives/20260614T180236Z-v197-coverage-abstention-repair-design.tar.zst`
  with archive SHA256
  `f11dd5c9cd3eea3f30c2854320a435035277b74b4792f008cd687235ad2b2b40` and
  `rclone check` verification of `0` differences and `1` matching file. The
  next route is
  `v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training`;
  the campaign budget remains 3/10.
  v198 then completed that no-training high-specificity coverage/model-capacity
  repair. It validated the pinned v197 digest
  `8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a`,
  required route
  `v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training`,
  inherited v196/v195 pins, real replay provenance, and closed lifecycle flags.
  The frozen v195 artifact has `7457` high-specificity keys, but real broad plus
  `carrion_only@120` replay with transition-value action override disabled found
  high-specificity key presence in only `430/13634` decisions, complete
  high-specificity current-valid-action coverage `0.0`, observed-support-floor
  pass share `0.0`, and selected high-specificity source share `0.0`. All `694`
  present high-specificity key/category occurrences were
  present-but-action-incomplete. Low-specificity collapse remained blocked with
  `7681` low-specificity would-be rejections, heuristic action-source count
  `0`, no broad/carrion alive-birth regressions, dominant requested action `eat`
  at share `0.403403`, and `carrion_only@120` terminal survivors still `0`.
  v198 wrote report digest
  `91d4132c07ea18b0603626d4c080bdfed01f39677594737cda3ca0233fd95af8`,
  trained nothing, consumed no slice 4, changed no runtime behavior, relaxed no
  gates, generated or expanded no support, and authorized no promotion. The
  v198 report is durable after
  `gdrive:evolution-sim-backups/archives/20260614T191500Z-v198-high-specificity-coverage-or-model-capacity-repair.tar.zst`
  with archive SHA256
  `6d9627c305e56d9083861123e4114635c30c1ae2d939d499026fa73d8f63dbcf` and
  `rclone check` verification of `0` differences and `1` matching file. The
  next route is
  `v199_high_specificity_action_complete_contract_audit_no_training`; the
  campaign budget remains 3/10.
  v199 then completed the no-training high-specificity action-complete contract
  audit. It validated the pinned v198 digest
  `91d4132c07ea18b0603626d4c080bdfed01f39677594737cda3ca0233fd95af8`,
  classification
  `m3_carrion_survivor_continuation_v198_high_specificity_action_complete_or_support_floor_gap_routes_to_contract_audit_no_training`,
  route `v199_high_specificity_action_complete_contract_audit_no_training`,
  primary blocker `high_specificity_action_incomplete`, archive pin, and closed
  lifecycle flags. It reran only the required candidate-key diagnostics with
  transition-value action override disabled. The audit evaluated `54536`
  high-specificity candidate key/category occurrences: `53842` absent, `694`
  present, and all `694` present occurrences were present-but-action-incomplete.
  Present-complete observed-support-floor failures were `0`; missing
  current-valid action counts across present high-specificity keys were
  `eat=628`, `move_west=475`, `move_south=469`, `move_east=339`, `stay=333`,
  `move_north=245`, `attack_west=80`, `attack_east=76`, `drink=65`, and
  `attack_north=4`. Heuristic action-source count was `0`, runtime
  action-selection changes were `0`, and support generation/expansion stayed
  closed. v199 wrote report digest
  `d1111534a01d8cd36d6a8f309b6b05eb28f829c382e2cba6da1f58e011b4d04c`,
  trained nothing, consumed no slice 4, changed no runtime behavior, relaxed no
  gates, generated or expanded no support, and authorized no promotion. The
  v199 report is durable after
  `gdrive:evolution-sim-backups/archives/20260614T195931Z-v199-high-specificity-action-complete-contract-audit.tar.zst`
  with archive SHA256
  `38704355bb89da1a9a12c73284baeaa3f6a9f4984ea7d101bf8d249b92e58bdb` and
  `rclone check` verification of `0` differences and `1` matching file. The
  next route is
  `v200_high_specificity_action_complete_contract_repair_design_no_training`;
  the campaign budget remains 3/10.
  CI/ML
  reproducibility now has compact push coverage plus an explicit NVIDIA-trainer
  validation path for the optional torch stack.

## Confirmed Serious Findings

1. At audit intake, `habitat_state_codes` encoded water tiles as habitat code
   `0` (`stable`) while sibling surfaces used the `-1` non-land sentinel. Counts
   excluded water, so the per-cell matrix and `habitat_state_counts` disagreed,
   and the viewer could label water as stable habitat. This was a replay/viewer
   contract bug, not a taxonomy or simulation-evolution bug.
2. At audit intake, the v137-v176 Mind v3 chain existed only in the dirty
   working tree while the evidence artifacts behind hardcoded digests were
   gitignored. That source backlog is now committed and pushed, and the artifact
   tree is backed up as recorded in the P0 durability inventory.
3. At audit intake, the uncommitted backlog blocked the prescribed
   remote-trainer path because the trainer workflow pulls committed GitHub
   source. The v137-v176 source blocker is resolved; future trainer work still
   requires pushed source and durable artifact provenance.
4. Mind v3 strategy has drifted into small 1-nearest-neighbor and micro-archive
   diagnostics after the lane itself showed support starvation, feature
   aliasing, and tie collapse. The central carrion survival blocker has not
   moved materially.
5. At audit intake, `mind/` library modules imported private helpers from the
   CLI `mind_v3_evaluate.py`. The harness now lives under
   `python/evolution_sim/mind/`, and `mind_v3_evaluate.py` is back to a thin
   parser/orchestrator with compatibility imports.

## Immediate Operating Rules

- Do not start a new Mind v3 experiment slice while that slice depends on
  dirty-tree source, uncommitted package entrypoints/tests/docs, or
  local-only digest-referenced artifacts.
- Do not create another scalar tuning, actor-bias, residual-threshold, or
  tiny-nearest-neighbor support probe on the current representation unless it is
  explicitly labeled as a negative control.
- Do not claim broad held-out promotion evidence from seeds
  `5,13,19,29,37,41` for any scorer trained or selected using artifacts that
  consumed those seeds as support/provenance. Mint a fresh promotion-heldout
  matrix before promotion-style claims.
- Do not hardcode a digest to a gitignored artifact unless the artifact has a
  durable backup path and the report records where to retrieve it.
- Do not relax action-collapse, heuristic-action, or per-seed gates to pass a
  candidate.
- The habitat water encoding fix was treated as an explicit replay/viewer
  contract change with tests, viewer handling, contract notes, and regenerated
  replay goldens.

## Action Order

### P0: Source And Evidence Durability

Status: complete for the audit backlog. The dirty tree was inventoried, the
source backlog was committed and pushed, and `output/mind/` was copied locally
and uploaded to the documented backup target. Future experiment slices should
use the clean long-term shape: CLI wrapper, `mind/` implementation, tests,
`package.json` script, and ledger entry together in a reviewable unit.

Back up any new `output/mind/` artifacts referenced by hardcoded digests before
relying on them from another machine. The backup location must not expose
private host paths or secrets, but reports should record enough provenance to
fetch the artifact again.

No commit, push, reset, clean, or destructive git command should be run without
explicit user authorization.

### P1: Strategy Reset

Status: v179 implemented and locally validated with pinned local `output/mind/`
artifacts, and v180 has now consumed the durable pinned v178 authorization for
the first explicit opt-in transition-row policy training slice. The v176
recommendation,
`v177_exact_branch_replay_expansion_no_training`, now materializes exact branch
replay evidence and compact transition rows with `next_public_observation`,
`next_public_action_mask`, and `previous_same_agent_public_context`. The v178
transition-row dataset audit found the v177 dataset source-valid,
schema-valid, key/value leakage-free, nested trainable-feature contract-valid,
and observation-decodable, but it has only `84` rows, `2` source seeds, and
`16` branch points under the default support minimums. The v179 implementation
expands exact branch transition-row support without training, and local
validation materialized `180` rows across `6` seeds, `30` branches, and `7`
forced actions. The pinned v179 local report is
`output/mind/mind-v3-v179-carrion-survivor-continuation-exact-branch-transition-row-expansion.json`
with exact digest
`1e18703d7f0f3b5666968051f8e3865a05ce7d781046aff7e5d5a07732907171`; the
dataset is
`output/mind/mind-v3-v179-carrion-survivor-continuation-compact-transition-rows.jsonl`
with digest
`df9043666639dc9d606e118c5c3733efc0ae9ac1c76a7126cb8d009b1591e9bf`. A pinned
v178 default-threshold audit of that local v179 output passed all contracts,
including the embedded v179 proof that upstream v177 source report and dataset
digests were pinned, and authorized only the first opt-in same-lane
transition-row training slice. Its report is
`output/mind/mind-v3-v178-carrion-survivor-continuation-transition-row-dataset-audit-v179-expanded.json`
with exact digest
`8d10ee77315de87a15ed296d87482d2335008009e4bcce2d72f60399ede92923`. Do not
train from non-durable source/artifacts or another undersupported
nearest-neighbor scorer.

The first same-lane scale/capacity step has run as v180. It wrote
`output/mind/mind-v3-v180-carrion-survivor-continuation-transition-row-policy-training.json`
with exact digest
`ab894238d9f6ed5041b4587fe69ceeddeb36fb6af1aeaffde6339d73a6f8146f`
and trained
`output/mind/mind-v3-v180-carrion-survivor-continuation-transition-row-policy-artifact.json`
with artifact digest
`66f4fd956111bbda031c122643de12ce556b7cdeb35b70f2d0031cc89fac82b0`.
Its classification is
`m3_carrion_survivor_continuation_v180_transition_row_policy_training_first_slice_shadow_acceptance_failed_no_promotion`.
The blocker is exact: `carrion_only@120` had `0` terminal survivors, and broad
seeds `5,13,19,29,37,41` regressed alive/birth versus the linear Mind v3
baseline. Dominant requested-action share stayed below the cap at `0.483`, and
heuristic action sources stayed at `0`. This counts as slice 1 of the 10-slice
carrion campaign budget. Stop for direction before another slice; do not relax
thresholds, promote/runtime-integrate the artifact, or route this failure into
another pre-training audit. The v180 report and artifact are durable only after
the recorded backup:
`/Users/njm/evolution-sim-p0-backups/20260611T110135Z-v180-transition-row-policy-training.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260611T110135Z-v180-transition-row-policy-training.tar.zst`,
archive SHA256
`e9c95f424e0827a7ed25f7de523fc61e9fe38d6062861466a81f1149671aac59`.
Local copy verification matched the original v180 report file SHA256
`296e910de2dc40d8a16d42565280436bdc3e601792064fb0be43514156c4c055`
and artifact file SHA256
`3d9016f2b7ea5e6d1e9c1ef63afdbca5fb2fea96a3ccbadba91089a93a154432`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260611T110135Z-v180-transition-row-policy-training.tar.zst*" --one-way`
reported `0` differences and `2` matching files. The next useful work after
durability was v181 failure-response, not another pre-training audit and not a
rerun of the first training slice.

The v181 failure-response was diagnostics-only. It wrote
`output/mind/mind-v3-v181-carrion-survivor-continuation-v180-failure-response-autopsy.json`
with exact digest
`a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751` and
classification
`m3_carrion_survivor_continuation_v181_v180_failure_response_autopsy_imputed_sparse_eat_override_and_carrion_no_coverage_no_training`.
It validated the pinned v180 report exact digest
`ab894238d9f6ed5041b4587fe69ceeddeb36fb6af1aeaffde6339d73a6f8146f`, v180
artifact digest
`66f4fd956111bbda031c122643de12ce556b7cdeb35b70f2d0031cc89fac82b0`, and v179
dataset digest
`df9043666639dc9d606e118c5c3733efc0ae9ac1c76a7126cb8d009b1591e9bf`.
Lifecycle flags stayed closed: `training_ran=false`,
`training_artifact_created=false`, `runtime_artifact_created=false`,
`runtime_action_selection_changed=false`, and `promotion_authorized=false`.
The diagnostic trace replay found the v180 artifact was `0.7` imputed action
stats; broad regression seeds had `1337` runtime action changes with predicted
action counts `eat:1450`, `drink:15`, and `move_south:17`; and the carrion
fixture had `0` runtime action changes with missing supported-score share
`1.0`. v181 consumed no second training slice, so the campaign budget remains
1/10. The v181 report is durable after
`/Users/njm/evolution-sim-p0-backups/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`,
archive SHA256
`a2b03e862ee095562c21482afcf0b1057a9ebb8f8b0b0d0f08e973dd884b1fd6`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst*" --one-way`
reported `0` differences and `2` matching files.

The v182 failure-response design was diagnostics-only. It wrote
`output/mind/mind-v3-v182-carrion-survivor-continuation-imputed-abstention-design.json`
with exact digest
`3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b` and
classification
`m3_carrion_survivor_continuation_v182_imputed_abstention_design_strict_support_routes_to_exact_transition_support_expansion_no_training`.
It validated the pinned v181 report exact digest
`a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751`, v180
report exact digest
`ab894238d9f6ed5041b4587fe69ceeddeb36fb6af1aeaffde6339d73a6f8146f`, v180
artifact digest
`66f4fd956111bbda031c122643de12ce556b7cdeb35b70f2d0031cc89fac82b0`, and v179
dataset digest
`df9043666639dc9d606e118c5c3733efc0ae9ac1c76a7126cb8d009b1591e9bf`.
Lifecycle flags stayed closed: `training_ran=false`,
`training_artifact_created=false`, `runtime_artifact_created=false`,
`runtime_action_selection_changed=false`, `promotion_authorized=false`, and
`slice_2_training_consumed=false`. The design adds explicit imputed-valid-action
and observed-support-floor diagnostics and makes opt-in transition-value
overrides abstain when any currently valid action score is imputed or below
observed support floor `2`. The shadow evaluation reduced broad overrides to
`122`, but broad seed `19` still regressed alive/birth by `-1/-2`, and
`carrion_only@120` still had `0` terminal survivors with observed support-floor
coverage `0/2122`. v182 consumed no second training slice, so the campaign
budget remained 1/10 at that point; v186 later consumed slice 2/10. The v182
report is durable after
`/Users/njm/evolution-sim-p0-backups/20260611T165440Z-v182-imputed-abstention-design.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260611T165440Z-v182-imputed-abstention-design.tar.zst`,
archive SHA256
`d53401e5096d4c616a26691faa555ba53ffbdf84b4767061ff09d820340faf54`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260611T165440Z-v182-imputed-abstention-design.tar.zst*" --one-way`
reported `0` differences and `2` matching files.

The v183 exact transition-support expansion was diagnostics-only. It wrote
`output/mind/mind-v3-v183-carrion-survivor-continuation-exact-transition-support-expansion.json`
with exact digest
`7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da` and
`output/mind/mind-v3-v183-carrion-survivor-continuation-expanded-compact-transition-rows.jsonl`
with dataset digest
`e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`. Its
classification is
`m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_targeted_exact_support_ready_for_fresh_v178_audit_no_training`.
It validated the pinned v182 report exact digest
`3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b`, v181
report exact digest
`a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751`, v180
report exact digest
`ab894238d9f6ed5041b4587fe69ceeddeb36fb6af1aeaffde6339d73a6f8146f`, v180
artifact digest
`66f4fd956111bbda031c122643de12ce556b7cdeb35b70f2d0031cc89fac82b0`, v179
report exact digest
`1e18703d7f0f3b5666968051f8e3865a05ce7d781046aff7e5d5a07732907171`, and v179
dataset digest
`df9043666639dc9d606e118c5c3733efc0ae9ac1c76a7126cb8d009b1591e9bf`.
Lifecycle flags stayed closed: `training_ran=false`,
`training_artifact_created=false`, `runtime_artifact_created=false`,
`runtime_action_selection_changed=false`, `promotion_authorized=false`, and
`slice_2_training_consumed=false`. The expanded dataset has `144` rows, `24`
branches, `6` seeds, `9` forced actions, and passes the default support
summary. It moved carrion observed-support target coverage from `0` to `18`
materialized states and broad seed `19` support-hole coverage to `6`
materialized states. The v183 report, dataset, and source trajectories are
durable after
`/Users/njm/evolution-sim-p0-backups/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`,
archive SHA256
`b4e57537142303e622656c8eb51fde114a9452e20641a85cfdb4d742b425eb5c`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260611T192709Z-v183-exact-transition-support-expansion.tar.zst*" --one-way`
reported `0` differences and `2` matching files. At that point the next useful
lane was a fresh v178-style audit of the v183 expanded dataset before any
slice-2 training; v184, v185, and v186 have since run.

The v184 fresh audit has now run as diagnostics-only work. It wrote
`output/mind/mind-v3-v184-carrion-survivor-continuation-v183-transition-row-dataset-audit.json`
with exact digest
`ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa` and
classification
`m3_carrion_survivor_continuation_v184_v183_transition_row_dataset_audit_dataset_contract_invalid_closed_no_training`.
The canonical v183 source report and dataset digests matched, source
validation passed, and the v183 pinned v182/v181/v180/v179 evidence was valid.
The blocker is the target audit: `14` failures across `7` attack-forced rows
resolved to `stay` with `current_resolution_action_valid=false`. Lifecycle
flags stayed closed: `training_ran=false`, `training_artifact_created=false`,
`runtime_artifact_created=false`, `runtime_action_selection_changed=false`,
`promotion_authorized=false`, and `slice_2_training_consumed=false`. The v184
report is durable after
`/Users/njm/evolution-sim-p0-backups/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`,
archive SHA256
`057d34c88f8a5da0c4ad98d87e4451caa043b7ee53c2af1037aa231dc5bcf382`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst*" --one-way`
reported `0` differences and `2` matching files. The next useful lane is
`v186_transition_row_policy_training_slice_2_opt_in` if explicitly requested,
not automatic training.

The v185 target-resolution repair has now run as diagnostics-only work. It
wrote
`output/mind/mind-v3-v185-carrion-survivor-continuation-v183-target-resolution-repair.json`
with exact digest
`327586651948e13670cf15285472a934c8cd0a8fc9784d2dcdb092d102c46486` and wrote
`output/mind/mind-v3-v185-carrion-survivor-continuation-v183-target-resolution-repaired-compact-transition-rows.jsonl`
with dataset digest
`532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`.
It removed the `7` invalid target-resolution rows, leaving `137` rows while
preserving the v178 default support counts (`24` branches, `6` seeds, `9`
forced actions). The paired repaired-dataset audit wrote
`output/mind/mind-v3-v185-carrion-survivor-continuation-repaired-transition-row-dataset-audit.json`
with exact digest
`abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`. The audit
passed target validation with `0` failures and route
`v186_transition_row_policy_training_slice_2_opt_in`, while lifecycle flags
stayed closed: `training_ran=false`, `training_artifact_created=false`,
`runtime_artifact_created=false`, `runtime_action_selection_changed=false`,
`promotion_authorized=false`, and `slice_2_training_consumed=false`. The v185
artifacts are durable after
`/Users/njm/evolution-sim-p0-backups/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`,
archive SHA256
`c64aa17964462a4bbbc71fc83b98779213c50e26b714aedffe8cb7b42b9b5d2a`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260612T131407Z-v185-v183-target-resolution-repair.tar.zst*" --one-way`
reported `0` differences and `2` matching files.

The v186 explicit opt-in transition-row policy training slice 2 has now run
from the repaired v185 dataset. Command:
`npm run sim:mind:v3:carrion-survivor-continuation-v186-transition-row-policy-training`.
It validated the repaired audit exact digest
`abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`, repaired
dataset digest
`532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`, source
producer `v185_v183_target_resolution_repair`, and route
`v186_transition_row_policy_training_slice_2_opt_in`. It wrote
`output/mind/mind-v3-v186-carrion-survivor-continuation-transition-row-policy-training.json`
with exact digest
`f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075` and
`output/mind/mind-v3-v186-carrion-survivor-continuation-transition-row-policy-artifact.json`
with artifact digest
`729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce`.
It consumed slice 2 (`training_ran=true`, `training_artifact_created=true`,
`slice_2_training_consumed=true`) and failed shadow acceptance:
`carrion_only@120` terminal survivors stayed `0`. Dominant requested-action
share stayed under the cap at `0.4179`, heuristic action-source count stayed
`0`, and there were no broad per-seed alive/birth regressions. Runtime and
promotion flags stayed closed: `runtime_artifact_created=false`,
`runtime_action_selection_changed=false`, and `promotion_authorized=false`.
The v186 report and artifact are durable after
`/Users/njm/evolution-sim-p0-backups/20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst`,
archive SHA256
`9fba7700ec13b23f70f31b0269022c19b74bb7ed2967d2d70a9add05919eb31a`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst*" --one-way`
reported `0` differences and `2` matching files. The campaign budget is now
2/10. Do not consume slice 3, relax acceptance gates, integrate runtime
behavior, or promote from v186 without a separate explicit task.

The v187 diagnostics-only v186-delta blocker review has now run. Command:
`npm run sim:mind:v3:carrion-survivor-continuation-v187-v186-delta-blocker-review`.
It validated the v186 report exact digest
`f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075` and v186
artifact digest
`729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce`. It
explicitly marked v31, v36, v90/v91, and v181/v182 conclusions as inherited
prior findings, not rediscovered by v187. The new v186 delta was: broad
alive/birth regressions empty, dominant requested-action share `0.4179`,
heuristic action-source count `0`, `carrion_only@120` terminal survivors still
`0`, and carrion fixture births mean `2.8333`. It wrote
`output/mind/mind-v3-v187-carrion-survivor-continuation-v186-delta-blocker-review.json`
with exact digest
`4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6`.
Classification:
`m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_terminal_survival_support_generation_before_slice_3_no_training`.
Recommended next route:
`v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`.
Lifecycle flags stayed closed: `training_ran=false`,
`training_artifact_created=false`, `slice_3_training_consumed=false`,
`runtime_artifact_created=false`, `runtime_action_selection_changed=false`,
`promotion_authorized=false`, `gate_relaxation_allowed=false`,
`support_expansion_ran=false`, `v180_rerun=false`, and `v186_rerun=false`.
The v187 report is durable after
`/Users/njm/evolution-sim-p0-backups/20260612T161155Z-v187-v186-delta-blocker-review.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260612T161155Z-v187-v186-delta-blocker-review.tar.zst`,
archive SHA256
`c4705d5a53a5f6e5b47dd2125f4cd92d54fc5afb0cf5911044b05fc3f9dab01f`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260612T161155Z-v187-v186-delta-blocker-review.tar.zst*" --one-way`
reported `0` differences and `2` matching files. The campaign budget remains
2/10. v187 is not promotion evidence and authorizes no runtime integration,
gate relaxation, support expansion, or slice-3 training without a separate
explicit task.

The v188 diagnostics/support evidence run has now completed. Command:
`npm run sim:mind:v3:carrion-survivor-continuation-v188-terminal-carrion-survival-support`.
It validated the v187 report exact digest
`4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6` and route
`v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`
before searching `carrion_only@120` seeds `13,19,29,37,41,43`. The search used
`6` branch points, `30` bounded policy-visible script continuations, and
replay verification. Aggregate attempted continuations had terminal survivors
by seed `{13: 3, 19: 3, 29: 4, 37: 3, 41: 5, 43: 4}` but included `285`
unsupported resolved actions, so those aggregate attempts are not support
evidence. The selected support evidence is one legal deterministic
replay-verified trajectory: seed `13`, branch
`carrion-only-seed-13-branch-0-tick-0-agent-9`, continuation
`conserve_after_carrion`, terminal alive `1`, births `5`, unsupported requested
actions `0`, and unsupported resolved actions `0`. It wrote
`output/mind/mind-v3-v188-carrion-survivor-continuation-terminal-survival-support.json`
with exact digest
`9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0` and
trajectory evidence under
`output/mind/v188-terminal-carrion-survival-support-trajectories/`.
Classification:
`m3_carrion_survivor_continuation_v188_terminal_carrion_survival_support_positive_legal_branch_support_found_no_training`.
Recommended next route:
`v189_terminal_survival_support_dataset_audit_before_slice_3_training`.
Lifecycle flags stayed closed: `training_ran=false`,
`training_artifact_created=false`, `slice_3_training_consumed=false`,
`runtime_artifact_created=false`, `runtime_action_selection_changed=false`,
`promotion_authorized=false`, and `gate_relaxation_allowed=false`. The v188
report and trajectory evidence are durable after
`/Users/njm/evolution-sim-p0-backups/20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst`,
archive SHA256
`67856f7b67c9156ad64c7da66ce5f118b03c1132d30c83b00c097ea0dedf7747`;
`rclone check /Users/njm/evolution-sim-p0-backups
gdrive:evolution-sim-backups/archives --include
"20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst*" --one-way`
reported `0` differences and `2` matching files. The campaign budget remains
2/10. v188 is not promotion evidence and authorizes no runtime integration,
gate relaxation, or slice-3 training without a separate explicit v189 audit
task.

### P1: Documentation And Agent Instructions

Keep `AGENTS.md`, `README.md`, onboarding, versioning, and local skills aligned
with this audit. Agent loops must include a durability checkpoint and must not
encode stale strategy anchors as the default path.

### P2: Habitat Water Contract Fix

Status: complete on the dedicated habitat-water contract branch. Water cells now
serialize as the non-land sentinel in `habitat_state_codes`; viewer labels and
colors handle that sentinel; focused regressions cover the land-count recount;
generated viewer contracts and replay goldens were refreshed.

### P2: Mind v3 Shared Harness Extraction

Status: complete. Pure report/metric helpers, including JSON canonicalization,
safe path fragments, rounding, counter conversion, dominant action summaries,
shares, means, and comparison deltas, live in
`python/evolution_sim/mind/evaluation_helpers.py`. Fixture construction, report
aggregation, strict gate helpers, digest validation, and leakage scans live in
`python/evolution_sim/mind/evaluation_harness.py`.

`python/evolution_sim/cli/mind_v3_evaluate.py` now defines only parser/main
orchestration and keeps compatibility imports for historical callers. Mind
modules and sibling live A/B CLIs import shared helpers from the Mind layer
rather than private CLI helpers. A focused boundary regression rejects new
production `mind/` imports from `evolution_sim.cli`.

### P2: CI And ML Reproducibility

Status: complete for audit remediation. Push-time CI now runs a compact 20-tick
Mind gate with default non-vacuous criteria and temporary outputs. Keep strict
Mind gates on scheduled/manual lanes. The optional torch stack now has an
explicit non-CI validation path:
`npm run sim:mind:torch:validate` for any local ML environment and
`npm run trainer -- run npm run sim:mind:torch:validate:cuda` for the configured
NVIDIA trainer. The CUDA path requires torch CUDA visibility, runs the
torch-gated Mind tests, executes a tiny CUDA-backed training smoke, and records
dependency versions plus device metadata in
`output/mind/mind-torch-validation-report.json`.

### P3: Orientation Cleanup

Refresh old onboarding claims, remove stale exact private-call count tables
unless they are regenerated mechanically, and keep the Mind v3 ledger pointing
to the current next action rather than the superseded v61-v63 scalar-family
boundary.

## Prompt For The Next Coder

Use this prompt when dispatching the next implementation agent:

```text
You are working in /Users/njm/Projects/evolution-sim after the 2026-06-10 full
repository audit. Start by reading AGENTS.md,
docs/repository-audit-2026-06-10-remediation.md, and the latest section of
docs/mind-v3-autonomous-evolution.md. Check git status before edits and treat
all existing dirty-tree changes as user-owned.

Do not assume access to any prior chat, pasted strategic ledger, or thread
attachment. If the dispatcher says a ledger is source of truth, they must paste
the relevant contents into this prompt or point you to a committed repo summary.
For this handoff, AGENTS.md and the 2026-06-10 checkpoint section of
docs/mind-v3-autonomous-evolution.md are the durable summary of that direction.

Do not start a new Mind v3 experiment or train anything in a handoff that only
asks for v180 durability. v180 has already run and failed as slice 1/10 after
consuming the durable default-threshold v178 authorization. Caller-lowered
`--min-*` thresholds are diagnostic only and must not authorize training. The
v137-v176 durability backlog is already resolved; before any new experiment or
trainer work, verify the current slice's source, package entrypoints, tests,
ledger entries, and any digest-referenced `output/mind/` artifacts are durable.
If durability depends on git or artifact-backup actions that are not authorized,
stop after producing the exact commit/artifact plan.

If source durability, the habitat water contract fix, shared-harness
extraction, CI/ML reproducibility hardening, v177 exact-branch-replay
transition-row support, the v178 transition-row dataset audit, v179
exact-branch transition-row support expansion, and the v180 report/artifact
backup are already durable, v181 failure-response and v182 imputed-abstention
design plus v183 exact transition-support expansion should also be checked
before new work. v184, v185, v186, and v187 have now run after that prompt. Do
not rerun the first opt-in transition-row training slice, route the failure into
another support expansion by default, rerun v186, rerun v187, or consume slice 3
unless the user explicitly asks for a new task after a durability check. Keep CI
slices compact and do not turn push CI into promotion evidence.

Keep strict Mind v3 gates hard. Do not create another scalar-tuning,
actor-bias, residual-threshold, or tiny nearest-neighbor micro-archive probe.
Promotion-style Mind v3 claims need fresh held-out seeds because
5,13,19,29,37,41 have been consumed as support/provenance for recent lanes.
```
