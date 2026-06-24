---
name: mind-v3-autonomous-loop
description: "Use for Mind v3 autonomous-controller work in evolution-sim: no-heuristic policy experiments, rollout-context policy capacity, carrion recovery, strict labeled IQL slices, sequence/flow/world-model branches, exact branch replay, controlled fixture gates, and Mind v3 experiment documentation."
---

# Mind v3 Autonomous Loop

Use this skill for current Mind v3 autonomous-controller work.

## Current Boundary

Mind v3 is the strategic track for reducing heuristic action selection. The
Foundation, replay/viewer contracts, trajectory data contract, and guarded Mind
v1/v2 gates are the safety floor.

As of the v61-v63 closeout, the current single-observation IQL
coefficient/prior/extraction family is exhausted for promotion. The 2026-06-10
full-repository audit also closes the recent tiny support-archive /
nearest-neighbor scorer loop as strategically saturated. Do not start a new
slice by tuning scalar IQL weights, prior blends, action-distribution loss, risk
extraction, global actor bias, residual thresholds, or another micro-archive
nearest-neighbor scorer unless the user explicitly asks for a negative-control
probe.

Current durability constraint: the v137-v176 experiment chain has been
committed, pushed, and backed up, but the same failure mode must not recur.
Because the RTX trainer pulls committed source, do not launch new Mind v3
experiments or trainer jobs while the current slice's source, package
entrypoints, tests, ledger entries, or digest-referenced artifacts are still
local-only. A docs-only remediation pass or explicit git/backlog triage is
allowed.

Strategic handoff constraint: thread-local ledgers and attachments are not
durable context. A fresh coder should not be expected to know them unless the
dispatch prompt includes the relevant content or points to a committed summary
in `AGENTS.md` or `docs/mind-v3-autonomous-evolution.md`.

Historical evidence constraint: the version chain is an evidence bank, not
dead history. Before proposing a new same-lane Mind v3 action, mine the durable
ledger, source constants, tests, and pinned reports for reusable constraints:
what was already tried, what failed, what was ruled out, which seeds/artifacts
were consumed as support, which reports authorized the current route, and which
mechanisms remain active. Convert that evidence into explicit stop rules and
test expectations. Do not rediscover closed lanes as "fresh diagnostics" unless
the user explicitly asks for a negative control.

## Inspect First

Read the smallest relevant set:

- `AGENTS.md`
- `README.md`
- `docs/repository-audit-2026-06-10-remediation.md`
- `docs/mind-v3-autonomous-evolution.md`, especially the latest milestone
  sections and research direction
- `python/evolution_sim/cli/mind_v3_labeled_iql_slice.py`
- `python/evolution_sim/cli/mind_gate.py`
- `python/evolution_sim/mind/torch_trainer.py`
- `python/evolution_sim/mind/v3_policy.py`
- `python/evolution_sim/mind/v3_neural.py`
- `python/evolution_sim/mind/dataset.py`
- `python/evolution_sim/env/runtime/trajectory.py`
- relevant tests in `python/tests/`

Check `git status --short` before edits. Existing unrelated changes are
user-owned.

## Allowed Novelty

Novel controller ideas are welcome, but they must pass through repo contracts.
Current same-lane work is narrower:

- v180 has already consumed the durable pinned v178 authorization and run the
  first opt-in transition-row policy training slice;
- v180 failed shadow acceptance with zero `carrion_only@120` terminal survivors
  and broad per-seed alive/birth regressions, so it is not promotion evidence
  and authorizes no runtime integration, runtime action-selection change, gate
  relaxation, or promotion;
- v180 counts as slice 1 of the 10-slice carrion campaign budget. Its report
  and policy artifact are durable only after the recorded backup
  `gdrive:evolution-sim-backups/archives/20260611T110135Z-v180-transition-row-policy-training.tar.zst`
  with archive SHA256
  `e9c95f424e0827a7ed25f7de523fc61e9fe38d6062861466a81f1149671aac59`;
- v181 has already run as diagnostics-only failure-response work. It validated
  the pinned v180 report, artifact, and dataset, diagnosed sparse imputed
  transition-value support plus broad `eat` overrides and carrion no-coverage,
  and consumed no second training slice. Its report exact digest is
  `a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751` and is
  durable after
  `gdrive:evolution-sim-backups/archives/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`
  with archive SHA256
  `a2b03e862ee095562c21482afcf0b1057a9ebb8f8b0b0d0f08e973dd884b1fd6`;
- v182 has already run as diagnostics-only failure-response design. It added
  imputed-valid-action and observed-support-floor diagnostics, made opt-in
  transition-value overrides abstain on imputed or below-floor currently valid
  action scores, and consumed no second training slice. Its report exact digest
  is `3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b` and is
  durable after
  `gdrive:evolution-sim-backups/archives/20260611T165440Z-v182-imputed-abstention-design.tar.zst`
  with archive SHA256
  `d53401e5096d4c616a26691faa555ba53ffbdf84b4767061ff09d820340faf54`;
- v183 has already run as diagnostics-only exact transition-support expansion.
  It validated the pinned v182/v181/v180/v179 evidence, wrote report digest
  `7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, wrote a
  `144`-row expanded dataset with digest
  `e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and
  kept training/runtime/promotion/slice-2 lifecycle flags closed. It moved
  carrion target coverage from `0` to `18` materialized states and broad seed
  `19` support-hole coverage to `6` materialized states. Its report, dataset,
  and source trajectories are durable after
  `gdrive:evolution-sim-backups/archives/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`
  with archive SHA256
  `b4e57537142303e622656c8eb51fde114a9452e20641a85cfdb4d742b425eb5c`;
- v184 has already run as a diagnostics-only fresh v178-style audit of the
  canonical v183 expanded dataset. It validated the v183 source report digest
  `7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, the
  v183 dataset digest
  `e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and the
  pinned v182/v181/v180/v179 evidence, but target audit failed with `14`
  failures across `7` attack-forced rows resolving to `stay` with
  `current_resolution_action_valid=false`. Its report exact digest is
  `ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`
  with archive SHA256
  `057d34c88f8a5da0c4ad98d87e4451caa043b7ee53c2af1037aa231dc5bcf382`;
- v185 has already repaired the v183 target-resolution blocker as
  diagnostics-only work. It strict-filtered the `7` invalid rows without
  backfill, leaving `137` rows, `24` branches, `6` seeds, and `9` forced
  actions. The repair report exact digest is
  `327586651948e13670cf15285472a934c8cd0a8fc9784d2dcdb092d102c46486`, the
  repaired dataset digest is
  `532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`, and
  the paired repaired-dataset audit exact digest is
  `abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`. The
  repaired audit passed the v178-style source, schema, leakage, identity,
  action-mask, observation, target, and default-support checks and authorized
  only the future explicit route
  `v186_transition_row_policy_training_slice_2_opt_in`. v185 trained nothing,
  spent no slice 2, created no runtime artifact, changed no runtime action
  selection, and authorized no promotion. Its artifacts are durable after
  `gdrive:evolution-sim-backups/archives/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`
  with archive SHA256
  `c64aa17964462a4bbbc71fc83b98779213c50e26b714aedffe8cb7b42b9b5d2a`;
- v186 has already consumed the explicit
  `v186_transition_row_policy_training_slice_2_opt_in` route from the repaired
  v185 dataset. It validated the repaired audit exact digest
  `abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`,
  repaired dataset digest
  `532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`, source
  producer `v185_v183_target_resolution_repair`, and route
  `v186_transition_row_policy_training_slice_2_opt_in`. It wrote report digest
  `f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075` and
  artifact digest
  `729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce`.
  Shadow acceptance failed with `0` `carrion_only@120` terminal survivors,
  dominant requested-action share `0.4179`, heuristic action-source count `0`,
  and no broad per-seed alive/birth regressions. Runtime and promotion flags
  stayed closed. The v186 report and artifact are durable after
  `gdrive:evolution-sim-backups/archives/20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst`
  with archive SHA256
  `9fba7700ec13b23f70f31b0269022c19b74bb7ed2967d2d70a9add05919eb31a`;
- v187 has already run as a diagnostics-only v186-delta blocker review. It
  validated the pinned v186 report digest
  `f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075` and
  artifact digest
  `729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce`,
  marked v31/v36/v90-v91/v181-v182 findings as inherited rather than
  rediscovered, and extracted only the new v186 delta: no broad alive/birth
  regressions, dominant requested-action share `0.4179`, heuristic
  action-source count `0`, zero `carrion_only@120` terminal survivors, and
  carrion fixture births mean `2.8333`. It wrote report digest
  `4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6`,
  classified as
  `m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_terminal_survival_support_generation_before_slice_3_no_training`,
  and recommended exactly
  `v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`.
  The v187 report is durable after
  `gdrive:evolution-sim-backups/archives/20260612T161155Z-v187-v186-delta-blocker-review.tar.zst`
  with archive SHA256
  `c4705d5a53a5f6e5b47dd2125f4cd92d54fc5afb0cf5911044b05fc3f9dab01f`;
- v188 has already run as diagnostics/support evidence only. It validated the
  pinned v187 exact digest
  `4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6` and route
  `v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`.
  It searched `carrion_only@120` seeds `13,19,29,37,41,43` with `6` branch
  points and `30` bounded policy-visible continuations. Aggregate attempts had
  terminal survivors for all six seeds but included `285` unsupported resolved
  actions, so only the selected legal support trajectory counts as evidence:
  seed `13`, branch `carrion-only-seed-13-branch-0-tick-0-agent-9`,
  continuation `conserve_after_carrion`, terminal alive `1`, births `5`,
  replay verified, unsupported requested actions `0`, and unsupported resolved
  actions `0`. The v188 report digest is
  `9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0`, and its
  report plus trajectory evidence are durable after
  `gdrive:evolution-sim-backups/archives/20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst`
  with archive SHA256
  `67856f7b67c9156ad64c7da66ce5f118b03c1132d30c83b00c097ea0dedf7747` and
  `rclone check` verification of `0` differences and `2` matching files;
- v189 has already run as the requested diagnostics-only terminal-survival
  support dataset audit. It validated v188 exact digest
  `9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0`, audited
  the selected support trajectory as path-backed, replay-valid by report,
  action-mask legal, terminal-fact matched, and trainable-leakage clean, and
  recorded historical dedupe so v32/v51, v53-v63, v154-v176, v181-v182, and
  v183-v186 are not rerun as missing facts. It blocked slice 3 because clean
  legal support coverage is only `1/6` target seeds, the selected dominant
  requested action share is `0.539244` above `0.50`, and aggregate v188
  attempts had `285` unsupported resolved actions. The v189 report digest is
  `2505049a4a81f17ca5fb79f004c9d4ea7f43b1169f9e461e6a722df04e7d369e`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260612T175835Z-v189-terminal-survival-support-dataset-audit.tar.zst`
  with archive SHA256
  `c9a347d1504e2222473f3db2d49f2f0c5afe21c61ff877730c1fefe12045ff92` and
  `rclone check` verification of `0` differences and `2` matching files;
- v190 has already run as diagnostics-only targeted legal terminal-survival
  support expansion. It validated v189 digest
  `2505049a4a81f17ca5fb79f004c9d4ea7f43b1169f9e461e6a722df04e7d369e`, targeted
  seeds `13,19,29,37,41,43`, ran `18` exact branch points and `90`
  replay-verified continuations, but found `0/6` clean legal support under the
  action-share cap; aggregate attempted continuations had `860` unsupported
  resolved actions. The v190 report digest is
  `b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260613T100008Z-v190-targeted-legal-terminal-survival-support-expansion.tar.zst`
  with archive SHA256
  `e2c1392b066342a43c78274646bfce6589adacc5a61ec3e296f574fed1e7d333` and
  `rclone check` verification of `0` differences and `2` matching files;
- at v190 close, the campaign budget was still 2/10. That historical result
  did not authorize slice 3, gate relaxation, runtime behavior, promotion from
  v190, or v180/v186 reruns;
- v191 has now run as diagnostics-only targeted legal-support repair /
  architecture review. It validated the pinned v190 digest
  `b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`, read the
  v190 report and trajectories, and classified all `860` unsupported resolved
  actions as observation-valid movement requests that became invalid under the
  live resolution mask and resolved to `stay`, with legality reason
  `not_in_resolution_action_mask`. Root cause:
  `action_mask_timing_mismatch_same_tick_movement_occupancy_race`. Its report
  digest is
  `eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260613T122723Z-v191-legal-support-repair-architecture-review.tar.zst`
  with archive SHA256
  `bf0184dc2caa6be860723a4b0d6daa35b43264560117dc0794b45c8ce45fdff3` and
  `rclone check` verification of `0` differences and `2` matching files. v191
  generated no support, trained nothing, consumed no slice 3, changed no runtime
  behavior, and recommended exactly
  `v192_action_resolution_contract_repair_no_training`;
- v192 has already run as diagnostics-only action-resolution contract repair.
  It validated the pinned v191 digest
  `eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5` and route
  `v192_action_resolution_contract_repair_no_training`, selected exactly
  `support_evidence_contract_repair`, and changed no runtime/replay semantics,
  runtime action selection, or trajectory schema. It serialized movement
  target/blocker audit fields for all `860` unsupported resolved actions and
  classified all as `resolution_invalid_same_tick_occupancy_race`, with no
  bounds, water, hazard, depleted-resource, or stale/illegal-script blockers.
  The v192 report digest is
  `17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260613T151045Z-v192-action-resolution-contract-repair.tar.zst`
  with archive SHA256
  `e5210b816dddfb6e54ad48fadad417f057fa7de93af6981acbec46991f265fac` and
  `rclone check` verification of `0` differences and `1` matching file. v192
  generated no support, trained nothing, consumed no slice 3, changed no runtime
  behavior, and recommended exactly
  `v193_fresh_targeted_legal_support_expansion_after_action_resolution_contract_repair_no_training`.
- v193 has already run as diagnostics-only fresh targeted legal
  terminal-survival support expansion under the v192 repaired support-evidence
  contract. It validated v192 digest
  `17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`, ran `18`
  branch points and `90` replay-verified continuations for seeds
  `13,19,29,37,41,43`, and found `6/6` repaired-contract support under the
  action-share cap. Unsupported requested actions were `0`; expected same-tick
  occupancy drift was `860`; unexpected resolution-invalid events were `0`;
  selected-support dominant requested-action share was `0.207558`. The v193
  report digest is
  `006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21`; the
  report and trajectories are durable after
  `gdrive:evolution-sim-backups/archives/20260613T165955Z-v193-fresh-targeted-legal-support-expansion-after-contract-repair.tar.zst`
  with archive SHA256
  `ceaeb7c2027551393adb1d1374d7f5fd5621aadc3374275992a28b002da0b964` and
  `rclone check` verification of `0` differences and `1` matching file. v193
  trained nothing, consumed no slice 3, changed no runtime behavior, and
  recommended exactly
  `v194_repaired_contract_terminal_survival_support_dataset_audit_before_slice_3_training_no_training`;
  v194 has now run as diagnostics-only repaired-contract terminal-survival
  support dataset audit before slice-3 training. It validated v193 digest
  `006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21`, route
  `v194_repaired_contract_terminal_survival_support_dataset_audit_before_slice_3_training_no_training`,
  v193 backup metadata, inherited v192/v191/v190 digests, and `6/6`
  repaired-contract support. It wrote report digest
  `ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e` and a
  compact support dataset digest
  `dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc`.
  Unsupported requested actions stayed `0`; expected same-tick occupancy drift
  was counted separately at aggregate `860` and selected `77`; unexpected
  resolution-invalid events were `0`; selected dominant requested-action share
  was `0.207558`. v194 trained nothing, consumed no slice 3, generated no new
  support, changed no runtime behavior, and recommended exactly
  `v195_repaired_contract_terminal_survival_support_training_slice_3_opt_in`.
  The v194 report and dataset are durable after
  `gdrive:evolution-sim-backups/archives/20260613T175848Z-v194-repaired-contract-terminal-survival-support-dataset-audit.tar.zst`
  with archive SHA256
  `461295919b0ed81c4d7dc2a4a7ffcb4ceab82c5e4779256205419180129d748e`;
  v195 has now consumed the explicit opt-in repaired-contract terminal-survival
  support training slice 3. It validated the v194 report digest
  `ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e`,
  compact dataset digest
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
  and broad alive/birth regressions. Runtime integration, runtime action
  selection, promotion, and gate relaxation stayed closed. The v195 report and
  artifact are durable after
  `gdrive:evolution-sim-backups/archives/20260613T185630Z-v195-repaired-contract-terminal-survival-support-training-slice-3.tar.zst`
  with archive SHA256
  `f4d3785619e3937949654cd586d1896fdc20d89fa3d9f3f14d191748ca776049` and
  `rclone check` verification of `0` differences and `1` matching file. v196
  has now run as diagnostics-only slice-3 failure response. It validated the
  v195 report digest
  `0d69540a3817b7b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde`, v195
  artifact digest
  `9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470`,
  v194 report/dataset digests, required route
  `v196_repaired_contract_slice_3_failure_response_no_training`, v195 backup
  metadata, slice-3 consumption, and exact failure facts. It wrote report digest
  `7b3707b60575a14aee806b3989615582369f021f1ceced80e664e4fdd1794d90`.
  The confirmed mechanism was
  `artifact_low_specificity_feature_coverage_collapse_to_eat_with_miss_abstention`:
  exact feature-hit share `0.0`, mask-only feature-hit share `0.778513`,
  override-applied share `0.782944`, applied override `eat` share `0.939312`,
  and misses did not default to `stay`, despite training support dominant action
  share `0.207558`. v196 trained nothing, created no training or runtime
  artifact, consumed no slice 4, changed no runtime behavior, authorized no
  promotion, and relaxed no gate. The v196 report is durable after
  `gdrive:evolution-sim-backups/archives/20260613T213246Z-v196-repaired-contract-slice-3-failure-response.tar.zst`
  with archive SHA256
  `363b4ddb144b8966095c9a1e66f259c418b2a9f4f0c474449fa60d5f06baa45b` and
  `rclone check` verification of `0` differences and `1` matching file. The
  campaign budget remains 3/10. The current same-lane next route is exactly
  `v197_slice_4_design_requires_coverage_or_abstention_repair_no_training`;
- CLI `--min-*` support overrides are diagnostic only and must not authorize
  training routes.

Do not use:

- fixture identity as runtime policy input;
- private `SimulationWorld` reads from policy code;
- hidden heuristic action selection;
- relaxed acceptance gates to make a candidate pass;
- long sweeps as a substitute for compact contract tests.

## Work Loop

1. Define the hypothesis and first falsification metric.
2. Run the historical-evidence checkpoint: summarize the durable prior evidence
   that constrains the route, including the current report digest, route, slice
   budget, known failure mechanism, and lanes that must not be repeated.
3. Run the durability checkpoint: inspect `git status --short`, identify
   whether the work depends on local-only source or gitignored evidence, and
   stop for commit/artifact triage when the backlog blocks reproducibility.
   For remote trainer work, also verify the trainer checkout is clean/current;
   if the default trainer checkout is dirty or stale, do not train there. Use a
   separate clean checkout via private `TRAINER_REPO`/`--repo` configuration or
   stop for explicit trainer cleanup approval.
4. Add or update the smallest diagnostic/report contract before heavy training.
5. Implement the smallest opt-in policy/data/artifact path.
6. Run focused tests and JSON parse checks before RTX work.
7. Before ending a completed source slice, make a Git durability decision:
   commit and push when authorized, or record the exact reason the slice is
   intentionally left local-only. Do not let code, tests, package scripts, or
   digest-bound ledger entries accumulate as an undocumented dirty backlog.
8. Use the RTX trainer only for CUDA or long seed/gate runs after source has
   been pushed and the trainer checkout has passed the clean/current check.
9. Evaluate on the strict broad-plus-fixture matrix or on a freshly declared
   diagnostic matrix when prior held-out seeds have been consumed as support.
10. Document the exact command, artifact/report paths, blockers, and next stop
   rule in `docs/mind-v3-autonomous-evolution.md`.

## Acceptance Surface

For promotion-style Mind v3 candidates, keep these hard:

- broad held-out seeds must be uncontaminated by the candidate's training,
  selection, support, or provenance artifacts. The historical broad diagnostic
  seeds `5,13,19,29,37,41` are no longer clean promotion-heldout evidence for
  scorers trained or selected on recent v165+ support/provenance artifacts;
  mint and document a fresh promotion-heldout matrix before promotion claims;
- controlled `carrion_only` fixture seeds `13,19,29,37,41,43` at `120` ticks
  for the current carrion recovery boundary;
- dominant requested-action share `<= 0.50`;
- zero heuristic action-source count;
- no per-seed alive or birth regression versus the linear Mind v3 baseline;
- carrion fixture milestone: terminal alive greater than zero on
  `carrion_only@120`; blocker-count reduction is diagnostic only.

Train-gate success alone is not promotion evidence.

## Preferred Commands

Focused local tests:

```bash
PYTHONPATH=python PYTHONHASHSEED=0 python3 -m unittest python.tests.<module>
git diff --check
```

Fast suite:

```bash
npm run sim:test
```

Historical strict candidate slice. Do not treat this labeled-IQL command, the
v179 audit route, a rerun of v180, or the already-completed v181/v182/v183/v184
diagnostics as the current next route; v180 already spent the first opt-in
transition-row training slice, v181/v182/v183/v184/v185 consumed no second
slice, v186 has now spent slice 2/10 and failed shadow acceptance, v187
recommended the no-training v188 terminal carrion survival support route, and
v188 found one selected legal support trajectory without training, v189 audited
that support and blocked slice 3, v190 targeted the support gaps but found
`0/6` clean legal support under the action-share cap, and v191 diagnosed the
unsupported resolved-action blocker as same-tick action-mask timing /
movement-occupancy races. v192 repaired the support-evidence contract without
runtime/replay behavior changes. v193 found `6/6` repaired-contract support
without training or slice-3 consumption. v194 audited that support into a
compact dataset without training or slice-3 consumption. v195 consumed slice
3/10 and failed shadow acceptance. v196 diagnosed the failure as
low-specificity artifact coverage collapse to `eat` without training or
slice-4 consumption. Future same-lane work routes to
`v197_slice_4_design_requires_coverage_or_abstention_repair_no_training`
and must not treat this command as a route back to v179/v180 authorization:

```bash
npm run sim:mind:v3:labeled-iql-slice -- \
  --candidate-artifact <artifact.json> \
  --enable-mind \
  --mind-runtime-mode autonomous \
  --seeds <fresh-promotion-heldout-seeds> \
  --ticks 120 \
  --fixture-names carrion_only \
  --fixture-seeds 13,19,29,37,41,43 \
  --fixture-ticks 120 \
  --output <slice-report.json>
```

Remote RTX pattern:

```bash
npm run trainer -- start mind-v3 --pull -- <command>
npm run trainer -- logs mind-v3
npm run trainer -- fetch <remote-artifact-or-report>
```

## Report

Include:

- files changed;
- behavior or contract changed;
- validation commands and results;
- artifact/report paths;
- strict acceptance blockers;
- dominant action share and heuristic action-source count;
- per-seed alive/birth deltas;
- controlled fixture result;
- why the next step is or is not worth pursuing.
