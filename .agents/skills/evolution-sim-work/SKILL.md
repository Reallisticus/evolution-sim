---
name: evolution-sim-work
description: Use for implementation work in this evolution-sim repository, including simulator fixes, replay/viewer changes, runtime refactors, tests, benchmarks, gates, and documentation updates. Follow the repo-specific loop instead of generic TypeScript, pnpm, or web-app workflows.
---

# Evolution Sim Work

Use this as the default implementation workflow for this repository.

## Start Here

1. Inspect the current repo context before editing:
   - `README.md`
   - `package.json`
   - relevant files in `docs/`
   - relevant source under `python/evolution_sim/`
   - relevant tests under `python/tests/`
   - recent audit docs when the task touches Foundation or Mind readiness
2. Check the working tree with `git status --short`.
3. Treat existing unrelated changes as user-owned. Do not revert them.
4. For Mind v3 work, read
   `docs/repository-audit-2026-06-10-remediation.md` before editing. The
   v137-v176 backlog has been committed and backed up; apply the same rule to
   new work by stopping before any experiment slice that depends on dirty-tree
   source, uncommitted entrypoints/tests/docs, or local-only digest-referenced
   artifacts unless the user explicitly asks for docs-only work, backlog triage,
   or a negative control.
5. For non-trivial work, state:
   - what you found
   - what files are likely involved
   - what behavior or invariant will change
   - what validation you will run

## Repo Shape

- Python simulator code lives in `python/evolution_sim/`.
- Tests live in `python/tests/`.
- Browser viewer code lives in `viewer/`.
- Docs and durable project specs live in `docs/`.
- `package.json` npm scripts are the preferred command entrypoints.
- Use `PYTHONHASHSEED=0` and `PYTHONPATH=python` when running raw simulator commands.

## Current Mind v3 Context

Mind v3 is the active autonomous-controller track. Read
`docs/mind-v3-autonomous-evolution.md` before touching Mind v3 policy,
dataset, training, gate, artifact, ledger, or acceptance code.

As of the v61-v63 closeout, the current single-observation IQL
coefficient/prior/extraction family is exhausted for promotion. The 2026-06-10
audit also closed the recent tiny support-archive / nearest-neighbor scorer loop
as strategically saturated until exact branch replay, transition-row support, or
scaled policy capacity changes the data path. Do not spend a new slice on
scalar IQL tuning, prior blends, action-share losses, risk extraction, global
actor-bias calibration, residual-threshold tuning, or another micro-archive
nearest-neighbor probe unless the user explicitly asks for a negative-control
probe.

Preserve these boundaries:

- no heuristic runtime action selection in autonomous Mind v3 candidates;
- no fixture identity or private world state in policy inputs;
- strict per-seed no-regression gates for alive and birth metrics;
- dominant action and heuristic-action gates remain hard blockers;
- controlled carrion fixture behavior is measured, not inferred from broad runs;
- every new policy state, artifact field, or data contract gets diagnostics and
  tests.

v180 has already consumed the durable pinned v178 authorization and run the
first opt-in transition-row policy training slice. It failed shadow acceptance
with zero `carrion_only@120` terminal survivors and broad per-seed alive/birth
regressions, while dominant requested-action share stayed below `0.50` and
heuristic action sources stayed at `0`. v180 counts as slice 1 of the 10-slice
carrion campaign budget. Its report and policy artifact are durable only after
the recorded backup
`gdrive:evolution-sim-backups/archives/20260611T110135Z-v180-transition-row-policy-training.tar.zst`
with archive SHA256
`e9c95f424e0827a7ed25f7de523fc61e9fe38d6062861466a81f1149671aac59`.
The v181 failure-response autopsy has now run as diagnostics-only work. It
validated the pinned v180 report, artifact, and dataset, diagnosed the failure
as sparse imputed transition-value support driving broad `eat` overrides while
carrion fixture states had no complete valid-action support, and consumed no
training slice. The v181 report exact digest is
`a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751`; it is
durable after
`gdrive:evolution-sim-backups/archives/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`
with archive SHA256
`a2b03e862ee095562c21482afcf0b1057a9ebb8f8b0b0d0f08e973dd884b1fd6`.
The v182 failure-response design then ran as diagnostics-only work. It added
imputed-valid-action and observed-support-floor diagnostics, made opt-in
transition-value overrides abstain on imputed or below-floor currently valid
action scores, and consumed no second training slice. Its report exact digest is
`3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b`; it is
durable after
`gdrive:evolution-sim-backups/archives/20260611T165440Z-v182-imputed-abstention-design.tar.zst`
with archive SHA256
`d53401e5096d4c616a26691faa555ba53ffbdf84b4767061ff09d820340faf54`.
The v183 exact transition-support expansion then ran as diagnostics-only work.
It validated the pinned v182/v181/v180/v179 evidence, wrote report digest
`7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, wrote a
`144`-row expanded transition dataset with digest
`e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and
created no training artifact, runtime artifact, runtime action-selection
change, promotion, or slice-2 consumption. Carrion observed-support target
coverage moved from `0` to `18` materialized states and broad seed `19`
support-hole coverage moved to `6` materialized states. The report, dataset,
and source trajectories are durable after
`gdrive:evolution-sim-backups/archives/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`
with archive SHA256
`b4e57537142303e622656c8eb51fde114a9452e20641a85cfdb4d742b425eb5c`.
The v184 fresh v178-style audit of the canonical v183 dataset then ran as
diagnostics-only work. It validated the v183 source report digest
`7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, the v183
dataset digest
`e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and
v183's pinned v182/v181/v180/v179 evidence, but it did not authorize slice 2:
the target audit found `14` failures across `7` attack-forced rows that
resolved to `stay` with `current_resolution_action_valid=false`. The v184
report exact digest is
`ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa`; it is
durable after
`gdrive:evolution-sim-backups/archives/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`
with archive SHA256
`057d34c88f8a5da0c4ad98d87e4451caa043b7ee53c2af1037aa231dc5bcf382`.
v185 then repaired the v183 target-resolution blocker as diagnostics-only work.
It strict-filtered the `7` invalid rows without backfill, leaving `137` rows,
`24` branches, `6` seeds, and `9` forced actions. The v185 repair report exact
digest is
`327586651948e13670cf15285472a934c8cd0a8fc9784d2dcdb092d102c46486`; the
repaired dataset digest is
`532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`; the
paired repaired-dataset audit exact digest is
`abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`. The
repaired audit passed source, schema, leakage, identity, action-mask,
observation, target, and default-support checks and authorized only the future
explicit route `v186_transition_row_policy_training_slice_2_opt_in`. v185
trained nothing, spent no slice 2, created no runtime artifact, changed no
runtime action selection, and authorized no promotion. Its artifacts are durable
after
`gdrive:evolution-sim-backups/archives/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`
with archive SHA256
`c64aa17964462a4bbbc71fc83b98779213c50e26b714aedffe8cb7b42b9b5d2a`.
Future same-lane work should not run another support expansion by default,
reroute to v179/v180 authorization, or rerun the first training slice. CLI
`--min-*` support overrides are diagnostic only and must not authorize training
routes. v186 has now consumed the explicit slice-2 route from the repaired v185
dataset. It validated the repaired audit exact digest
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
and no broad per-seed alive/birth regressions. The v186 report and artifact are
durable after
`gdrive:evolution-sim-backups/archives/20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst`
with archive SHA256
`9fba7700ec13b23f70f31b0269022c19b74bb7ed2967d2d70a9add05919eb31a` and
`rclone check` verification of `0` differences and `2` matching files. The
campaign budget is now 2/10. A separate explicit v187 diagnostics-only
delta blocker review has now run. It validated the pinned v186 report digest
`f4f404b88093f00bc8e7d655ff6f1937c4e4786acdca6118275decb463193075` and
artifact digest
`729997cd3a3672dd3ceabf08ccb4a9b5a7ce67f0621ecdb73ff4216dd921b6ce`,
marked v31/v36/v90-v91/v181-v182 findings as inherited rather than
rediscovered, and extracted only the new v186 delta: no broad alive/birth
regressions, dominant requested-action share `0.4179`, heuristic action-source
count `0`, zero `carrion_only@120` terminal survivors, and carrion fixture
births mean `2.8333`. It wrote report digest
`4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6`, classified
as
`m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_terminal_survival_support_generation_before_slice_3_no_training`,
and recommended exactly
`v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no runtime artifact, no runtime action-selection change, no
promotion, no gate relaxation, no support expansion, and no v180/v186 rerun.
The v187 report is durable after
`gdrive:evolution-sim-backups/archives/20260612T161155Z-v187-v186-delta-blocker-review.tar.zst`
with archive SHA256
`c4705d5a53a5f6e5b47dd2125f4cd92d54fc5afb0cf5911044b05fc3f9dab01f` and
`rclone check` verification of `0` differences and `2` matching files. Do not
consume slice 3, relax gates, integrate runtime behavior, or promote from v187
without a separate explicit task. v188 has now run as diagnostics/support
evidence only. It validated the pinned v187 exact digest
`4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6` and route
`v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`,
then searched `carrion_only@120` seeds `13,19,29,37,41,43` with `6` branch
points and `30` bounded policy-visible continuations. Aggregate attempted
continuations had positive terminal survivors for all six seeds but included
`285` unsupported resolved actions, so only the selected legal evidence counts:
seed `13`, branch `carrion-only-seed-13-branch-0-tick-0-agent-9`, continuation
`conserve_after_carrion`, terminal alive `1`, births `5`, replay verified,
unsupported requested actions `0`, and unsupported resolved actions `0`. The
v188 report digest is
`9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0`, and its
report plus trajectory evidence are durable after
`gdrive:evolution-sim-backups/archives/20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst`
with archive SHA256
`67856f7b67c9156ad64c7da66ce5f118b03c1132d30c83b00c097ea0dedf7747` and
`rclone check` verification of `0` differences and `2` matching files. v188
kept training, slice-3 consumption, runtime artifacts, runtime action
selection, promotion, and gate relaxation closed, and recommended exactly
`v189_terminal_survival_support_dataset_audit_before_slice_3_training`. v189 has
now run as the requested diagnostics-only terminal-survival support dataset
audit. It validated v188 exact digest
`9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0`, audited
the selected support trajectory as path-backed, replay-valid by report,
action-mask legal, terminal-fact matched, and trainable-leakage clean, and
recorded historical dedupe so v32/v51, v53-v63, v154-v176, v181-v182, and
v183-v186 are not rerun as missing facts. It blocked slice 3 because clean
legal support coverage is only `1/6` target seeds, the selected dominant
requested action share is `0.539244` above `0.50`, and aggregate v188 attempts
had `285` unsupported resolved actions. The v189 report digest is
`2505049a4a81f17ca5fb79f004c9d4ea7f43b1169f9e461e6a722df04e7d369e`; it is
durable after
`gdrive:evolution-sim-backups/archives/20260612T175835Z-v189-terminal-survival-support-dataset-audit.tar.zst`
with archive SHA256
`c9a347d1504e2222473f3db2d49f2f0c5afe21c61ff877730c1fefe12045ff92` and
`rclone check` verification of `0` differences and `2` matching files. It
recommended exactly
`v190_targeted_legal_terminal_survival_support_expansion_before_slice_3_training_no_training`. The
campaign budget remains 2/10. v190 has now run as diagnostics-only targeted
legal terminal-survival support expansion. It validated v189 digest
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
`rclone check` verification of `0` differences and `2` matching files. It
recommended exactly
`v191_targeted_legal_support_repair_or_architecture_review_no_training`. The
campaign budget remains 2/10. v191 has now run as diagnostics-only
legal-support repair / architecture review. It validated v190 digest
`b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`, read the
canonical v190 trajectories, and classified all `860` unsupported resolved
actions as observation-valid movement requests that became invalid under the
live resolution mask and resolved to `stay`. Root cause:
`action_mask_timing_mismatch_same_tick_movement_occupancy_race`. It wrote
report digest
`eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5`; the
report is durable after
`gdrive:evolution-sim-backups/archives/20260613T122723Z-v191-legal-support-repair-architecture-review.tar.zst`
with archive SHA256
`bf0184dc2caa6be860723a4b0d6daa35b43264560117dc0794b45c8ce45fdff3` and
`rclone check` verification of `0` differences and `2` matching files. v191
generated no support, trained nothing, consumed no slice 3, changed no runtime
behavior, and recommended exactly
`v192_action_resolution_contract_repair_no_training`. The campaign budget
remains 2/10.

Handoffs must not rely on pasted strategic ledgers or thread attachments unless
the relevant content is included in the new prompt or committed to repo docs.
Fresh coders and remote trainers should be assumed to have only repository
state, explicit prompt text, and named artifacts.

Current durability constraint: the v137-v176 source/evidence backlog is already
preserved, committed, pushed, and backed up. Keep that standard for the current
slice: source, package entrypoints, tests, ledger entries, and any
digest-referenced artifacts must be durable before remote-trainer scale work.
The trainer pulls committed source from GitHub; it cannot execute local-only
dirty-tree experiments.

## Remote Trainer

The project can use a locally configured remote trainer for accelerator-backed
training, long Mind gates, long seed sweeps, and heavier CPU-bound simulator
runs that would block local development. Keep host aliases, remote paths,
network details, account names, and hardware identifiers in private local
configuration. The generic workflow is documented in
`docs/remote-rtx-trainer.md`.

Default workflow:

1. Edit, review, and run quick checks on the Mac.
2. Push source changes to GitHub.
3. Fast-forward the trainer with `npm run trainer:pull`.
4. Start long trainer jobs in remote `tmux` with `npm run trainer -- start ...`.
5. Fetch only needed artifacts back with `npm run trainer -- fetch ...`.

Useful commands:

- `npm run trainer:status`
- `npm run trainer:doctor`
- `npm run trainer:gpu`
- `npm run trainer -- run npm run sim:bench:quick`
- `npm run trainer -- start mind-gate --pull -- npm run sim:mind:gate:extended`
- `npm run trainer -- start mind-v3 --pull -- <mind-v3 command>`
- `npm run trainer -- logs mind-gate`
- `npm run trainer -- attach mind-gate`

Known trainer caveats:

- `sim:golden:quick` has a known Mac/Linux last-bit viewer-float mismatch, so treat that as a repo determinism issue rather than a trainer setup failure.
- Do not commit private trainer topology, host aliases, usernames, remote paths, or network details.

## Implementation Rules

- Make the smallest coherent vertical change.
- Prefer existing simulator patterns and public test surfaces.
- Preserve deterministic behavior under fixed seeds.
- Preserve replay compatibility unless the task explicitly changes the contract.
- Keep `FULL_REPLAY` and `SUMMARY_ONLY` semantics distinct.
- Every world mechanic needs matching observability, metrics, and tests.
- Do not make long sweeps part of the normal fast loop.
- Do not add dependencies unless the repo clearly needs them.
- Avoid touching viewer, replay schema, runtime contracts, and goldens in the same change unless the task requires it.
- Do not hardcode a digest of a gitignored Mind artifact without documenting
  where that artifact is durably stored or how source-integrity should be
  overridden for regenerated, metric-equivalent artifacts.
- Do not add new imports from `python/evolution_sim/cli/` into `mind/` modules;
  shared experiment harness code belongs under `python/evolution_sim/mind/`.

## Test Discipline

Prefer focused behavior tests through public interfaces. Do not test by mutating cache internals or calling private invalidation helpers.

Use a narrow-to-broad validation ladder:

1. Targeted unit or integration test:
   - `PYTHONPATH=python PYTHONHASHSEED=0 python3 -m unittest python.tests.<module>`
2. Fast simulator suite:
   - `npm run sim:test`
3. Replay/golden compatibility when relevant:
   - `npm run sim:golden:quick`
   - `npm run sim:golden`
4. Foundation gate when behavior affects readiness:
   - `npm run sim:gate:quick`
5. Viewer smoke when replay/viewer surfaces change:
   - `npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json`
   - `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`
6. Benchmark checks when performance-sensitive:
   - `npm run sim:bench:quick`

Always run `git diff --check` before finalizing edits.

## Completion Report

Report:

- files changed
- behavior changed
- validation commands and results
- any checks skipped and why
- remaining uncertainty or next slice

For Mind v3 candidates, also report:

- artifact/report paths
- strict acceptance blockers
- dominant requested-action share and heuristic action-source count
- per-seed alive and birth deltas
- controlled fixture result
