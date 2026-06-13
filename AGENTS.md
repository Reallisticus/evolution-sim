# AGENTS.md

This repo is a deterministic artificial-life simulator.

Current phase: Mind v3 autonomous-controller experimentation. Foundation,
replay, viewer, trajectory, and Mind v1/v2 guarded safety-floor contracts are
the measurement boundary, not the active research goal. Learned-controller work
must reduce heuristic action selection while preserving deterministic replay,
held-out diagnostics, strict per-seed gates, and Foundation release behavior.

The v61-v63 IQL validation loop closed the current scalar coefficient,
prior-blend, extraction-only, and global actor-bias family as non-promotable.
The v137-v176 audit trail then closed the tiny support-archive /
nearest-neighbor scorer loop as strategically saturated: the central carrion
survival blocker did not move and strict seeds were consumed as support
provenance. v177 restored transition-row data support through exact branch
replay, v178 audited that dataset as contract-valid but too small for
training-scale capacity work, and v179 locally expanded exact-branch
transition rows above the default v178 support thresholds. v180 consumed the
durable pinned v178 authorization and trained the first explicit opt-in
transition-row policy artifact, but shadow evaluation failed with zero
`carrion_only@120` terminal survivors and broad per-seed alive/birth
regressions. Its report and policy artifact are durable only after the recorded
backup
`gdrive:evolution-sim-backups/archives/20260611T110135Z-v180-transition-row-policy-training.tar.zst`
with archive SHA256
`e9c95f424e0827a7ed25f7de523fc61e9fe38d6062861466a81f1149671aac59`
and `rclone check` verification of `0` differences and `2` matching files. The
v181 failure-response autopsy then validated the pinned v180 report, artifact,
and dataset and diagnosed the failure as sparse imputed transition-value support
driving broad `eat` overrides while carrion fixture states had no complete
valid-action support. v181 wrote diagnostics only: no training, no slice-2
consumption, no runtime artifact, no runtime action-selection change, and no
promotion. Its report has exact digest
`a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751` and is
durable after
`gdrive:evolution-sim-backups/archives/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`
with archive SHA256
`a2b03e862ee095562c21482afcf0b1057a9ebb8f8b0b0d0f08e973dd884b1fd6`
and `rclone check` verification of `0` differences and `2` matching files. The
v182 failure-response design then added explicit imputed-valid-action and
observed-support-floor diagnostics, made opt-in transition-value action
overrides abstain when any currently valid action score is imputed or below the
chosen observed-support floor, and ran diagnostics-only shadow evaluation with
observed support floor `2`. It wrote
`output/mind/mind-v3-v182-carrion-survivor-continuation-imputed-abstention-design.json`
with exact digest
`3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b`.
Lifecycle flags stayed closed: no training, no training artifact, no runtime
artifact, no default runtime action-selection change, no promotion, and no
slice-2 consumption. v182 reduced broad overrides to `122`, kept dominant
requested-action share below the cap at `0.4157`, and kept heuristic action
sources at `0`, but broad seed `19` still regressed alive/birth by `-1/-2`,
`carrion_only@120` still had `0` terminal survivors, and carrion observed
support-floor coverage stayed `0/2122`. Its report is durable after
`gdrive:evolution-sim-backups/archives/20260611T165440Z-v182-imputed-abstention-design.tar.zst`
with archive SHA256
`d53401e5096d4c616a26691faa555ba53ffbdf84b4767061ff09d820340faf54`
and `rclone check` verification of `0` differences and `2` matching files.
v183 then ran diagnostics-only exact transition-support expansion against the
v182 support holes. It validated the pinned v182/v181/v180/v179 evidence,
materialized `24` exact branch points, and wrote
`output/mind/mind-v3-v183-carrion-survivor-continuation-exact-transition-support-expansion.json`
with exact digest
`7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da` plus
`output/mind/mind-v3-v183-carrion-survivor-continuation-expanded-compact-transition-rows.jsonl`
with dataset digest
`e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`.
The expanded dataset has `144` rows, `24` branches, `6` seeds, `9` forced
actions, and passes the default support summary. v183 moved carrion
observed-support target coverage from `0` to `18` materialized states and broad
seed `19` support-hole coverage to `6` materialized states. Lifecycle flags
stayed closed: no training, no training artifact, no runtime artifact, no
runtime action-selection change, no promotion, and no slice-2 consumption. The
report, dataset, and source trajectories are durable after
`gdrive:evolution-sim-backups/archives/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`
with archive SHA256
`b4e57537142303e622656c8eb51fde114a9452e20641a85cfdb4d742b425eb5c`
and `rclone check` verification of `0` differences and `2` matching files.
v184 then ran the fresh v178-style audit of the canonical v183 expanded dataset.
It validated the canonical v183 source report digest
`7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, the v183
dataset digest
`e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and the
v183 pinned v182/v181/v180/v179 source evidence, but it closed without slice-2
authorization because the target audit found `14` failures across `7`
attack-forced rows that resolved to `stay` with
`current_resolution_action_valid=false`. v184 wrote
`output/mind/mind-v3-v184-carrion-survivor-continuation-v183-transition-row-dataset-audit.json`
with exact digest
`ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa`.
Lifecycle flags stayed closed: no training, no training artifact, no runtime
artifact, no runtime action-selection change, no promotion, and no slice-2
consumption. Its report is durable after
`gdrive:evolution-sim-backups/archives/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`
with archive SHA256
`057d34c88f8a5da0c4ad98d87e4451caa043b7ee53c2af1037aa231dc5bcf382`
and `rclone check` verification of `0` differences and `2` matching files.
v185 then ran a diagnostics-only repair for the v183 target-resolution blocker.
It strict-filtered the `7` invalid attack-resolution rows from the canonical
v183 expanded dataset without backfill, leaving `137` rows, `24` branches, `6`
seeds, and `9` forced actions. It wrote
`output/mind/mind-v3-v185-carrion-survivor-continuation-v183-target-resolution-repair.json`
with exact digest
`327586651948e13670cf15285472a934c8cd0a8fc9784d2dcdb092d102c46486` plus
`output/mind/mind-v3-v185-carrion-survivor-continuation-v183-target-resolution-repaired-compact-transition-rows.jsonl`
with dataset digest
`532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`.
The paired v185 repaired-dataset audit then passed source, schema, leakage,
identity, action-mask, observation, target, and default-support checks with
explicit report and dataset digest pins. It wrote
`output/mind/mind-v3-v185-carrion-survivor-continuation-repaired-transition-row-dataset-audit.json`
with exact digest
`abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50` and
authorized only the future explicit route
`v186_transition_row_policy_training_slice_2_opt_in`. Lifecycle flags stayed
closed throughout v185: no training, no training artifact, no runtime artifact,
no runtime action-selection change, no promotion, and no slice-2 consumption.
The v185 report, repaired dataset, and repaired audit report are durable after
`gdrive:evolution-sim-backups/archives/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`
with archive SHA256
`c64aa17964462a4bbbc71fc83b98779213c50e26b714aedffe8cb7b42b9b5d2a`
and `rclone check` verification of `0` differences and `2` matching files.
Future work should not route back to v179/v180 authorization, rerun the first
transition-row training slice, or run more support expansion by default. After
explicit approval, v186 consumed the slice-2 route from the repaired v185 dataset
after validating the pinned repaired audit exact digest
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
Shadow acceptance failed with `0` `carrion_only@120` terminal survivors,
dominant requested-action share `0.4179`, heuristic action-source count `0`,
and no broad per-seed alive/birth regressions. Lifecycle flags stayed closed
for runtime and promotion: `runtime_artifact_created=false`,
`runtime_action_selection_changed=false`, and `promotion_authorized=false`.
The v186 report and artifact are durable after
`gdrive:evolution-sim-backups/archives/20260612T151102Z-v186-transition-row-policy-training-slice-2.tar.zst`
with archive SHA256
`9fba7700ec13b23f70f31b0269022c19b74bb7ed2967d2d70a9add05919eb31a` and
`rclone check` verification of `0` differences and `2` matching files. Do not
consume slice 3, relax gates, integrate runtime behavior, or promote from v186
without a separate explicit task. A separate explicit v187 diagnostics-only
delta blocker review then validated the pinned v186 report and artifact
digests, marked the v31/v36/v90-v91/v181-v182 findings as inherited rather
than rediscovered, and extracted only the new v186 delta: broad alive/birth
regressions were empty, dominant requested-action share was `0.4179`,
heuristic action-source count was `0`, `carrion_only@120` terminal survivors
stayed `0`, and carrion fixture births mean was `2.8333`. It wrote
`output/mind/mind-v3-v187-carrion-survivor-continuation-v186-delta-blocker-review.json`
with exact digest
`4366edf8b1b53e9974d6548e5fe9876e305693711a1b5d66192ef42f9b6228c6`.
Classification:
`m3_carrion_survivor_continuation_v187_v186_delta_blocker_review_terminal_survival_support_generation_before_slice_3_no_training`.
It recommended exactly
`v188_terminal_carrion_survival_support_generation_or_feasibility_proof_no_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no runtime artifact, no runtime action-selection change, no
promotion, no gate relaxation, no support expansion, and no v180/v186 rerun.
The v187 report is durable after
`gdrive:evolution-sim-backups/archives/20260612T161155Z-v187-v186-delta-blocker-review.tar.zst`
with archive SHA256
`c4705d5a53a5f6e5b47dd2125f4cd92d54fc5afb0cf5911044b05fc3f9dab01f` and
`rclone check` verification of `0` differences and `2` matching files. v188
then ran diagnostics/support evidence only after validating the pinned v187
exact digest and required route. It searched `carrion_only@120` seeds
`13,19,29,37,41,43` with `6` branch points and `30` bounded policy-visible
script continuations. Aggregate attempted continuations had terminal survivors
by seed `{13: 3, 19: 3, 29: 4, 37: 3, 41: 5, 43: 4}` but also `285`
unsupported resolved actions, so aggregate attempts are not support evidence.
The selected positive support evidence is one legal replay-verified branch
continuation for seed `13`, branch
`carrion-only-seed-13-branch-0-tick-0-agent-9`, continuation script
`conserve_after_carrion`, terminal alive `1`, births `5`, unsupported requested
actions `0`, and unsupported resolved actions `0`. It wrote
`output/mind/mind-v3-v188-carrion-survivor-continuation-terminal-survival-support.json`
with exact digest
`9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0` and
trajectory evidence under
`output/mind/v188-terminal-carrion-survival-support-trajectories/`.
Classification:
`m3_carrion_survivor_continuation_v188_terminal_carrion_survival_support_positive_legal_branch_support_found_no_training`.
It recommended exactly
`v189_terminal_survival_support_dataset_audit_before_slice_3_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no runtime artifact, no runtime action-selection change, no
promotion, and no gate relaxation. The v188 report and trajectory evidence are
durable after
`gdrive:evolution-sim-backups/archives/20260612T171041Z-v188-terminal-carrion-survival-support.tar.zst`
with archive SHA256
`67856f7b67c9156ad64c7da66ce5f118b03c1132d30c83b00c097ea0dedf7747` and
`rclone check` verification of `0` differences and `2` matching files.
v189 then ran a diagnostics-only terminal-survival support dataset audit before
slice-3 training. It validated the pinned v188 exact digest
`9a5a28685fd7adea175b1c8370f7042cdfbd1624b2c9ed3d0e42235b0bd4e5c0`, the
required v188 route
`v189_terminal_survival_support_dataset_audit_before_slice_3_training`, and the
inherited v187 digest pin. It audited the selected v188 support trajectory as
path-backed, replay-valid by report, action-mask legal, terminal-fact matched
(`alive=1`, `births=5`), and trainable-leakage clean. It also explicitly
deduped prior v32/v51 feasibility, v53-v63 IQL/coefficient/prior/action
distribution, v154-v176 tiny scorer/archive, v181-v182 sparse/imputed support,
and v183-v186 transition-row/repair/training work so those lanes should not be
rediscovered. Aggregate attempted v188 continuations remain diagnostic-only
because they had `285` unsupported resolved actions. Slice 3 stayed blocked:
clean legal support coverage is only `1/6` target seeds and the selected
trajectory dominant requested action is `stay` at share `0.539244`, above the
`0.50` cap. v189 wrote
`output/mind/mind-v3-v189-carrion-survivor-continuation-terminal-survival-support-dataset-audit.json`
with exact digest
`2505049a4a81f17ca5fb79f004c9d4ea7f43b1169f9e461e6a722df04e7d369e`.
Classification:
`m3_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit_support_insufficient_routes_to_targeted_legal_expansion_no_training`.
It recommended exactly
`v190_targeted_legal_terminal_survival_support_expansion_before_slice_3_training_no_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no runtime artifact, no runtime action-selection change, no
promotion, no gate relaxation, and no support expansion. The v189 report is
durable after
`gdrive:evolution-sim-backups/archives/20260612T175835Z-v189-terminal-survival-support-dataset-audit.tar.zst`
with archive SHA256
`c9a347d1504e2222473f3db2d49f2f0c5afe21c61ff877730c1fefe12045ff92` and
`rclone check` verification of `0` differences and `2` matching files.
v190 then ran the requested diagnostics-only targeted legal terminal-survival
support expansion before slice-3 training. It validated the pinned v189 exact
digest `2505049a4a81f17ca5fb79f004c9d4ea7f43b1169f9e461e6a722df04e7d369e`,
the required v189 route
`v190_targeted_legal_terminal_survival_support_expansion_before_slice_3_training_no_training`,
the v189 backup metadata, the exact v189 blockers, historical dedupe, and closed
v189 lifecycle flags. v190 targeted only seeds `13,19,29,37,41,43`: seeds
`19,29,37,41,43` lacked clean legal support and seed `13` needed
action-diversity repair. It ran `18` exact branch points and `90`
replay-verified continuations using the existing policy-visible script set.
Aggregate attempted continuations still had terminal survivors but `860`
unsupported resolved actions, and no trajectory satisfied the v190 support
contract; clean legal support coverage was `0/6`. v190 wrote
`output/mind/mind-v3-v190-carrion-survivor-continuation-targeted-legal-terminal-survival-support-expansion.json`
with exact digest
`b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5` and
trajectory evidence under
`output/mind/v190-targeted-legal-terminal-survival-support-trajectories/`.
Classification:
`m3_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion_partial_or_empty_support_routes_to_repair_no_training`.
It recommended exactly
`v191_targeted_legal_support_repair_or_architecture_review_no_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no runtime artifact, no runtime action-selection change, no
promotion, and no gate relaxation. The v190 report and trajectory evidence are
durable after
`gdrive:evolution-sim-backups/archives/20260613T100008Z-v190-targeted-legal-terminal-survival-support-expansion.tar.zst`
with archive SHA256
`e2c1392b066342a43c78274646bfce6589adacc5a61ec3e296f574fed1e7d333` and
`rclone check` verification of `0` differences and `2` matching files.
v191 then ran diagnostics-only targeted legal-support repair / architecture
review. It validated the pinned v190 exact digest
`b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`, required
route `v191_targeted_legal_support_repair_or_architecture_review_no_training`,
v190 backup metadata, `18` branch points, `90` replay-verified continuations,
`0/6` clean support, `0` unsupported requested actions, `860` unsupported
resolved actions, and closed training/runtime/promotion lifecycle flags. v191
read the v190 trajectory artifacts instead of rerunning support expansion. All
`860` unsupported resolved records were observation-valid movement requests
whose requested action was true in the tick-start public action mask and false
in the live resolution mask, resolved to `stay`, and carried legality reason
`not_in_resolution_action_mask`; the classified root cause is same-tick
action-mask timing / movement-occupancy races, not an illegal script request,
water/hazard blocker, or digest/report count mismatch. Near-clean seed `29`
needs action-diversity cap repair only (`stay` share `0.5154` with zero
unsupported actions); near-clean seed `41` needs both action-resolution contract
repair and small action-share repair (`stay` share `0.5014` with `4`
unsupported resolved actions). v191 wrote
`output/mind/mind-v3-v191-carrion-survivor-continuation-legal-support-repair-architecture-review.json`
with exact digest
`eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5`.
Classification:
`m3_carrion_survivor_continuation_v191_legal_support_repair_architecture_review_action_resolution_contract_repair_route_no_training`.
It recommended exactly `v192_action_resolution_contract_repair_no_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no support generation, no runtime artifact, no runtime
action-selection change, no promotion, and no gate relaxation. The v191 report
is durable after
`gdrive:evolution-sim-backups/archives/20260613T122723Z-v191-legal-support-repair-architecture-review.tar.zst`
with archive SHA256
`bf0184dc2caa6be860723a4b0d6daa35b43264560117dc0794b45c8ce45fdff3` and
`rclone check` verification of `0` differences and `2` matching files.
v192 then ran diagnostics-only action-resolution contract repair after
validating the pinned v191 exact digest
`eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5`,
required route `v192_action_resolution_contract_repair_no_training`, v190
digest pin, all v191 unsupported-action counts, and closed lifecycle flags.
It chose exactly `support_evidence_contract_repair`, not a runtime/replay
semantics change and not a trajectory schema change. It re-read the canonical
v190 trajectories and serialized derived movement target/blocker audit fields
for all `860` unsupported resolved actions: all were observation-valid movement
requests, resolution-invalid under the live mask, in-bounds by target
coordinates, resolved to `stay`, and classified as
`resolution_invalid_same_tick_occupancy_race`; bounds, water, hazard,
depleted-resource, and stale/illegal script causes were not present. v192 wrote
`output/mind/mind-v3-v192-carrion-survivor-continuation-action-resolution-contract-repair.json`
with exact digest
`17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`.
Classification:
`m3_carrion_survivor_continuation_v192_action_resolution_contract_repair_support_evidence_contract_repaired_routes_to_fresh_support_no_training`.
It recommended exactly
`v193_fresh_targeted_legal_support_expansion_after_action_resolution_contract_repair_no_training`.
Lifecycle flags stayed closed: no training, no training artifact, no slice-3
consumption, no support generation, no support expansion, no runtime artifact,
no runtime action-selection change, no promotion, and no gate relaxation. The
v192 report is durable after
`gdrive:evolution-sim-backups/archives/20260613T151045Z-v192-action-resolution-contract-repair.tar.zst`
with archive SHA256
`e5210b816dddfb6e54ad48fadad417f057fa7de93af6981acbec46991f265fac` and
`rclone check` verification of `0` differences and `1` matching file.

Carrion lane campaign charter: once a transition-row dataset passes the v178
default support thresholds, has explicit source-report and dataset digest pins,
and all source, schema, leakage, identity, action-mask, observation, and target
contracts are valid, the next same-lane route is the first opt-in transition-row
training slice, not another design audit. For v179-sourced datasets, v178 must
also verify that v179 itself pinned the upstream v177 source report and dataset
digests. The CLI `--min-*` support overrides are diagnostic only and must not
authorize that training route unless they exactly match the default v178
thresholds. The campaign budget is at most 10 slices before a forced retro. The
first milestone is nonzero terminal survivors on `carrion_only@120`, no
dominant requested action share above `0.50`, and zero heuristic action
sources. v180 counts as slice 1 of that budget; its dominant requested-action
share was under the cap and heuristic action sources stayed zero, but the
survivor and broad regression gates failed. The first opt-in transition-row
training slice has already been spent; do not route future work back to
pre-v180 authorization unless a later durability check proves the recorded
artifacts are unavailable and the user explicitly asks for regeneration. v181
and v182 were diagnostics-only, v183 created only diagnostic transition-row
evidence, v184 was an audit-only closed report, and v185 repaired/re-audited
diagnostic transition-row evidence; none consume slice 2. v186 consumed slice
2/10 from the repaired v185 dataset and failed shadow acceptance only on the
controlled carrion survivor milestone. The campaign budget is now 2/10; v186
is not promotion evidence and authorizes no runtime integration. v187 consumed
no slice and authorizes no training/runtime/promotion work; the next same-lane
route it authorized has now completed as v188 diagnostics/support evidence.
v188 found one selected legal terminal-survival support trajectory and
authorizes only the fresh v189 terminal-survival support dataset audit route
before any slice-3 training. That v189 audit has now completed and did not
authorize slice 3. v190 targeted those exact support gaps and also failed to
produce clean legal support under the action-share cap. v191 diagnosed the v190
unsupported resolved-action blocker as a same-tick action-mask timing /
movement-occupancy race and generated no support. v192 repaired the
support-evidence contract for that blocker without changing runtime/replay
semantics or generating support. The next same-lane route is
`v193_fresh_targeted_legal_support_expansion_after_action_resolution_contract_repair_no_training`,
not slice-3 training, not runtime integration, and not another old feasibility,
counterfactual, IQL, scorer, blind support expansion, or v180/v186 rerun. The
campaign budget remains 2/10.

Handoff constraint: pasted strategic ledgers and thread attachments are not
durable repo context, and a fresh coder or remote trainer may not have access to
them. Any direction that must guide future work needs to be copied into the
dispatch prompt or summarized in committed docs before it is treated as source
of truth.

2026-06-10 repository-audit checkpoint: the v137-v176 source backlog was
preserved, committed, pushed, and paired with a documented `output/mind/`
backup. Keep that durability bar for new Mind v3 work: do not start a new
experiment while its source, package entrypoints, tests, ledger entries, or
digest-referenced artifacts are local-only unless the user explicitly asks for a
docs-only or negative-control slice. See
`docs/repository-audit-2026-06-10-remediation.md`.

PRs must not be merged until CI is green.

## Environment

- Python: 3.14.3 from `.python-version`
- Node: npm scripts are the command entrypoints
- Python package path: `PYTHONPATH=python`
- Determinism: use `PYTHONHASHSEED=0` for simulator commands

Prefer npm scripts over raw Python commands when a script exists.

## Validation Ladder

Fast local checks:
- `npm run sim:test`
- `npm run sim:golden:quick`
- `npm run sim:gate:quick`
- `npm run sim:bench:quick`
- `npm run viewer:contracts:check`
- `npm run viewer:model:test`
- `npm run viewer:map:test`
- `npm run viewer:validate`
- `git diff --check`

Viewer check:
- `npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json`
- `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`

Gate boundary checks:
- `npm run sim:golden`
- `npm run sim:test:full`
- `npm run sim:gate:release`
- `npm run sim:mind:gate -- --reuse-trajectories --fail-on-blockers`
- `npm run sim:mind:gate:extended`
- `npm run sim:bench`

## Project Rules

- Full replay is the compatibility contract.
- Summary-only mode must stay lightweight and must not build viewer/replay payloads.
- Long sweeps are opt-in; use compact contract equivalents in the fast loop.
- Every world mechanic needs matching observability, metrics, tests, and viewer/replay semantics where applicable.
- Mind contracts may evolve only with matching artifact/runtime diagnostics and held-out gates.
- Do not change golden/replay semantics casually.
- Do not make tests poke cache internals or call `_invalidate_biotic_state()` directly.
- Mind v3 promotion requires strict broad held-out seed behavior and controlled fixture evidence; aggregate gains do not excuse per-seed alive or birth regressions.
- Do not relax action-collapse, heuristic-action, or per-seed gates to make a candidate pass. Record the first failing seed, fixture, and action distribution instead.
- Keep novel controller work opt-in, serialized, deterministic, and replayable. Do not add hidden heuristic action selection or fixture-specific runtime identity.
- Do not treat seeds `5,13,19,29,37,41` as clean promotion-heldout evidence for scorers trained or selected using recent support/provenance artifacts that consumed those seeds. Mint and document a fresh promotion-heldout matrix before promotion-style claims.
- Do not hardcode digests of gitignored Mind artifacts without a durable artifact backup path and retrieval note.
- Do not make `python/evolution_sim/cli/` a shared library for new Mind experiment code. Shared fixture, gate, digest, and report helpers belong under `python/evolution_sim/mind/` and CLIs should be thin wrappers.
- The habitat water encoding bug was fixed as a dedicated replay/viewer contract change. Future replay/viewer contract changes need the same tests, docs, viewer handling, and golden regeneration.
- Push-time CI Mind gates are compact contract coverage, not promotion evidence. Strict Mind claims require the explicit strict commands and recorded per-seed blockers.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `Reallisticus/evolution-sim`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default Matt Pocock skills triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Use a multi-context domain-doc layout rooted at `CONTEXT-MAP.md`. See `docs/agents/domain.md`.
