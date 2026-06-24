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
v193 then ran diagnostics-only fresh targeted legal terminal-survival support
expansion under the v192 repaired support-evidence contract. It validated the
pinned v192 exact digest
`17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`, required
route
`v193_fresh_targeted_legal_support_expansion_after_action_resolution_contract_repair_no_training`,
and inherited v191/v190 digests
`eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5` /
`b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`.
It ran the same bounded exact branch exploration over seeds
`13,19,29,37,41,43`: `18` branch points and `90` replay-verified continuations.
Under the repaired contract, observation-valid same-tick occupancy drift is
counted separately, not as an unsupported requested action or successful move.
v193 found repaired-contract support for all `6/6` target seeds, with aggregate
unsupported requested actions `0`, expected same-tick occupancy drift `860`,
unexpected resolution-invalid events `0`, and selected-support dominant
requested-action share `0.207558` (`stay`). Per-seed selected support:
`13 alive=4 births=12 drift=9`, `19 alive=2 births=10 drift=14`,
`29 alive=3 births=12 drift=23`, `37 alive=3 births=10 drift=7`,
`41 alive=2 births=13 drift=10`, `43 alive=5 births=12 drift=14`.
v193 wrote
`output/mind/mind-v3-v193-carrion-survivor-continuation-fresh-targeted-legal-support-expansion-after-contract-repair.json`
with exact digest
`006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21`.
Classification:
`m3_carrion_survivor_continuation_v193_fresh_targeted_legal_support_expansion_after_contract_repair_all_target_seed_support_routes_to_fresh_dataset_audit_no_training`.
It recommended exactly
`v194_repaired_contract_terminal_survival_support_dataset_audit_before_slice_3_training_no_training`.
Lifecycle flags stayed closed for training/runtime/promotion: no training, no
training artifact, no slice-3 consumption, no runtime artifact, no runtime
action-selection change, no promotion, and no gate relaxation. The v193 report
and trajectory evidence are durable after
`gdrive:evolution-sim-backups/archives/20260613T165955Z-v193-fresh-targeted-legal-support-expansion-after-contract-repair.tar.zst`
with archive SHA256
`ceaeb7c2027551393adb1d1374d7f5fd5621aadc3374275992a28b002da0b964` and
`rclone check` verification of `0` differences and `1` matching file.
v194 then ran diagnostics-only repaired-contract terminal-survival support
dataset audit before any slice-3 training. It validated the pinned v193 exact
digest `006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21`,
required route
`v194_repaired_contract_terminal_survival_support_dataset_audit_before_slice_3_training_no_training`,
inherited v192/v191/v190 digests
`17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae` /
`eef890ac70140028f9d407d0b24fca40827947e75153ba5e9563b027a4233ce5` /
`b53de9d19f67f681e334520c030978f97f6f7486e44a9f9808be11de269234b5`, and
validated the v193 backup metadata. It audited the six selected support
trajectories as path-backed, readable, replay-verified, terminal-fact matched,
leakage-clean, action-mask legal under the repaired contract, and target-valid.
Aggregate selected support kept dominant requested action `stay` at share
`0.207558`, unsupported requested actions `0`, expected same-tick occupancy
drift counted separately at aggregate `860` and selected `77`, and unexpected
resolution-invalid events `0`. v194 wrote
`output/mind/mind-v3-v194-carrion-survivor-continuation-repaired-contract-terminal-survival-support-dataset-audit.json`
with exact digest
`ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e` and a
gitignored compact support dataset
`output/mind/mind-v3-v194-carrion-survivor-continuation-repaired-contract-terminal-survival-support-compact-dataset.jsonl`
with dataset digest
`dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc`.
Classification:
`m3_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit_support_dataset_ready_for_future_explicit_slice_3_training`.
It recommended exactly
`v195_repaired_contract_terminal_survival_support_training_slice_3_opt_in`.
Lifecycle flags stayed closed for training/runtime/promotion: no training, no
training artifact, no slice-3 consumption, no runtime artifact, no runtime
action-selection change, no promotion, no gate relaxation, no support
generation, and no support expansion. The v194 report and compact dataset are
durable after
`gdrive:evolution-sim-backups/archives/20260613T175848Z-v194-repaired-contract-terminal-survival-support-dataset-audit.tar.zst`
with archive SHA256
`461295919b0ed81c4d7dc2a4a7ffcb4ceab82c5e4779256205419180129d748e` and
`rclone check` verification of `0` differences and `1` matching file.
v195 then ran the explicit opt-in repaired-contract terminal-survival support
training slice 3 after validating the pinned v194 report exact digest
`ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e`, compact
dataset digest `dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc`,
required route
`v195_repaired_contract_terminal_survival_support_training_slice_3_opt_in`,
v194 backup metadata, v193 digest
`006a1bf33a4c092a9ae584603cddb434b868f21493cde8f710c9d816c66c1a21`, v192
digest `17c49629d1268ff2f15d248465241775b9205b212222cc56a3132a7b9b8dfdae`,
closed v194 lifecycle flags, `3599` dataset rows, leakage-free trainable
payloads, unsupported requested actions `0`, selected expected same-tick
occupancy drift `77`, and unexpected resolution-invalid events `0`. Training
ran and consumed slice 3/10, writing
`output/mind/mind-v3-v195-carrion-survivor-continuation-repaired-contract-terminal-survival-support-training.json`
with exact digest
`0d69540a3817b7b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde` and
`output/mind/mind-v3-v195-carrion-survivor-continuation-repaired-contract-terminal-survival-support-policy-artifact.json`
with artifact digest
`9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470`.
Shadow acceptance failed: `carrion_only@120` terminal survivors stayed `0`,
dominant requested-action share was `0.9422`, heuristic action-source count was
`0`, and broad seeds `5,13,19,29,37,41` all regressed alive/birth. Broad
alive/birth deltas were `5:-6/-5`, `13:-6/-3`, `19:-10/-10`, `29:-9/-5`,
`37:-10/-8`, and `41:-7/-6`. Classification:
`m3_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training_slice_3_shadow_acceptance_failed_routes_to_failure_response`.
It recommended exactly
`v196_repaired_contract_slice_3_failure_response_no_training`. Runtime and
promotion lifecycle flags stayed closed: no runtime artifact, no runtime
action-selection change, no promotion, and no gate relaxation. The v195 report
and artifact are durable after
`gdrive:evolution-sim-backups/archives/20260613T185630Z-v195-repaired-contract-terminal-survival-support-training-slice-3.tar.zst`
with archive SHA256
`f4d3785619e3937949654cd586d1896fdc20d89fa3d9f3f14d191748ca776049` and
`rclone check` verification of `0` differences and `1` matching file.
v196 then ran diagnostics-only failure response for the v195 slice-3 failure.
It validated the pinned v195 report digest
`0d69540a3817b7b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde`, v195
artifact digest
`9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470`, v194
report digest `ccd02e7bc4862f8d8797fa21efb3d29fc2034196d2a54a6d42032062e2caaf2e`,
v194 dataset digest
`dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc`, required
route `v196_repaired_contract_slice_3_failure_response_no_training`, v195
backup metadata, slice `3/10` consumption, and the exact v195 failure facts.
It compared training labels (`stay` dominant share `0.207558`) against shadow
actions (`eat` dominant share `0.9422` on `carrion_only@120`), reran only
diagnostic lookup coverage with the frozen artifact, and found the action
collapse mechanism to be
`artifact_low_specificity_feature_coverage_collapse_to_eat_with_miss_abstention`.
Combined diagnostics had override-applied share `0.782944`, dominant applied
override action `eat` at share `0.939312`, mask-only feature-hit share
`0.778513`, exact feature-hit share `0.0`, and missing states did not default
to `stay`. v186 comparison confirmed that v186 also had zero carrion terminal
survivors but not the v195 `0.9422` action collapse or broad regression
profile. v196 wrote
`output/mind/mind-v3-v196-carrion-survivor-continuation-repaired-contract-slice-3-failure-response.json`
with exact digest
`7b3707b60575a14aee806b3989615582369f021f1ceced80e664e4fdd1794d90`.
Lifecycle stayed closed: no training, no slice-4 consumption, no runtime
artifact, no runtime action-selection change, no gate relaxation, and no
promotion. The v196 report is durable after
`gdrive:evolution-sim-backups/archives/20260613T213246Z-v196-repaired-contract-slice-3-failure-response.tar.zst`
with archive SHA256
`363b4ddb144b8966095c9a1e66f259c418b2a9f4f0c474449fa60d5f06baa45b` and
`rclone check` verification of `0` differences and `1` matching file.
v197 then completed the no-training, contract-first coverage/abstention repair
for the v195/v196 failure. It added an opt-in transition-value source-key
specificity gate that rejects low-specificity supported hits without changing
default runtime action selection. v197 validated the pinned v196 report digest
`7b3707b60575a14aee806b3989615582369f021f1ceced80e664e4fdd1794d90`, required
route `v197_slice_4_design_requires_coverage_or_abstention_repair_no_training`,
v196 mechanism
`artifact_low_specificity_feature_coverage_collapse_to_eat_with_miss_abstention`,
v195 report digest
`0d69540a3817b7c10f19b440bc63d9651b26a5c9d4d199f55b48253e2edcde`, and v195
artifact digest
`9cc3df4b5d87af5dd4dfb0e79f7282debbcc9f98ba7393f42a45d360e46c4470`. With the
v197 gate enabled, diagnostic replay reduced override-applied share from
`0.782944` to `0.0`, rejected `7681` low-specificity decisions, kept heuristic
action-source count at `0`, had no broad alive/birth regressions, and kept
dominant requested action `eat` under the cap at share `0.403329`; however,
exact/high-specificity hit share stayed `0.0` and `carrion_only@120` terminal
survivors stayed `0`. v197 route logic requires real replay provenance for any
future slice-4 route; mock or override diagnostics fail closed to coverage or
model-capacity repair. v197 wrote
`output/mind/mind-v3-v197-carrion-survivor-continuation-coverage-abstention-repair-design.json`
with exact digest
`8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a`.
Classification:
`m3_carrion_survivor_continuation_v197_coverage_abstention_repair_low_specificity_collapse_blocked_routes_to_coverage_or_model_capacity_no_training`.
It recommended exactly
`v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training`.
Lifecycle stayed closed: no training, no training artifact, no slice-4
consumption, no runtime artifact, no runtime action-selection change, no gate
relaxation, no support generation or expansion, and no promotion. The v197
report is durable after
`gdrive:evolution-sim-backups/archives/20260614T180236Z-v197-coverage-abstention-repair-design.tar.zst`
with archive SHA256
`f11dd5c9cd3eea3f30c2854320a435035277b74b4792f008cd687235ad2b2b40` and
`rclone check` verification of `0` differences and `1` matching file.
v198 then completed the no-training high-specificity coverage gap /
model-capacity repair contract before any slice-4 training. It validated the
pinned v197 report digest
`8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a`,
required route
`v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training`,
inherited v196/v195 pins, v197 real replay provenance, and closed lifecycle
flags. The v198 probe loaded the frozen v195 artifact and ran real replay over
the same broad and `carrion_only@120` diagnostic surfaces with transition-value
action override disabled, so default runtime action selection stayed unchanged.
The artifact contains `7457` high-specificity feature keys, but live replay had
high-specificity key presence in only `430/13634` decisions
(`0.031539` share), high-specificity complete-for-current-valid-action share
`0.0`, observed-support-floor share `0.0`, and selected high-specificity source
share `0.0`. The high-specificity candidate breakdown found `694` present
high-specificity key/category occurrences and all `694` were
present-but-action-incomplete; imputed, support-floor-only, and unclear-best
high-specificity blockers were `0`. Low-specificity collapse remained blocked
with `7681` low-specificity would-be rejections, heuristic action-source count
`0`, no broad or carrion alive/birth regressions, dominant requested action
`eat` at share `0.403403`, and `carrion_only@120` terminal survivors still `0`.
v198 wrote
`output/mind/mind-v3-v198-carrion-survivor-continuation-high-specificity-coverage-or-model-capacity-repair.json`
with exact digest
`91d4132c07ea18b0603626d4c080bdfed01f39677594737cda3ca0233fd95af8`.
Classification:
`m3_carrion_survivor_continuation_v198_high_specificity_action_complete_or_support_floor_gap_routes_to_contract_audit_no_training`.
It recommended exactly
`v199_high_specificity_action_complete_contract_audit_no_training`.
Lifecycle stayed closed: no training, no fit, no training artifact, no slice-4
start or consumption, no runtime artifact, no runtime integration, no runtime
policy or action-selection change, no gate relaxation, no support generation or
expansion, and no promotion. The v198 report is durable after
`gdrive:evolution-sim-backups/archives/20260614T191500Z-v198-high-specificity-coverage-or-model-capacity-repair.tar.zst`
with archive SHA256
`6d9627c305e56d9083861123e4114635c30c1ae2d939d499026fa73d8f63dbcf` and
`rclone check` verification of `0` differences and `1` matching file.
v199 then completed the no-training high-specificity action-complete contract
audit. It validated the pinned v198 report digest
`91d4132c07ea18b0603626d4c080bdfed01f39677594737cda3ca0233fd95af8`,
classification
`m3_carrion_survivor_continuation_v198_high_specificity_action_complete_or_support_floor_gap_routes_to_contract_audit_no_training`,
route `v199_high_specificity_action_complete_contract_audit_no_training`,
primary blocker `high_specificity_action_incomplete`, and closed v198 lifecycle
flags. Because v198 did not persist per-key/per-action action coverage, v199
reran only the required real diagnostic replay with transition-value action
override disabled. It evaluated `54536` high-specificity candidate key/category
occurrences: `53842` were absent, `694` were present, and all `694` present
occurrences were present-but-action-incomplete. Present-complete
observed-support-floor failures were `0`, heuristic action-source count was
`0`, runtime action-selection changes were `0`, and support generation/
expansion stayed closed. Missing current-valid action counts across present
high-specificity keys were `eat=628`, `move_west=475`, `move_south=469`,
`move_east=339`, `stay=333`, `move_north=245`, `attack_west=80`,
`attack_east=76`, `drink=65`, and `attack_north=4`. v199 wrote
`output/mind/mind-v3-v199-carrion-survivor-continuation-high-specificity-action-complete-contract-audit.json`
with exact digest
`d1111534a01d8cd36d6a8f309b6b05eb28f829c382e2cba6da1f58e011b4d04c`.
Classification:
`m3_carrion_survivor_continuation_v199_high_specificity_action_complete_gap_confirmed_routes_to_repair_design_no_training`.
It recommended exactly
`v200_high_specificity_action_complete_contract_repair_design_no_training`.
Lifecycle stayed closed: no training, no fit, no training artifact, no slice-4
start or consumption, no runtime artifact, no runtime integration, no runtime
policy or action-selection change, no gate relaxation, no support generation or
expansion, and no promotion. The v199 report is durable after
`gdrive:evolution-sim-backups/archives/20260614T195931Z-v199-high-specificity-action-complete-contract-audit.tar.zst`
with archive SHA256
`38704355bb89da1a9a12c73284baeaa3f6a9f4984ea7d101bf8d249b92e58bdb` and
`rclone check` verification of `0` differences and `1` matching file.
v200 then completed the no-training high-specificity action-complete contract
repair design. It directly validated the pinned v199 report digest
`d1111534a01d8cd36d6a8f309b6b05eb28f829c382e2cba6da1f58e011b4d04c`,
classification
`m3_carrion_survivor_continuation_v199_high_specificity_action_complete_gap_confirmed_routes_to_repair_design_no_training`,
and route
`v200_high_specificity_action_complete_contract_repair_design_no_training`;
directly validated the pinned v198 report digest
`91d4132c07ea18b0603626d4c080bdfed01f39677594737cda3ca0233fd95af8`; and
validated v197 through v198's source-pin validation, including v197 report
digest `8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a`
and route
`v198_high_specificity_coverage_or_model_capacity_repair_before_slice_4_training_no_training`.
v200 did not rerun replay: it audited the v199 report and frozen v195 artifact
metadata. Facts were consistent: `54536` high-specificity candidates evaluated,
`53842` absent, `694` present, all `694` present occurrences
present-but-action-incomplete, and present-complete support-floor failures `0`.
It selected repair class
`source_contract_audit_for_missing_current_valid_actions`, explicitly rejected
existing-artifact selection/probe-only repair, and marked lower-specificity
imputation, default-stay fallback, mock provenance, blind support expansion, and
gate relaxation as unsafe/non-authorizing. The future dataset/harness contract
must prove public observation/mask identity, high-specific key, current-valid
actions, observed support count per action, no private/fixture/seed leakage,
real replay provenance, and exact digest pins before any future specificity-gated
slice-4 route can even be considered. v200 wrote
`output/mind/mind-v3-v200-carrion-survivor-continuation-high-specificity-action-complete-contract-repair-design.json`
with exact digest
`71d8959ac4bdb73a45c888e159ea032ac3a5064352c890eee9914356bbb3184c`.
Classification:
`m3_carrion_survivor_continuation_v200_existing_artifact_action_incomplete_routes_to_source_contract_audit_no_training`.
It recommended exactly
`v201_high_specificity_action_complete_source_contract_audit_no_training`.
Lifecycle stayed closed: no training, no fit, no training artifact, no slice-4
start or consumption, no runtime artifact, no runtime integration, no runtime
policy or action-selection change, no gate relaxation, no support generation or
expansion, and no promotion. The v200 report is durable after
`gdrive:evolution-sim-backups/archives/20260614T202942Z-v200-high-specificity-action-complete-contract-repair-design-qa-repair.tar.zst`
with archive SHA256
`62cb570dd26ffc9e3c8f5062531b0de6ac955cf27eb3d01a3334e84ade60f1cd` and
`rclone check` verification of `0` differences and `1` matching file.
v201 then completed the no-training high-specificity action-complete source
contract audit. It directly validated the durable v200/v199/v198 exact digests
and validated v197 through v198's source-pin validation. v201 audited the v199
candidate-key rows against the frozen v195 artifact utility table without
replay, support generation, dataset mutation, or training. The audit confirmed
`2714` missing current-valid action demands across present high-specific keys,
same high-specific action support count `0`, same high-specific support-floor
failures `0`, and lower-specificity-only support for `2362` demands
(`0.870302` share). Missing actions absent at both the same high-specific key
and known lower-specificity keys accounted for `352` demands. Because
lower-specificity-only evidence dominates and grafting/imputation remains
unsafe and non-authorizing, v201 selected
`v202_public_masked_model_capacity_harness_contract_no_training` rather than any
slice-4 training route. v201 wrote
`output/mind/mind-v3-v201-carrion-survivor-continuation-high-specificity-action-complete-source-contract-audit.json`
with exact digest
`ed401b7a4e5ef5ae572040d6a9973e23b51951f2bd273e48797c094132af94d2`.
Classification:
`m3_carrion_survivor_continuation_v201_lower_specificity_only_evidence_dominates_routes_to_public_masked_model_capacity_harness_contract_no_training`.
Lifecycle stayed closed: no training, no fit, no training artifact, no slice-4
start or consumption, no runtime artifact, no runtime integration, no runtime
policy or action-selection change, no gate relaxation, no support generation or
expansion, no dataset mutation, and no promotion. The v201 report is durable
after
`gdrive:evolution-sim-backups/archives/20260614T205238Z-v201-high-specificity-action-complete-source-contract-audit.tar.zst`
with archive SHA256
`8502ba0adb72b8052498fb1ce630dc73a0baa06c02c415af0443a1c7e97d1162` and
`rclone check` verification of `0` differences and `1` matching file.
v202 then completed the no-training public masked model-capacity harness
contract. It directly validated the durable v201/v200/v199/v198 exact digests
and validated v197 through v198's source-pin validation. It preserved the v201
blocker facts: `2714` missing current-valid action demands, same high-specific
action support `0`, same high-specific below-floor support `0`,
lower-specificity-only demands `2362` (`0.870302` share), absent same/lower
known support `352`, candidate key action coverage count `215`, and artifact
feature key count `8575`. The contract defines only public runtime inputs
(`observation_input`, current `action_mask`, and deterministic public
history/context already available at runtime), forbids seed/fixture identity,
private world state, future outcomes, held-out labels, provenance/source paths,
support-count oracles, and train/test split identity at inference, and requires
current-action-mask-constrained argmax/sampling with fail-closed abstention for
invalid predictions. It also records deterministic replay/split requirements,
future artifact metadata, per-seed broad and carrion gates, leakage checks, and
explicit rejection of lower-specificity grafting as a runtime substitute for
high-specific current-valid support. v202 wrote
`output/mind/mind-v3-v202-carrion-survivor-continuation-public-masked-model-capacity-harness-contract.json`
with exact digest
`5776fd258e7a6ae5c2b650c88f868c74cbfa0fc13c76a0f29fc23e6ab96a1a8a`.
Classification:
`m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_ready_for_scaffold_no_training`.
It recommended exactly
`v203_public_masked_model_capacity_harness_scaffold_no_training`, not training
or support expansion. Lifecycle stayed closed: no training start, no slice-4
consumption, no training artifact, no runtime artifact or integration, no
runtime/default policy or action-selection change, no support generation or
expansion, no dataset mutation, no gate relaxation, and no promotion. The v202
report is durable after
`gdrive:evolution-sim-backups/archives/20260614T211415Z-v202-public-masked-model-capacity-harness-contract.tar.zst`
with archive SHA256
`285f2f2c68547dc46c5c3cc095282591686fbdf82f4d3ff2d2b6f87a9b25befc` and
`rclone check` verification of `0` differences and `1` matching file. The
campaign budget remains 3/10.

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
semantics or generating support. v193 reran targeted support expansion under
that repaired contract and found `6/6` repaired-contract support under the
action-share cap, but still did not authorize slice 3 directly. v194 audited
that repaired-contract support into a compact gitignored dataset with digest
`dd0061fae106fbe9c101c426357b3f78c9b5c4ed5989011baaf9a3cc514c2ffc` and
recommended the future explicit opt-in slice-3 training route. v195 consumed
slice 3/10 and failed shadow acceptance on zero carrion terminal survivors,
dominant requested-action share `0.9422`, and broad alive/birth regressions.
v196 diagnosed the new blocker as low-specificity artifact coverage collapse to
`eat` with miss abstention, not label imbalance alone and not a default `stay`
fallback on misses. v197 blocked that known low-specificity override collapse
but proved exact/high-specificity coverage was still too low for blind slice-4
training. v198 then showed the remaining high-specificity blocker is not another
low-specificity collapse: high-specificity live keys exist but are incomplete for
current valid actions. The next same-lane route is
`v199_high_specificity_action_complete_contract_audit_no_training`,
not runtime integration, not promotion, not gate relaxation, not slice-4
training, and not another old feasibility, counterfactual, IQL, scorer, blind
support expansion, or v180/v186 rerun. The campaign budget remains 3/10.

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

### RALPH orchestration

Use `.agents/skills/ralph-orchestration-loop/SKILL.md` and
`docs/agents/ralph-orchestration-loop.md` when acting as the second brain,
prompt manager, orchestrator, or cross-session QA coordinator.
