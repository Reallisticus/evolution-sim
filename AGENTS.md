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
transition-row training slice, run more support expansion by default, or spend
slice 2 without explicit approval; the next same-lane route is
`v186_transition_row_policy_training_slice_2_opt_in`.

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
diagnostic transition-row evidence; none consume slice 2. The budget remains
1/10 until a future explicitly authorized training slice runs from the repaired
v185 dataset.

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
