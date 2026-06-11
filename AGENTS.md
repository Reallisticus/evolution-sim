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
and `rclone check` verification of `0` differences and `2` matching files.
Future work should not route back to v179/v180 authorization or rerun the first
transition-row training slice; after explicit direction, the next lane is v182
failure-response design that fixes imputed utility abstention/support coverage
before any new training slice.

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
was diagnostics-only and does not consume slice 2; the budget remains 1/10 until
a future explicitly authorized training slice runs.

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
