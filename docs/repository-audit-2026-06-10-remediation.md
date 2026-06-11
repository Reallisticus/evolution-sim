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
  After explicit direction, next work is v182 failure-response design, not
  another pre-training audit and not a rerun of v180. CI/ML
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
reported `0` differences and `2` matching files. After explicit direction, the
next useful lane is v182 failure-response design that fixes imputed utility
abstention/support coverage before any new training slice.

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
backup are already durable, v181 failure-response should also be checked before
new work. If the v181 source/docs/report backup above are durable, do not rerun
the first opt-in transition-row training slice or route the failure into another
pre-training audit unless the user explicitly asks for regeneration after a
durability check. After explicit direction, the next focused strategy slice is
v182 failure-response design that fixes imputed utility abstention/support
coverage before any new training slice. Keep CI slices compact and do not turn
push CI into promotion evidence.

Keep strict Mind v3 gates hard. Do not create another scalar-tuning,
actor-bias, residual-threshold, or tiny nearest-neighbor micro-archive probe.
Promotion-style Mind v3 claims need fresh held-out seeds because
5,13,19,29,37,41 have been consumed as support/provenance for recent lanes.
```
