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
- The first strategy-reset implementation slice is complete: v177 exact branch
  replay expansion now emits compact transition rows with current/next public
  observations and action masks plus previous same-agent public context. CI/ML
  reproducibility now has compact push coverage plus an explicit
  NVIDIA-trainer validation path for the optional torch stack.

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

Status: v177 complete. The v176 recommendation,
`v177_exact_branch_replay_expansion_no_training`, now materializes exact branch
replay evidence and compact transition rows with `next_public_observation`,
`next_public_action_mask`, and `previous_same_agent_public_context`. The next
Mind v3 slice should audit the v177 transition-row dataset and decide the v178
data-support route before any transition/world-model, rollout-context, or
neural capacity work. Do not train another undersupported nearest-neighbor
scorer.

After source durability is restored, move scale/capacity work through the
remote trainer: vectorized branch replay, quality-diversity archive expansion,
or a neural/recurrent learner with source validation and strict held-out gates.
Make lane-level stop rules explicit before running long jobs.

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

Do not start a new Mind v3 experiment or train anything. First restore source
and evidence durability if the user has authorized git actions: inventory the
v137-v176 dirty tree, prepare reviewable commit slices, and make sure any
digest-referenced output/mind artifacts have a documented durable backup path.
If git actions are not authorized, stop after producing the exact commit plan.

If source durability, the habitat water contract fix, shared-harness
extraction, CI/ML reproducibility hardening, and v177 exact-branch-replay
transition-row support are already handled, take the next focused strategy
slice: v178 transition-row dataset audit. Keep CI slices compact and do not
turn push CI into promotion evidence.

Keep strict Mind v3 gates hard. Do not create another scalar-tuning,
actor-bias, residual-threshold, or tiny nearest-neighbor micro-archive probe.
Promotion-style Mind v3 claims need fresh held-out seeds because
5,13,19,29,37,41 have been consumed as support/provenance for recent lanes.
```
