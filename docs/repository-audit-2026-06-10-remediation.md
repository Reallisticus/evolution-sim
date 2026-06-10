# Repository Audit Remediation Plan

Date: 2026-06-10

This document preserves the action plan from the full repository audit covering
Foundation, replay/viewer, Mind v1/v2, Mind v3, CLI, docs, tests, CI, and local
agent skills. The audit found no evidence that existing recorded results are
invalid. The Foundation measurement boundary is mostly sound. The urgent
failures are process and strategy failures around Mind v3 durability,
reproducibility, and experiment direction.

## Current State

- Last commit: `e7d7e3a` on 2026-05-28.
- Current dirty tree at audit intake: 7 modified tracked paths and 126
  untracked paths, including 45 CLI modules, 40 `mind/` modules, 40 tests, and
  `docs/versioning.md`.
- `output/` is 8.9 GB locally; `output/mind/` is 7.5 GB locally and contains
  digest-referenced evidence that is gitignored.
- CI continuously exercises Foundation and viewer checks, but the push-time
  Mind gate is a two-tick smoke with vacuous thresholds. Strict Mind gates run
  only in the expensive scheduled/manual job.

## Confirmed Serious Findings

1. `habitat_state_codes` encodes water tiles as habitat code `0` (`stable`)
   while sibling surfaces use the `-1` non-land sentinel. Counts exclude water,
   so the per-cell matrix and `habitat_state_counts` disagree, and the viewer
   can label water as stable habitat. This is a replay/viewer contract bug, not
   a taxonomy or simulation-evolution bug.
2. The v137-v176 Mind v3 chain exists only in the dirty working tree while the
   evidence artifacts behind hardcoded digests are gitignored. The chain cannot
   be reproduced on another machine without the exact local `output/mind/`
   state.
3. The uncommitted backlog blocks the prescribed remote-trainer path because
   the trainer workflow pulls committed GitHub source. The capacity/scale route
   cannot run until the source is committed and pushed.
4. Mind v3 strategy has drifted into small 1-nearest-neighbor and micro-archive
   diagnostics after the lane itself showed support starvation, feature
   aliasing, and tie collapse. The central carrion survival blocker has not
   moved materially.
5. `mind/` library modules import private helpers from the CLI
   `mind_v3_evaluate.py`. That makes a CLI file a live shared library and
   prevents safe refactoring of historical experiment records.

## Immediate Operating Rules

- Do not start a new Mind v3 experiment slice until the dirty-tree backlog is
  either committed and pushed or intentionally shelved with a documented reason.
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
- Treat the habitat water encoding fix as an explicit replay/viewer contract
  change requiring tests, contract notes, and golden regeneration.

## Action Order

### P0: Source And Evidence Durability

Inventory the dirty tree, then slice commits so CI, review, and the remote
trainer can see the v137-v176 work. The cleanest long-term shape is one commit
per experiment slice: CLI wrapper, `mind/` implementation, tests, `package.json`
script, and ledger entry together. If patch staging that many historical ledger
chunks is too costly, use reviewable lane ranges and record the compromise in
the commit message.

Back up `output/mind/` artifacts referenced by hardcoded digests before relying
on them from another machine. The backup location must not expose private host
paths or secrets, but reports should record enough provenance to fetch the
artifact again.

No commit, push, reset, clean, or destructive git command should be run without
explicit user authorization.

### P1: Strategy Reset

The next Mind v3 slice should execute the v176 recommendation:
`v177_exact_branch_replay_expansion_no_training`, focused on exact branch replay
and compact transition rows with `next_public_observation`,
`next_public_action_mask`, and `previous_same_agent_public_context`. The goal is
to produce data support for a transition/world-model or rollout-context policy,
not to train another undersupported nearest-neighbor scorer.

After source durability is restored, move scale/capacity work through the
remote trainer: vectorized branch replay, quality-diversity archive expansion,
or a neural/recurrent learner with source validation and strict held-out gates.
Make lane-level stop rules explicit before running long jobs.

### P1: Documentation And Agent Instructions

Keep `AGENTS.md`, `README.md`, onboarding, versioning, and local skills aligned
with this audit. Agent loops must include a durability checkpoint and must not
encode stale strategy anchors as the default path.

### P2: Habitat Water Contract Fix

Change habitat per-cell water encoding to the non-land sentinel, update viewer
handling, add a regression that matrix recounts match land-only counts, update
contract docs, regenerate generated viewer contracts if necessary, and
regenerate goldens in a dedicated contract-change commit.

### P2: Mind v3 Shared Harness Extraction

Move fixture construction, report aggregation, strict gate helpers, digest
validation, JSON rounding, and leakage scans out of
`python/evolution_sim/cli/mind_v3_evaluate.py` into a versioned shared module
under `python/evolution_sim/mind/`. Then update CLI wrappers to import from
that module. Historical experiment modules should not import private CLI
helpers.

### P2: CI And ML Reproducibility

Give push-time CI at least one non-vacuous Mind gate or focused Mind v3
contract check, and keep strict Mind gates on scheduled/manual lanes. Add a
torch-installed CI lane or explicit non-CI validation script so torch-dependent
tests stop silently skipping everywhere. Lock or otherwise record the
Mind-ML dependency environment used for trained artifacts.

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

If source durability is already handled, take the next focused code slice:
implement the habitat water-encoding contract fix as a dedicated
replay/viewer/golden change. Add a regression proving water tiles use the
non-land sentinel and land habitat counts still match the matrix recount, update
viewer/contract docs, regenerate goldens, and run the narrowest relevant tests
before broader replay/viewer checks.

Keep strict Mind v3 gates hard. Do not create another scalar-tuning,
actor-bias, residual-threshold, or tiny nearest-neighbor micro-archive probe.
Promotion-style Mind v3 claims need fresh held-out seeds because
5,13,19,29,37,41 have been consumed as support/provenance for recent lanes.
```
