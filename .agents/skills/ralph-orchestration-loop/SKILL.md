---
name: ralph-orchestration-loop
description: Use when acting as the evolution-sim second brain, prompt manager, project orchestrator, or RALPH loop that polls Codex sessions, dispatches specialists, coordinates QA, preserves git/artifact durability, or prepares the next bounded route.
---

# RALPH Orchestration Loop

RALPH means Reconcile, Authorize, Launch, Poll, Hold/Handoff. It is a
coordination loop, not permission to start experiments.

## Required Inputs

Read these before steering another session:

- Latest `/tmp/evolution-sim-*-handoff*.md`, if supplied.
- `AGENTS.md`
- `docs/agents/ralph-orchestration-loop.md`
- Current route section in `docs/mind-v3-autonomous-evolution.md`
- `git status --short --branch --untracked-files=all`

Also load `evolution-sim-work`, `enki-coordinator-loop`, and
`mind-v3-autonomous-loop` when Mind v3 is in scope.

## The Loop

1. Reconcile facts.
   Verify branch, HEAD, origin ref, upstream, staged files, dirty files,
   `git ls-files output/mind`, latest report digest, and active thread state.
   Treat handoffs and other agents as hypotheses until local facts agree.

2. Authorize the route.
   Name the exact route, lifecycle type, allowed write scope, forbidden work,
   expected report/digest, validation ladder, and stop condition. If the route
   is not explicitly authorized, stay read-only.

3. Launch specialists only when useful.
   Use read-only research/QA agents freely after user authorization for
   delegation. Use write-capable workers only with disjoint write sets or
   isolated worktrees. Do not launch remote trainer work until source is pushed
   and trainer hygiene passes.

4. Poll with cadence.
   Active interactive thread: wait 10-15 seconds. Local test/build: 30-60
   seconds. Remote trainer or long sweep: 2-5 minutes. Idle or blocked thread:
   stop polling and send a steering/hold prompt. Do not busy-loop.

5. Hold or hand off.
   Stop on drift, stale prompts, unexpected dirty files, failed source/digest
   checks, missing route authority, context degradation, or completed durable
   source slice. Write handoffs under `/tmp`, not the repo.

## Specialist Roles

- Orchestrator: route decision, prompt quality, reconciliation, no edits unless
  explicitly assigned.
- Coder: one bounded implementation slice; no commits/pushes unless separately
  authorized.
- Researcher: external/source research, non-authorizing recommendations.
- QA analyst: independent diff, tests, digest, dirty-tree, and stop-rule review.
- AI/ML engineer: dataset/model/trainer design and hygiene; no training unless
  route explicitly authorizes it.
- Architect: interface and contract design; no runtime behavior change by
  default.

## Prompt Contract

Every dispatch prompt must include:

- Branch, route, upstream report/dataset digests, and lifecycle type.
- Allowed files/systems and explicit forbidden actions.
- Dirty-tree handling rules and whether existing dirt is user-owned.
- Validation commands and exact report/digest expectations.
- Final response fields and stop/hold instructions.

Use `references/prompt-templates.md` for copyable prompt skeletons.

## Stop Rules

Hold immediately before staging, committing, pushing, resetting, cleaning,
training, consuming a slice, changing runtime/default policy behavior,
generating support, mutating datasets, relaxing gates, or promoting unless the
current user prompt explicitly authorizes that exact action.
