---
name: enki-coordinator-loop
description: "Use for automating the evolution-sim ENKI coordinator/planner/builder/reviewer development loop: project-wide repo audit, Foundation-to-viewer-to-Mind planning, contract-first implementation slices, validation routing, and long-running experiment handoffs without manual copy-paste between agents."
---

# Enki Coordinator Loop

Use this skill when a change is big enough that the user would otherwise run a
Coordinator, Planner, Coder, and Reviewer as separate manual chats. Keep the loop
inside one Codex thread unless the user explicitly asks for separate artifacts.

This is a repo-specific coordinator skill. It routes into the project skills
`evolution-sim-work`, `foundation-gate-loop`, `mind-contract-loop`, and
`mind-v3-autonomous-loop` when their narrower trigger applies.

Use the installed Superpowers skills as the generic methodology baseline when
available: `superpowers:using-superpowers`, `superpowers:writing-plans`,
`superpowers:test-driven-development`, `superpowers:executing-plans`,
`superpowers:systematic-debugging`, and
`superpowers:verification-before-completion`. The repo-specific contracts in
this skill and the narrower evolution-sim skills override generic methodology
whenever they conflict.

## Core Rule

Do not ask the user to copy text from one role to another. The coordinator owns
the whole cycle:

1. Audit the actual repo state.
2. Produce the plan.
3. Execute the smallest coherent slice.
4. Review the diff and validation.
5. Report the next decision or stop rule.

If subagent tools are available and the user has allowed that style of work, the
coordinator may dispatch planner/reviewer/audit branches. The coordinator still
must reconcile their output against the local source tree before editing.

## Start With Source

Do not trust docs, commit messages, old plans, or previous session summaries as
authority. Treat them as hypotheses only.

Before non-trivial work, inspect:

- `git status --short`
- `package.json`
- `.github/workflows/ci.yml`
- `docs/repository-audit-2026-06-10-remediation.md` when Mind v3, CI,
  trainer, docs, or agent-loop direction is in scope
- the relevant source files under `python/evolution_sim/`
- the relevant tests under `python/tests/`
- the relevant viewer files under `viewer/` when replay or UI changes are in scope

Use `references/repo-grounding.md` for the current repo map and validation
surface, then refresh any file facts that matter to the task with `rg`, `sed`,
or targeted test runs.

## Route The Work

Use the narrowest applicable loop:

- Foundation/runtime/replay readiness: use `foundation-gate-loop`.
- Observation/action/trajectory/data-contract work: use `mind-contract-loop`.
- Mind v3 autonomous policy, rollout context, controlled fixtures, or training
  evidence: use `mind-v3-autonomous-loop`.
- General repo implementation with no narrower route: use
  `evolution-sim-work`.

If a task crosses boundaries, sequence it contract-first:

1. Foundation or runtime contract.
2. Mind interface or data contract.
3. Policy/training change.
4. Viewer/replay presentation.
5. Gates, docs, and ledgers.

For Mind v3 work after the 2026-06-10 audit, insert a durability checkpoint
before any new experiment: the coordinator must decide whether existing dirty
source and local-only `output/mind/` evidence block reproducibility or trainer
execution. If they do, route to backlog commit/artifact triage before policy
work.

For long-running Mind v3 campaigns, also insert a historical-evidence
checkpoint before dispatching a coder. The coordinator must turn durable prior
runs into actionable constraints: consumed slice budget, report and artifact
digests, routes already authorized or closed, seeds consumed as support, failure
mechanisms that were ruled in or out, and lanes that must not be repeated. The
coder prompt should carry those constraints explicitly so older v40/v60/v90 and
current v190+ work are used as accumulated evidence rather than rediscovered.

For RTX work, trainer hygiene is the coordinator's responsibility. Run the
trainer status check before dispatching remote work. If the default trainer
checkout is dirty, stale, or on the wrong branch, do not launch jobs there. Use
a separate clean checkout configured through private `TRAINER_REPO`/`--repo`
settings, or stop for explicit cleanup approval. Never hide trainer dirtiness in
the coder prompt.

For local validation/tooling gaps, fix narrow missing developer dependencies
instead of treating them as blockers. Prefer the project virtual environment or
another local/dev-scoped install, then report what was installed. Do not expand
the committed runtime dependency surface unless the task itself requires it.

## Coordinator Cycle

### 1. Intake

State the real objective in repo terms:

- behavior to change or evidence to produce;
- source boundary involved;
- contracts that must stay stable;
- likely tests or gates;
- stop conditions.

For unclear work, ask at most one blocking question. Otherwise proceed with a
reasonable narrow assumption and name it.

### 2. Audit

Inspect the exact files that can make the claim true or false. Prefer source and
tests over docs. For this repo, never skip the runtime/contract layer just
because the task sounds like a Mind or viewer task.

Use mechanical inventory for wide work:

```bash
rg --files -g '!output/**' -g '!node_modules/**' -g '!.venv/**'
git ls-files | awk 'BEGIN{FS="/"} /^python\/evolution_sim\//{src++} /^python\/tests\//{tests++} /^viewer\//{viewer++} END{print src, tests, viewer, NR}'
```

### 3. Plan

Write a short implementation plan only after the audit. A good plan names:

- files likely to change;
- tests to add or adjust first;
- validation commands;
- expected artifacts or reports;
- risks to replay compatibility, determinism, or gates.
- historical evidence being reused and the traps it prevents;
- trainer state, whether a clean checkout is available, and whether RTX work is
  allowed in this slice.

For learned-controller work, the plan must include the falsification metric
before any training command. For Mind v3, it must also state whether the
candidate uses a fresh promotion-heldout matrix or only a diagnostic matrix.

### 4. Build

Implement the smallest vertical slice that proves or falsifies the plan.

Rules:

- keep fixed-seed determinism;
- preserve `FULL_REPLAY` and `SUMMARY_ONLY` separation;
- avoid changing replay schemas, goldens, runtime mechanics, and viewer UI in
  one broad edit unless the task requires it;
- add tests through public behavior, not cache internals;
- do not relax gates to make an experiment pass;
- do not introduce hidden heuristic action selection in autonomous Mind v3.
- do not add new `mind/` imports from private CLI helper modules; extract shared
  harness code into `python/evolution_sim/mind/`.

### 5. Review

Review the diff as if it came from another agent:

- behavior regressions;
- schema/version drift;
- private world-state leaks into policy inputs;
- missing diagnostics or artifact metadata;
- tests that assert implementation details instead of contracts;
- viewer contract/generated-file staleness;
- long-run claims without seeds, ticks, modes, and report paths.

Run `git diff --check` before completion.

For dirty-tree remediation, review git actions separately from code changes.
Never commit, push, reset, clean, or delete artifacts unless the user explicitly
authorized that operation in the current task.

### 6. Report

Final reports must include:

- files changed;
- behavior or contract changed;
- validation commands and results;
- checks skipped and why;
- remaining blockers or the next useful slice.

For Mind v3 candidates, also include artifact/report paths, strict blockers,
dominant requested-action share, heuristic action-source count, per-seed
alive/birth deltas, and controlled fixture result.

## Automation Policy

The automated loop is conversational automation, not blind background work.

- Keep coordinator/planner/builder/reviewer handoffs in the same turn or thread.
- For long local commands, monitor output and summarize concrete status.
- For RTX or long seed sweeps, use the repo trainer workflow and report the job
  id, command, log command, artifact paths, and fetch command.
- Create app reminders, monitors, or recurring automations only when the user
  explicitly asks for a later wakeup or recurring check.

## Stop Rules

Stop and explain before editing if the requested route would:

- relax Foundation, Mind, or viewer gates for convenience;
- make autonomous Mind v3 depend on fixture identity or private
  `SimulationWorld` state;
- replace a deterministic public contract with an unversioned side channel;
- rewrite unrelated subsystems;
- delete or regenerate goldens without explicit approval;
- require secrets, production credentials, or destructive git commands.
