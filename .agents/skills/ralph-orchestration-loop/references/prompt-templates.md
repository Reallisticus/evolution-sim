# RALPH Prompt Templates

Use these as starting points. Replace bracketed fields before sending.

## Read-Only Orchestrator Memo

```text
Second-brain request, read-only. Use [handoff path], AGENTS.md,
docs/mind-v3-autonomous-evolution.md, docs/agents/ralph-orchestration-loop.md,
and local git status as inputs.

Do not edit, stage, commit, push, set upstream, reset, clean, start [route],
spawn workers, train, mutate datasets, generate support, use remote trainer,
relax gates, or promote.

Report:
1. branch, HEAD, origin, upstream, staged area, output/mind tracking;
2. dirty files and ownership;
3. latest authoritative route and prohibitions;
4. whether a coder is needed;
5. bounded next coder prompt if and only if [route] is explicitly authorized;
6. non-authorizing research gaps;
7. stop/hold status.
```

## Coder Dispatch

```text
Implement [exact route] on branch [branch].

Lifecycle: [no-training scaffold | diagnostics-only | audit-only | explicit
training slice]. This prompt authorizes only [scope].

Must validate upstream pins:
- [report path] exact digest [digest]
- [dataset/artifact path] digest [digest]

Allowed writes:
- [files or modules]

Forbidden:
- training unless lifecycle says explicit training;
- slice consumption unless explicitly named;
- runtime/default policy behavior changes;
- support generation or expansion;
- dataset mutation;
- gate relaxation;
- promotion;
- remote trainer;
- staging output/mind;
- touching user-owned dirty files [list].

Before editing, report the files likely involved and validation plan. After
editing, run [commands]. Final answer must include files changed, behavior or
contract changed, validation results, lifecycle flags, report/digest paths, and
next route. Do not commit or push unless separately authorized.
```

## Steering Hold

```text
Second-brain steering update. Hold before any staging, commit, push, reset,
clean, further edits, training, runtime change, support generation, dataset
mutation, gate relaxation, or promotion.

Run only:
- git status --short --branch --untracked-files=all
- git diff --name-only
- git diff --cached --name-only

Report what changed, what is staged, and current stop/hold status. Wait for
second-brain/user confirmation before proceeding.
```

## Independent QA

```text
Read-only QA review for [route or diff]. Do not edit files.

Verify:
- git status and staged area;
- diff scope against the requested route;
- forbidden actions/lifecycle flags;
- report JSON parse and exact digest, if present;
- focused tests and git diff --check results;
- whether output/mind remains untracked;
- remaining blockers.

Return findings first, ordered by severity, with file/line references where
applicable. If no issues, say so and name residual test gaps.
```
