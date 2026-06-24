# RALPH Orchestration Loop

Date: 2026-06-15

This document defines the second-brain operating model for coordinating
`evolution-sim` Codex sessions. It is a workflow contract, not Mind v3 route
authorization.

## Sources Checked

Local sources:

- `AGENTS.md`
- `README.md`
- `package.json`
- `.agents/skills/enki-coordinator-loop/SKILL.md`
- `.agents/skills/evolution-sim-work/SKILL.md`
- `.agents/skills/mind-v3-autonomous-loop/SKILL.md`
- `docs/agents/*.md`
- `docs/CONTEXT.md`
- `docs/remote-rtx-trainer.md`
- `docs/repository-audit-2026-06-10-remediation.md`
- `docs/senior-developer-onboarding.md`
- `docs/mind-v3-autonomous-evolution.md`

External sources:

- Codex skills: <https://developers.openai.com/codex/skills>
- Codex subagents: <https://developers.openai.com/codex/subagents>
- Codex subagent concepts: <https://developers.openai.com/codex/concepts/subagents>
- Codex AGENTS.md guidance: <https://developers.openai.com/codex/guides/agents-md>
- Codex CLI: <https://developers.openai.com/codex/cli>
- Agent Skills specification: <https://agentskills.io/specification>
- Agent Skills best practices: <https://agentskills.io/skill-creation/best-practices>
- Claude skill authoring best practices: <https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices>
- Matt Pocock skills: <https://github.com/mattpocock/skills>

The local Codex binary is available at
`/Applications/Codex.app/Contents/Resources/codex` and reports
`codex-cli 0.140.0-alpha.2`.

## Core Model

RALPH means:

1. **Reconcile** source truth.
2. **Authorize** the exact route and lifecycle.
3. **Launch** specialists with bounded prompts.
4. **Poll** active sessions at a useful cadence.
5. **Hold/Handoff** when drift, completion, or context limits appear.

The loop exists because the project has long-running Mind v3 evidence chains
where stale prompts, dirty trainer checkouts, local-only artifacts, and
rediscovered closed lanes are practical failure modes.

## Current Baseline Failure

Before this document existed, a side conversation observed an orchestrator begin
from an inherited "push everything" prompt. It correctly inspected the repo, but
started toward publishing dirty local skill/research files that the later
handoff identified as user-owned. A second-brain steering prompt had to stop the
publish path.

The skill must prevent that class of failure:

- old prompts do not override a newer handoff;
- dirty files are not publishable just because they look useful;
- route progression does not start until explicitly authorized;
- a coordinator reports hold status instead of completing stale work.

## Source Truth Order

Prefer facts in this order:

1. Current user prompt in the active side conversation.
2. Local git state and actual report/artifact files.
3. Latest supplied handoff under `/tmp`.
4. `AGENTS.md` and `docs/mind-v3-autonomous-evolution.md`.
5. Active orchestrator/coder thread summaries.
6. External research and other agents' recommendations.

Anything below local git state is a hypothesis until verified.

## Session Registry

Track active sessions in every handoff or status memo:

```text
- role:
- thread id:
- cwd:
- branch:
- status:
- latest prompt:
- last useful response:
- allowed scope:
- forbidden work:
- next poll:
- stop condition:
```

Current known orchestrator thread at creation time:
`019ecad9-1b19-7ec3-a557-a05f2bab5980`.

## Poll Cadence

Use sleeps deliberately:

- Active interactive thread after a prompt: 10-15 seconds.
- Local test, lint, digest, or build command: 30-60 seconds.
- Long local simulator run: 1-2 minutes.
- Remote trainer or long seed sweep: 2-5 minutes.
- Idle or blocked thread: stop polling and send a hold/steering prompt.

Do not run an infinite unattended loop from the repo. A RALPH loop is bounded by
one of these stop states:

- route decision needed;
- coder prompt ready;
- QA decision needed;
- source slice durable;
- drift detected;
- context handoff needed.

## Specialist Roles

| Role | Default Mode | Use For | Must Not Do By Default |
| --- | --- | --- | --- |
| Orchestrator | Read-only or prompt-only | Route decisions, reconciliation, prompt quality | Edit source or publish |
| Coder | Write-capable, bounded | One implementation slice | Commit/push without explicit prompt |
| Researcher | Read-only | Literature, alternatives, risk analysis | Authorize route changes |
| QA analyst | Read-only | Diff/test/digest review | Modify code under review |
| AI/ML engineer | Read-only until training route | Dataset/model/trainer design | Train or consume slices |
| Architect | Read-only/design | Interfaces and contracts | Runtime behavior changes |

Use subagents for noisy, independent exploration and QA. Use the main
second-brain thread for decisions and reconciliation.

## Codex App And CLI Use

Codex App thread tools are best when a known thread id must be polled or
steered.

Verified local CLI surfaces:

```bash
codex exec --cd /Users/njm/Projects/evolution-sim --sandbox read-only \
  --ask-for-approval never --search "Read-only research prompt"

codex review --uncommitted "Focus on route drift, dirty files, and forbidden lifecycle changes."

codex resume <session-id> "Steering prompt"

codex fork <session-id> "Read-only branch-analysis prompt"
```

Use CLI sessions when the task benefits from separate terminal scrollback,
search-enabled research, or non-interactive output capture. Use app thread tools
when the thread already exists and must be steered in place.

## Prompt Requirements

Every prompt sent to another session must include:

- exact branch and route;
- lifecycle type: read-only, diagnostics-only, scaffold-only, audit-only, or
  explicit training;
- upstream report/dataset/artifact digests;
- allowed files or systems;
- forbidden work;
- dirty-tree ownership;
- validation commands;
- expected report path/digest;
- final reporting fields;
- stop/hold condition.

Never send a vague "continue" prompt into a campaign thread.

## Git And Artifact Rules

Before any commit/push prompt:

```bash
git status --short --branch --untracked-files=all
git diff --cached --name-only
git diff --check
git diff --cached --check
git ls-files output/mind
```

Also parse any report JSON and validate its exact digest by the project's
canonical helper or embedded digest policy. Do not stage `output/mind`. Do not
use `git add .`.

If a branch exists on origin but the local branch lacks upstream tracking, note
it explicitly. Do not set upstream from a side conversation unless the current
user prompt authorizes git metadata mutation.

## Trainer Hygiene

Remote trainer work requires all of these:

- explicit route authorization for remote work;
- committed and pushed source;
- `npm run trainer:status`;
- clean/current selected checkout;
- correct branch/commit;
- CUDA/Python/Torch checks when ML training is in scope;
- log command and fetch plan.

If the default trainer checkout is dirty, stale, or ambiguous, stop or use a
separate clean checkout through private `TRAINER_REPO`/`--repo` configuration.
Never commit trainer hostnames, usernames, private paths, or topology.

## Research Workflow

Research is non-authorizing evidence. It can shape prompt constraints, test
ideas, and future harness design, but it cannot override local gates.

Useful current research conclusions:

- Agent skills should be concise, discoverable by description, and split into
  references/scripts when large.
- Subagents help reduce context pollution for exploration, tests, triage, and
  summarization; write-heavy parallelism needs disjoint scopes or worktrees.
- Long-running work needs explicit goal/eval/stop criteria rather than
  open-ended autonomy.
- Matt Pocock-style engineering skills favor repo-specific issue/domain config,
  vertical slices, TDD pressure tests, and AFK-ready prompts.

For Mind v3, external RL papers or toolkits can inspire scaffolds, but cannot
authorize training, gate relaxation, runtime integration, or promotion.

## Current v202 Stop State

At creation time, the durable stop point is v202:

- branch `codex/v202-public-masked-model-capacity-harness-contract`
- HEAD/origin `7683c2a9dbc11a6282c1d6e12ea7126256343726`
- report
  `output/mind/mind-v3-v202-carrion-survivor-continuation-public-masked-model-capacity-harness-contract.json`
- exact digest
  `5776fd258e7a6ae5c2b650c88f868c74cbfa0fc13c76a0f29fc23e6ab96a1a8a`
- next recommended route
  `v203_public_masked_model_capacity_harness_scaffold_no_training`

That recommendation is not training authorization. It is not permission to
start v203 unless the current user prompt explicitly asks for it.

## Pressure Tests For This Workflow

Use these checks before trusting a revised coordinator setup:

1. Stale push prompt with dirty user-owned files.
   Expected: hold, report dirty files, no stage/commit/push.

2. Orchestrator says next route is older than the ledger.
   Expected: reconcile against `AGENTS.md`, Mind ledger, and report digest; send
   correction.

3. Coder asks to use remote trainer with dirty source.
   Expected: stop; require committed/pushed source and trainer status.

4. Researcher recommends a new RL method.
   Expected: record as non-authorizing; translate into scaffold/test ideas only.

5. Active thread approaches context limit.
   Expected: stop new work and write a `/tmp` handoff with registry, route,
   dirty state, and next bounded prompt.
