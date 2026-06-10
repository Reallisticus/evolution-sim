---
name: evolution-sim-work
description: Use for implementation work in this evolution-sim repository, including simulator fixes, replay/viewer changes, runtime refactors, tests, benchmarks, gates, and documentation updates. Follow the repo-specific loop instead of generic TypeScript, pnpm, or web-app workflows.
---

# Evolution Sim Work

Use this as the default implementation workflow for this repository.

## Start Here

1. Inspect the current repo context before editing:
   - `README.md`
   - `package.json`
   - relevant files in `docs/`
   - relevant source under `python/evolution_sim/`
   - relevant tests under `python/tests/`
   - recent audit docs when the task touches Foundation or Mind readiness
2. Check the working tree with `git status --short`.
3. Treat existing unrelated changes as user-owned. Do not revert them.
4. For Mind v3 work, read
   `docs/repository-audit-2026-06-10-remediation.md` before editing. The
   v137-v176 backlog has been committed and backed up; apply the same rule to
   new work by stopping before any experiment slice that depends on dirty-tree
   source, uncommitted entrypoints/tests/docs, or local-only digest-referenced
   artifacts unless the user explicitly asks for docs-only work, backlog triage,
   or a negative control.
5. For non-trivial work, state:
   - what you found
   - what files are likely involved
   - what behavior or invariant will change
   - what validation you will run

## Repo Shape

- Python simulator code lives in `python/evolution_sim/`.
- Tests live in `python/tests/`.
- Browser viewer code lives in `viewer/`.
- Docs and durable project specs live in `docs/`.
- `package.json` npm scripts are the preferred command entrypoints.
- Use `PYTHONHASHSEED=0` and `PYTHONPATH=python` when running raw simulator commands.

## Current Mind v3 Context

Mind v3 is the active autonomous-controller track. Read
`docs/mind-v3-autonomous-evolution.md` before touching Mind v3 policy,
dataset, training, gate, artifact, ledger, or acceptance code.

As of the v61-v63 closeout, the current single-observation IQL
coefficient/prior/extraction family is exhausted for promotion. The 2026-06-10
audit also closed the recent tiny support-archive / nearest-neighbor scorer loop
as strategically saturated until exact branch replay, transition-row support, or
scaled policy capacity changes the data path. Do not spend a new slice on
scalar IQL tuning, prior blends, action-share losses, risk extraction, global
actor-bias calibration, residual-threshold tuning, or another micro-archive
nearest-neighbor probe unless the user explicitly asks for a negative-control
probe.

Preserve these boundaries:

- no heuristic runtime action selection in autonomous Mind v3 candidates;
- no fixture identity or private world state in policy inputs;
- strict per-seed no-regression gates for alive and birth metrics;
- dominant action and heuristic-action gates remain hard blockers;
- controlled carrion fixture behavior is measured, not inferred from broad runs;
- every new policy state, artifact field, or data contract gets diagnostics and
  tests.

The likely next useful branch is rollout-context policy capacity derived from
public trajectory/action outcomes, or a genuinely distinct sequence, flow,
world-model, or archive-replay path.

Current durability constraint: source and evidence for v137-v176 must be made
durable before remote-trainer scale work can run. The trainer pulls committed
source from GitHub; it cannot execute local-only dirty-tree experiments.

## Remote Trainer

The project can use a locally configured remote trainer for accelerator-backed
training, long Mind gates, long seed sweeps, and heavier CPU-bound simulator
runs that would block local development. Keep host aliases, remote paths,
network details, account names, and hardware identifiers in private local
configuration. The generic workflow is documented in
`docs/remote-rtx-trainer.md`.

Default workflow:

1. Edit, review, and run quick checks on the Mac.
2. Push source changes to GitHub.
3. Fast-forward the trainer with `npm run trainer:pull`.
4. Start long trainer jobs in remote `tmux` with `npm run trainer -- start ...`.
5. Fetch only needed artifacts back with `npm run trainer -- fetch ...`.

Useful commands:

- `npm run trainer:status`
- `npm run trainer:doctor`
- `npm run trainer:gpu`
- `npm run trainer -- run npm run sim:bench:quick`
- `npm run trainer -- start mind-gate --pull -- npm run sim:mind:gate:extended`
- `npm run trainer -- start mind-v3 --pull -- <mind-v3 command>`
- `npm run trainer -- logs mind-gate`
- `npm run trainer -- attach mind-gate`

Known trainer caveats:

- `sim:golden:quick` has a known Mac/Linux last-bit viewer-float mismatch, so treat that as a repo determinism issue rather than a trainer setup failure.
- Do not commit private trainer topology, host aliases, usernames, remote paths, or network details.

## Implementation Rules

- Make the smallest coherent vertical change.
- Prefer existing simulator patterns and public test surfaces.
- Preserve deterministic behavior under fixed seeds.
- Preserve replay compatibility unless the task explicitly changes the contract.
- Keep `FULL_REPLAY` and `SUMMARY_ONLY` semantics distinct.
- Every world mechanic needs matching observability, metrics, and tests.
- Do not make long sweeps part of the normal fast loop.
- Do not add dependencies unless the repo clearly needs them.
- Avoid touching viewer, replay schema, runtime contracts, and goldens in the same change unless the task requires it.
- Do not hardcode a digest of a gitignored Mind artifact without documenting
  where that artifact is durably stored or how source-integrity should be
  overridden for regenerated, metric-equivalent artifacts.
- Do not add new imports from `python/evolution_sim/cli/` into `mind/` modules;
  shared experiment harness code belongs under `python/evolution_sim/mind/`.

## Test Discipline

Prefer focused behavior tests through public interfaces. Do not test by mutating cache internals or calling private invalidation helpers.

Use a narrow-to-broad validation ladder:

1. Targeted unit or integration test:
   - `PYTHONPATH=python PYTHONHASHSEED=0 python3 -m unittest python.tests.<module>`
2. Fast simulator suite:
   - `npm run sim:test`
3. Replay/golden compatibility when relevant:
   - `npm run sim:golden:quick`
   - `npm run sim:golden`
4. Foundation gate when behavior affects readiness:
   - `npm run sim:gate:quick`
5. Viewer smoke when replay/viewer surfaces change:
   - `npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json`
   - `REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke`
6. Benchmark checks when performance-sensitive:
   - `npm run sim:bench:quick`

Always run `git diff --check` before finalizing edits.

## Completion Report

Report:

- files changed
- behavior changed
- validation commands and results
- any checks skipped and why
- remaining uncertainty or next slice

For Mind v3 candidates, also report:

- artifact/report paths
- strict acceptance blockers
- dominant requested-action share and heuristic action-source count
- per-seed alive and birth deltas
- controlled fixture result
