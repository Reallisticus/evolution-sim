---
name: mind-v3-autonomous-loop
description: "Use for Mind v3 autonomous-controller work in evolution-sim: no-heuristic policy experiments, rollout-context policy capacity, carrion recovery, strict labeled IQL slices, sequence/flow/world-model branches, exact branch replay, controlled fixture gates, and Mind v3 experiment documentation."
---

# Mind v3 Autonomous Loop

Use this skill for current Mind v3 autonomous-controller work.

## Current Boundary

Mind v3 is the strategic track for reducing heuristic action selection. The
Foundation, replay/viewer contracts, trajectory data contract, and guarded Mind
v1/v2 gates are the safety floor.

As of the v61-v63 closeout, the current single-observation IQL
coefficient/prior/extraction family is exhausted for promotion. The 2026-06-10
full-repository audit also closes the recent tiny support-archive /
nearest-neighbor scorer loop as strategically saturated. Do not start a new
slice by tuning scalar IQL weights, prior blends, action-distribution loss, risk
extraction, global actor bias, residual thresholds, or another micro-archive
nearest-neighbor scorer unless the user explicitly asks for a negative-control
probe.

Current blocking constraint: the v137-v176 experiment chain and its
digest-referenced `output/mind/` evidence are not durable until committed,
pushed, and backed up. Because the RTX trainer pulls committed source, do not
launch new Mind v3 experiments or trainer jobs while that backlog is still
local-only. A docs-only remediation pass or explicit git/backlog triage is
allowed.

## Inspect First

Read the smallest relevant set:

- `AGENTS.md`
- `README.md`
- `docs/repository-audit-2026-06-10-remediation.md`
- `docs/mind-v3-autonomous-evolution.md`, especially the latest milestone
  sections and research direction
- `python/evolution_sim/cli/mind_v3_labeled_iql_slice.py`
- `python/evolution_sim/cli/mind_gate.py`
- `python/evolution_sim/mind/torch_trainer.py`
- `python/evolution_sim/mind/v3_policy.py`
- `python/evolution_sim/mind/v3_neural.py`
- `python/evolution_sim/mind/dataset.py`
- `python/evolution_sim/env/runtime/trajectory.py`
- relevant tests in `python/tests/`

Check `git status --short` before edits. Existing unrelated changes are
user-owned.

## Allowed Novelty

Novel controller ideas are welcome, but they must pass through repo contracts.
Good next directions include:

- exact branch replay expansion from v176 into transition rows containing
  `next_public_observation`, `next_public_action_mask`, and
  `previous_same_agent_public_context`;
- policy-owned rollout context or option memory derived from public trajectory
  rows and finalized outcomes;
- exact branch replay, Go-Explore-style archive replay, and robustification;
- sequence, flow, or compact world-model branches with serialized artifacts;
- diagnostics that falsify state aliasing, action collapse, or data-support gaps
  before expensive training.

Do not use:

- fixture identity as runtime policy input;
- private `SimulationWorld` reads from policy code;
- hidden heuristic action selection;
- relaxed acceptance gates to make a candidate pass;
- long sweeps as a substitute for compact contract tests.

## Work Loop

1. Define the hypothesis and first falsification metric.
2. Run the durability checkpoint: inspect `git status --short`, identify
   whether the work depends on local-only source or gitignored evidence, and
   stop for commit/artifact triage when the backlog blocks reproducibility.
3. Add or update the smallest diagnostic/report contract before heavy training.
4. Implement the smallest opt-in policy/data/artifact path.
5. Run focused tests and JSON parse checks before RTX work.
6. Before ending a completed source slice, make a Git durability decision:
   commit and push when authorized, or record the exact reason the slice is
   intentionally left local-only. Do not let code, tests, package scripts, or
   digest-bound ledger entries accumulate as an undocumented dirty backlog.
7. Use the RTX trainer only for CUDA or long seed/gate runs after source has
   been pushed.
8. Evaluate on the strict broad-plus-fixture matrix or on a freshly declared
   diagnostic matrix when prior held-out seeds have been consumed as support.
9. Document the exact command, artifact/report paths, blockers, and next stop
   rule in `docs/mind-v3-autonomous-evolution.md`.

## Acceptance Surface

For promotion-style Mind v3 candidates, keep these hard:

- broad held-out seeds must be uncontaminated by the candidate's training,
  selection, support, or provenance artifacts. The historical broad diagnostic
  seeds `5,13,19,29,37,41` are no longer clean promotion-heldout evidence for
  scorers trained or selected on recent v165+ support/provenance artifacts;
  mint and document a fresh promotion-heldout matrix before promotion claims;
- controlled `carrion_only` fixture seeds `13,19,29,37,41,43` at `120` ticks
  for the current carrion recovery boundary;
- dominant requested-action share `<= 0.50`;
- zero heuristic action-source count;
- no per-seed alive or birth regression versus the linear Mind v3 baseline;
- carrion fixture movement: terminal alive greater than zero or blocker count
  below the linear baseline.

Train-gate success alone is not promotion evidence.

## Preferred Commands

Focused local tests:

```bash
PYTHONPATH=python PYTHONHASHSEED=0 python3 -m unittest python.tests.<module>
git diff --check
```

Fast suite:

```bash
npm run sim:test
```

Strict candidate slice:

```bash
npm run sim:mind:v3:labeled-iql-slice -- \
  --candidate-artifact <artifact.json> \
  --enable-mind \
  --mind-runtime-mode autonomous \
  --seeds <fresh-promotion-heldout-seeds> \
  --ticks 120 \
  --fixture-names carrion_only \
  --fixture-seeds 13,19,29,37,41,43 \
  --fixture-ticks 120 \
  --output <slice-report.json>
```

Remote RTX pattern:

```bash
npm run trainer -- start mind-v3 --pull -- <command>
npm run trainer -- logs mind-v3
npm run trainer -- fetch <remote-artifact-or-report>
```

## Report

Include:

- files changed;
- behavior or contract changed;
- validation commands and results;
- artifact/report paths;
- strict acceptance blockers;
- dominant action share and heuristic action-source count;
- per-seed alive/birth deltas;
- controlled fixture result;
- why the next step is or is not worth pursuing.
