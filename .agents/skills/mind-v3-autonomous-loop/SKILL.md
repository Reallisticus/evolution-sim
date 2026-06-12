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

Current durability constraint: the v137-v176 experiment chain has been
committed, pushed, and backed up, but the same failure mode must not recur.
Because the RTX trainer pulls committed source, do not launch new Mind v3
experiments or trainer jobs while the current slice's source, package
entrypoints, tests, ledger entries, or digest-referenced artifacts are still
local-only. A docs-only remediation pass or explicit git/backlog triage is
allowed.

Strategic handoff constraint: thread-local ledgers and attachments are not
durable context. A fresh coder should not be expected to know them unless the
dispatch prompt includes the relevant content or points to a committed summary
in `AGENTS.md` or `docs/mind-v3-autonomous-evolution.md`.

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
Current same-lane work is narrower:

- v180 has already consumed the durable pinned v178 authorization and run the
  first opt-in transition-row policy training slice;
- v180 failed shadow acceptance with zero `carrion_only@120` terminal survivors
  and broad per-seed alive/birth regressions, so it is not promotion evidence
  and authorizes no runtime integration, runtime action-selection change, gate
  relaxation, or promotion;
- v180 counts as slice 1 of the 10-slice carrion campaign budget. Its report
  and policy artifact are durable only after the recorded backup
  `gdrive:evolution-sim-backups/archives/20260611T110135Z-v180-transition-row-policy-training.tar.zst`
  with archive SHA256
  `e9c95f424e0827a7ed25f7de523fc61e9fe38d6062861466a81f1149671aac59`;
- v181 has already run as diagnostics-only failure-response work. It validated
  the pinned v180 report, artifact, and dataset, diagnosed sparse imputed
  transition-value support plus broad `eat` overrides and carrion no-coverage,
  and consumed no second training slice. Its report exact digest is
  `a89745e71daf8c6cc1651ad960a3098776509bca6e7cda84ca259fdbc57f5751` and is
  durable after
  `gdrive:evolution-sim-backups/archives/20260611T143628Z-v181-v180-failure-response-autopsy.tar.zst`
  with archive SHA256
  `a2b03e862ee095562c21482afcf0b1057a9ebb8f8b0b0d0f08e973dd884b1fd6`;
- v182 has already run as diagnostics-only failure-response design. It added
  imputed-valid-action and observed-support-floor diagnostics, made opt-in
  transition-value overrides abstain on imputed or below-floor currently valid
  action scores, and consumed no second training slice. Its report exact digest
  is `3ceb89524fbcddf2e9553fa06d932c1812d1c6a1f7d7f9f19c9237c34d22f51b` and is
  durable after
  `gdrive:evolution-sim-backups/archives/20260611T165440Z-v182-imputed-abstention-design.tar.zst`
  with archive SHA256
  `d53401e5096d4c616a26691faa555ba53ffbdf84b4767061ff09d820340faf54`;
- v183 has already run as diagnostics-only exact transition-support expansion.
  It validated the pinned v182/v181/v180/v179 evidence, wrote report digest
  `7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, wrote a
  `144`-row expanded dataset with digest
  `e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and
  kept training/runtime/promotion/slice-2 lifecycle flags closed. It moved
  carrion target coverage from `0` to `18` materialized states and broad seed
  `19` support-hole coverage to `6` materialized states. Its report, dataset,
  and source trajectories are durable after
  `gdrive:evolution-sim-backups/archives/20260611T192709Z-v183-exact-transition-support-expansion.tar.zst`
  with archive SHA256
  `b4e57537142303e622656c8eb51fde114a9452e20641a85cfdb4d742b425eb5c`;
- v184 has already run as a diagnostics-only fresh v178-style audit of the
  canonical v183 expanded dataset. It validated the v183 source report digest
  `7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da`, the
  v183 dataset digest
  `e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef`, and the
  pinned v182/v181/v180/v179 evidence, but target audit failed with `14`
  failures across `7` attack-forced rows resolving to `stay` with
  `current_resolution_action_valid=false`. Its report exact digest is
  `ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa`; it is
  durable after
  `gdrive:evolution-sim-backups/archives/20260612T091934Z-v184-v183-transition-row-dataset-audit.tar.zst`
  with archive SHA256
  `057d34c88f8a5da0c4ad98d87e4451caa043b7ee53c2af1037aa231dc5bcf382`;
- v185 has already repaired the v183 target-resolution blocker as
  diagnostics-only work. It strict-filtered the `7` invalid rows without
  backfill, leaving `137` rows, `24` branches, `6` seeds, and `9` forced
  actions. The repair report exact digest is
  `327586651948e13670cf15285472a934c8cd0a8fc9784d2dcdb092d102c46486`, the
  repaired dataset digest is
  `532817eb68cebf34cfb27f8abbbd86631cb142e03127661286d88d5154320f51`, and
  the paired repaired-dataset audit exact digest is
  `abd8c06733373b441c337187191cb04d3755968335b97f2fbcb350f550db8a50`. The
  repaired audit passed the v178-style source, schema, leakage, identity,
  action-mask, observation, target, and default-support checks and authorized
  only the future explicit route
  `v186_transition_row_policy_training_slice_2_opt_in`. v185 trained nothing,
  spent no slice 2, created no runtime artifact, changed no runtime action
  selection, and authorized no promotion. Its artifacts are durable after
  `gdrive:evolution-sim-backups/archives/20260612T131407Z-v185-v183-target-resolution-repair.tar.zst`
  with archive SHA256
  `c64aa17964462a4bbbc71fc83b98779213c50e26b714aedffe8cb7b42b9b5d2a`;
- the next useful lane is `v186_transition_row_policy_training_slice_2_opt_in`
  only if explicitly requested, not another support expansion by default and
  not a rerun of the first training slice;
- CLI `--min-*` support overrides are diagnostic only and must not authorize
  training routes.

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
- carrion fixture milestone: terminal alive greater than zero on
  `carrion_only@120`; blocker-count reduction is diagnostic only.

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

Historical strict candidate slice. Do not treat this labeled-IQL command, the
v179 audit route, a rerun of v180, or the already-completed v181/v182/v183/v184
diagnostics as the current next route; v180 already spent the first opt-in
transition-row training slice and v181/v182/v183/v184/v185 consumed no second
slice. Future same-lane work should use the v185 repaired dataset only through
an explicitly requested v186 slice-2 training command:

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
