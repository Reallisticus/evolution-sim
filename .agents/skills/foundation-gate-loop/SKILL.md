---
name: foundation-gate-loop
description: "Use for Foundation-layer work in evolution-sim: ecology readiness, trophic durability, hydrology/refuge/hazard/carrion semantics, config validation, replay invariants, release-gate behavior, summary-only/full-replay separation, and simulator benchmark reliability."
---

# Foundation Gate Loop

Use this skill when work affects the Foundation measurement boundary used by
Mind v1/v2 safety floors and Mind v3 autonomous-controller experiments.

## Objective

Foundation is the current compatibility and safety boundary. The world must keep
durable ecological pressure, and the project must inspect, replay, gate, and
benchmark that pressure reproducibly.

Do not route model-training work through this skill. If a Mind experiment exposes
Foundation ambiguity, close the relevant Foundation contract first, then return
to the Mind workflow.

## Inspect First

Read the smallest relevant set:

- `docs/evolution-simulator-blueprint.md`
- `docs/foundation-readiness-audit.md`
- `docs/mind-readiness-audit-2026-04-27.md`
- `docs/senior-developer-onboarding.md`
- `docs/two-gate-recovery-plan.md`
- `docs/benchmark-protocol.md`
- relevant runtime modules under `python/evolution_sim/env/`
- relevant CLI gate/evaluation/benchmark files under `python/evolution_sim/cli/`
- relevant tests under `python/tests/`

## Foundation Invariants

- Full replay output is the compatibility contract.
- `SUMMARY_ONLY` must not build replay payloads, viewer payloads, frame captures, or full replay taxonomy surfaces unless explicitly required by a compact compatibility probe.
- Long sweeps belong behind opt-in release or benchmark commands.
- Every long check needs a compact contract equivalent for regular development.
- Gates must fail on concrete blockers, not hide failures as warnings.
- Benchmark claims need the same machine profile before and after.
- Config errors should fail at the config boundary with clear messages.
- Viewer and replay semantics must keep hydrology, refuge, hazard, carcass, combat, trophic role, species, and ecology meanings distinct.

## Work Loop

1. Define the readiness question.
   - What Foundation behavior or invariant is being proved?
   - Is this ecology, replay compatibility, config validity, gate logic, or performance?
2. Reproduce or measure the current state.
   - Use the shortest seed/tick profile that demonstrates the issue.
   - For ecological claims, record seeds, ticks, mode, and metrics.
3. Add or update the narrowest test or gate assertion.
   - Prefer deterministic public interfaces.
   - Use compact full-replay tests for replay/viewer contracts.
   - Use summary-only for long multi-seed viability checks.
4. Make the minimal implementation change.
5. Validate with targeted checks first, then gates.
6. If behavior changes ecology, compare before/after metrics and explain the tradeoff.

## Preferred Commands

Fast local:

```bash
npm run sim:test
npm run sim:golden:quick
npm run sim:gate:quick
npm run sim:bench:quick
git diff --check
```

Release boundary:

```bash
npm run sim:golden
npm run sim:test:full
npm run sim:gate:release
npm run sim:bench
```

Viewer compatibility:

```bash
npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json
REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke
```

## Report

Include:

- readiness question answered
- seeds/ticks/mode used
- metrics before/after when applicable
- tests or gates changed
- validation commands run
- remaining risk to Foundation or Mind measurement integrity
