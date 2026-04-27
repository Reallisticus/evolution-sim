# Foundation Readiness Audit

Date: 2026-04-24

This audit checks whether `Foundation` is ready to support `Mind v1`: a learned
controller whose objective is not to match scripted behavior, but to discover
ecologically useful behavior under world pressure.

## Verdict

Foundation is substantially stronger than the MVP and is close enough to run
release-gate validation, but it is not yet cleanly ready for Mind work.

The world now has multiple real pressure channels:

- deterministic terrain, fertility, moisture, heat, seasons, and disturbances
- food, hydration, vegetation depletion, shelter, recovery debt, and regrowth
- hydrology split into hard access and support/refuge semantics
- hazards, health, injury, healing, combat, fresh kills, carcasses, and decay
- genome-derived trophic roles and meat modes
- lineage, ecotype, durable replay taxonomy, species metrics, collapse events,
  and cross-run evaluation

The remaining blockers are mostly not world richness. They are Mind-readiness
interfaces and training data discipline.

## Evidence Checked

- Project docs:
  - `docs/evolution-simulator-blueprint.md`
  - `docs/mvp-simulation-spec.md`
  - `docs/replay-invariants.md`
  - `docs/two-gate-recovery-plan.md`
  - `docs/benchmark-protocol.md`
  - `docs/capability-layer-spec.md`
- Runtime modules:
  - `python/evolution_sim/env/world.py`
  - `python/evolution_sim/env/runtime/*.py`
  - `python/evolution_sim/env/taxonomy.py`
  - `python/evolution_sim/env/contracts.py`
  - `python/evolution_sim/config/schema.py`
  - `python/evolution_sim/genome/schema.py`
- Tooling:
  - `python/evolution_sim/cli/evaluate.py`
  - `python/evolution_sim/cli/foundation_gate.py`
  - `python/evolution_sim/cli/bench.py`
  - `python/evolution_sim/cli/golden_harness.py`
- Viewer:
  - `viewer/app.js`
  - `viewer/index.html`
  - `viewer/smoke.mjs`
- Tests:
  - 53 discovered test methods across simulator, taxonomy, runtime contracts,
    evaluator, and gate tests

Static checks found no simulator `TODO`, `FIXME`, stubs, or unused config/genome
fields by declared-name scan. `compileall` passed for `python/evolution_sim` and
`python/tests`.

## Current Gate State

The quick gate passed:

- `npm run sim:gate:quick`
- status: `pass`
- blockers: none
- warnings: none
- compact full replay probe: `seed=7,ticks=40`
- replay size: about 20.8 MB, below the 30 MB quick budget

A short non-gate evaluation was also run:

```bash
npm run sim:evaluate -- --seeds 1,2,3 --ticks 80
```

All three runs remained viable and reproductive. They showed hazards, attacks,
carrion deposition, ecology recovery/depletion, and animal-resource consumption.
However, all three terminal populations were herbivore-only. That is not a
failure by itself, but it is a readiness warning: animal niches exist, yet
terminal trophic diversity may still be fragile. The release gate should decide
whether this is only short-run behavior or a long-run balance issue.

## Blockers Before Mind v1

### 1. No Formal Observation Contract

The current heuristic controller is private world logic in
`runtime/actions.py` plus many `SimulationWorld` helper calls. It reads local
tiles, derived fields, and biotic cues, but there is no frozen observation
schema for a learned controller.

Mind v1 needs:

- an observation encoder boundary independent of the heuristic policy
- an explicit local-patch shape
- scalar self-state fields
- nearby-agent/resource channels
- masking for impossible actions
- tests proving the learned controller cannot access privileged global state

Until this exists, a learned controller would either duplicate private world
knowledge or accidentally train on information the agent should not have.

### 2. No Complete Action/Outcome Trajectory Log

The replay logs successful events and frame state, but it does not log every
agent decision. Failed actions, `stay`, rejected attacks, failed `eat`/`drink`,
and action masks are not represented as first-class trajectory records.

Mind v1 needs a per-agent trajectory stream:

- tick
- agent id, lineage/species/ecotype
- observation digest or encoded observation
- action requested
- action mask
- action result
- energy/hydration/health delta
- reward or fitness proxy components
- death/reproduction/collapse attribution

This can start as JSONL for implementation speed. Parquet/DuckDB can follow
after the schema is stable.

### 3. No Explicit Reward/Fitness Definition For Learned Control

The ecosystem currently selects through survival and reproduction, which is
valid for evolution. A learned controller needs an explicit training signal, even
if it is only a shaped proxy used for comparison against the heuristic baseline.

Minimum reward components should be defined before Mind v1:

- survival continuation
- energy and hydration stability
- health preservation
- successful resource acquisition
- reproduction readiness and actual reproduction
- penalty for invalid/impossible action attempts
- optional novelty or niche-use component kept separate from survival reward

The reward must be logged, not only computed during training.

## Warnings

### 1. Trophic Diversity May Be Present But Not Durable

Quick gate seeds `7,8` at 40 ticks showed terminal herbivores, omnivores, and
carnivores. A separate short check with seeds `1,2,3` at 80 ticks ended
herbivore-only despite predator/carrion activity.

Action:

- wait for `sim:gate:release` results
- if release also shows weak terminal animal niches, tune predator/carrion
  economics before Mind v1
- make release-gate animal-resource checks stricter after the current run
  finishes, so we do not invalidate the in-flight command

### 2. Gate Criteria Still Need Learnability Checks

The current gate checks viability, terminal pressure channels, replay taxonomy,
and replay size. It does not yet check whether a learned controller would have
enough signal to beat the heuristic policy.

Add gate criteria for:

- sustained trophic diversity across long runs
- non-zero attack, fresh-kill, and carcass use over long sweeps
- trait drift or selection pressure over time
- at least one explainable collapse/recovery or dominance transition in full
  replay probes
- trajectory-log availability once the trajectory schema exists

### 3. Modularity Is Better, But `SimulationWorld` Still Owns Too Much

`world.py` is still about 4,744 lines. The biggest remaining concentrations are:

- frame capture and summary assembly
- consumption and meat-intake logic
- combat/damage/death/reproduction lifecycle
- ecology/hazard/hydrology snapshot helpers
- ecotype clustering and population snapshot logic

This does not block release-gate validation, but it will slow Mind integration.
Before adding learned control, extract the controller-facing boundary first:

- `runtime/observations.py`
- `runtime/action_space.py`
- `runtime/trajectory.py`
- a small policy adapter that can call either the heuristic policy or a learned
  policy

### 4. JSON Replay Is Not A Training Data Backbone

Full replay JSON is good for compatibility and viewer inspection. It is too
large and too coupled to viewer needs to become the primary training dataset.

Action:

- keep full replay as the compatibility artifact
- add a separate trajectory writer with a narrow schema
- later add Parquet/DuckDB once the JSONL trajectory schema stabilizes

## Strengths

- Determinism is heavily protected by goldens, contracts, replay taxonomy
  idempotence, and fixed seed behavior.
- Summary-only is now a real execution mode and avoids full replay/taxonomy
  work.
- The viewer exposes enough state to explain many ecological outcomes by eye:
  hydrology, refuge, hazards, carcasses, combat, trophic roles, ecology, species
  metrics, and collapse events.
- Runtime gates now separate cheap development checks from release/manual
  validation.
- Config and genome pressure knobs are all referenced by runtime behavior.

## Next Implementation Slices

1. Wait for `sim:gate:release`.
If it fails, fix the concrete Foundation blocker first. If it passes, continue
with the Mind-readiness interface work below.

2. Add an observation/action contract.
Create frozen local observation and action-mask builders, with tests that verify
shape, determinism, and no privileged global access.

3. Add trajectory logging.
Record every agent decision and outcome in summary-only compatible form. Do not
reuse full replay JSON as training data.

4. Define reward/fitness components.
Start conservative and log each component separately so later training can
change weighting without losing auditability.

5. Tighten the release gate.
After the in-flight release command finishes, add stricter checks for durable
trophic diversity, animal-resource use, trait drift, and trajectory-log
availability.

Only after these slices should `Mind v1` begin.
