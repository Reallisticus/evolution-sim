---
name: mind-contract-loop
description: Use for Mind-interface work in evolution-sim: observation schemas, action masks, policy interfaces, trajectory logging, reward components, trainable datasets, policy-owned runtime state, replayable update traces, environment reset semantics, and learned-controller readiness. Do not use for model training until the controller boundary and data contract are enforceable.
---

# Mind Contract Loop

Use this skill for the interface between the simulator and learned controllers.
Pair it with `mind-v3-autonomous-loop` when policy capacity, training behavior,
or promotion evidence is in scope.

## Hard Rule

Do not start model training, architecture selection, hyperparameter tuning, or
policy optimization until all relevant controller-boundary and data-contract
requirements are enforceable by tests and gates.

If a model needs new policy input, policy memory, update traces, artifact
metadata, trajectory labels, or acceptance metrics, harden that contract first.
The model must not smuggle private world state or fixture identity through the
new surface.

## Inspect First

Read the smallest relevant set:

- `docs/evolution-simulator-blueprint.md`
- `docs/foundation-readiness-audit.md`
- `docs/mind-readiness-audit-2026-04-27.md`
- `docs/mind-v3-autonomous-evolution.md`
- `python/evolution_sim/env/runtime/observations.py`
- `python/evolution_sim/env/runtime/action_space.py`
- `python/evolution_sim/env/runtime/actions.py`
- `python/evolution_sim/env/runtime/trajectory.py`
- `python/evolution_sim/env/world.py`
- `python/evolution_sim/mind/dataset.py`
- `python/evolution_sim/mind/policy_inputs.py`
- relevant tests in `python/tests/`

## Contract Requirements

Before learned control, the repo needs:

- A policy interface that consumes frozen observations, not `SimulationWorld`.
- A canonical observation encoder with declared shape, dtype, order, ranges, enum vocabularies, and missing-value policy.
- Policy-input fields separated from metadata such as agent id, lineage id, or debug labels.
- Action masks that distinguish observation-time validity from resolution-time validity, or a documented two-phase action model.
- Trajectory records that contain trainable observation payloads, not only digests.
- Trajectory collection independent of full replay/viewer payload generation.
- Reward or fitness components logged separately with tests for scale, terminal events, invalid actions, resource acquisition, and reproduction.
- Reset or one-shot environment semantics that cannot silently mix episodes.
- Policy-owned runtime state that is deterministic, serialized, or replayable
  through update traces, and derived only from public observations, policy
  actions, and finalized outcomes.

## Work Loop

1. Define the interface being hardened.
   - Observation
   - Action space or masks
   - Policy boundary
   - Trajectory data
   - Reward components
   - Episode/reset semantics
2. State what must be impossible after the change.
   - Example: a policy cannot read privileged global state.
   - Example: a trajectory row cannot omit the observation needed for training.
3. Add a focused contract test first where practical.
4. Implement the smallest boundary change.
5. Validate with targeted tests, then the fast simulator suite.
6. Update gate checks only after the contract itself is stable.

## Validation Ladder

Use targeted tests first:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_runtime_contracts.py'
PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest discover -s python/tests -p 'test_headless_sim.py'
```

Then broaden:

```bash
npm run sim:test
npm run sim:golden:quick
npm run sim:gate:quick
git diff --check
```

Use full gates only when the interface affects release readiness:

```bash
npm run sim:test:full
npm run sim:gate:release
```

## Report

Include:

- contract hardened
- privileged access or ambiguity removed
- tests added or changed
- validation commands run
- what still blocks safe learned-controller experimentation or promotion
