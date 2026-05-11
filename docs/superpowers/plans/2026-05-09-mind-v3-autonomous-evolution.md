# Mind v3 Autonomous Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Mind v3 path where agents choose actions through inherited/evolved controller state with no heuristic action selection, no hard-guard fallback, and no confidence delegation.

**Architecture:** Mind v3 is not another guarded IQL artifact and not a side quest. It is the target controller architecture: each agent owns bounded controller state in `mind_inheritance_metadata`; founders receive deterministic randomized controller heads; children inherit and mutate controller state; action choice is masked neural scoring over the existing world action API. Development remains feature-gated until v3 proves survival and reproduction, but the heuristic is only an external evaluation baseline, never a runtime fallback inside Mind v3.

**Tech Stack:** Python 3.14, deterministic simulator runtime, existing `Policy` protocol, existing `mind_observation_v3` input encoder, pure-Python neural scoring for runtime, optional PyTorch only for later offline analysis.

---

## Boundary Decision

We are ready to start Mind v3 as the main controller direction. We are not ready to make it the default controller until it independently sustains populations across the validation matrix.

The current code already has the required scaffolding:

- `python/evolution_sim/env/world.py` accepts an injected `Policy` and records policy sources.
- `python/evolution_sim/env/runtime/observations.py` exposes `metadata.agent_id` and a stable encoded observation vector.
- `python/evolution_sim/env/runtime/action_space.py` builds legal action masks.
- `python/evolution_sim/env/runtime/state.py` already carries inert `mind_inheritance_metadata`.
- `python/evolution_sim/env/runtime/reproduction.py` already threads child metadata through `build_child_agent`.
- `python/evolution_sim/mind/learned_policy.py` already proves heuristic-free runtime modes are possible.

The missing piece is agent-owned controller inheritance and an evaluation loop that measures autonomous survival/reproduction directly instead of optimizing guarded fallback rates. Runtime flags are development gates, not product direction; they exist so incomplete autonomous controllers cannot silently replace the current stable baseline.

Important distinction: Mind v3 removes hand-coded action selection, not the action vocabulary. `move_north`, `eat`, `drink`, `attack_*`, `stay`, `mate`, and signal actions are simulator actuators. The learned/evolved controller may choose among legal actuators, but the world still defines what those actuators do.

## File Map

- Create `python/evolution_sim/mind/evolution.py`
  - Owns Mind v3 controller metadata, founder initialization, parent inheritance, mutation, scoring, and diagnostics.
- Create `python/evolution_sim/mind/v3_policy.py`
  - Implements `MindV3EvolutionPolicy`, a pure autonomous `Policy` using per-agent inherited controller metadata.
- Modify `python/evolution_sim/env/runtime/reproduction.py`
  - Add a reproduction-context callback for child Mind metadata.
- Modify `python/evolution_sim/env/world.py`
  - Initialize founder Mind metadata when the active policy supports it.
  - Pass child Mind inheritance callback into reproduction context.
  - Notify the active policy when new child agents are placed.
- Modify `python/evolution_sim/cli/run_headless.py`
  - Add a feature-gated `--mind-v3-autonomous-evolution` runtime for the primary v3 controller path.
- Modify `python/evolution_sim/cli/collect_trajectory.py`
  - Add the same primary v3 runtime plus trajectory diagnostics.
- Create `python/evolution_sim/cli/mind_v3_evaluate.py`
  - Run heuristic baseline and Mind v3 autonomous evolution side by side across seeds/ticks.
- Modify `package.json`
  - Add `sim:mind:v3:evaluate`.
- Modify `python/evolution_sim/mind/contracts.py`
  - Add a Mind v3 autonomous-evolution contract.
- Modify `python/tests/test_mind_v1.py`
  - Add focused contract, inheritance, policy, and CLI tests.
- Create `docs/mind-v3-autonomous-evolution.md`
  - Durable human-readable design and promotion rules.

## Task 1: Contract And Invariants

**Files:**
- Modify: `python/evolution_sim/mind/contracts.py`
- Modify: `python/evolution_sim/mind/__init__.py`
- Test: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write the failing contract test**

Add this test near the existing online-learning contract tests:

```python
def test_mind_v3_autonomous_evolution_contract_declares_no_heuristic_fallback(self) -> None:
    from evolution_sim.mind.contracts import mind_v3_autonomous_evolution_contract

    contract = mind_v3_autonomous_evolution_contract()

    self.assertEqual(
        contract["contract_version"],
        "mind_v3_autonomous_evolution_contract_v1",
    )
    self.assertFalse(contract["enabled_by_default"])
    self.assertEqual(contract["policy"], "mind_v3_autonomous_evolution_policy_v1")
    self.assertEqual(contract["action_selection"], "inherited_controller_masked_argmax_v1")
    self.assertFalse(contract["heuristic_guard"])
    self.assertFalse(contract["heuristic_delegate"])
    self.assertFalse(contract["heuristic_action_selection"])
    self.assertTrue(contract["action_mask_required"])
    self.assertEqual(
        contract["learning_mechanism"],
        "bounded_parental_inheritance_with_mutation_v1",
    )
    self.assertEqual(contract["mind_state_storage"], "agent.mind_inheritance_metadata")
    self.assertEqual(contract["promotion_metric_family"], "autonomous_survival_reproduction")
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_mind_v3_autonomous_evolution_contract_declares_no_heuristic_fallback
```

Expected: fail because `mind_v3_autonomous_evolution_contract` does not exist.

- [ ] **Step 3: Add the contract**

In `python/evolution_sim/mind/contracts.py`, add:

```python
MIND_V3_AUTONOMOUS_EVOLUTION_CONTRACT_VERSION = (
    "mind_v3_autonomous_evolution_contract_v1"
)


def mind_v3_autonomous_evolution_contract() -> dict[str, object]:
    return {
        "contract_version": MIND_V3_AUTONOMOUS_EVOLUTION_CONTRACT_VERSION,
        "enabled_by_default": False,
        "policy": "mind_v3_autonomous_evolution_policy_v1",
        "action_selection": "inherited_controller_masked_argmax_v1",
        "heuristic_guard": False,
        "heuristic_delegate": False,
        "heuristic_action_selection": False,
        "action_mask_required": True,
        "learning_mechanism": "bounded_parental_inheritance_with_mutation_v1",
        "mind_state_storage": "agent.mind_inheritance_metadata",
        "promotion_metric_family": "autonomous_survival_reproduction",
        "controller": {
            "schema_version": "mind_v3_controller_metadata_v1",
            "runtime_backend": "pure_python_deterministic_v1",
            "feature_source": "mind_observation_v3_encoded_input",
            "architecture": "fixed_random_feature_projection_linear_action_head_v1",
            "inherited_parameters": "action_head_weights_and_bias",
        },
    }
```

Export the function from `python/evolution_sim/mind/__init__.py`.

- [ ] **Step 4: Verify the contract test passes**

Run the same unittest command. Expected: `OK`.

## Task 2: Bounded Controller Metadata

**Files:**
- Create: `python/evolution_sim/mind/evolution.py`
- Test: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write failing metadata tests**

Add:

```python
def test_mind_v3_founder_metadata_is_bounded_and_deterministic(self) -> None:
    from random import Random
    from evolution_sim.mind.evolution import (
        MIND_V3_CONTROLLER_SCHEMA_VERSION,
        founder_mind_v3_metadata,
        mind_v3_parameter_count,
    )

    first = founder_mind_v3_metadata(agent_id=7, rng=Random(123))
    second = founder_mind_v3_metadata(agent_id=7, rng=Random(123))

    self.assertEqual(first, second)
    self.assertEqual(first["schema_version"], MIND_V3_CONTROLLER_SCHEMA_VERSION)
    self.assertTrue(first["inherited_state"])
    self.assertEqual(first["state_size"], mind_v3_parameter_count())
    self.assertLessEqual(first["state_size"], 256)
    self.assertIn("action_head_weights", first)
    self.assertIn("action_head_bias", first)


def test_mind_v3_child_metadata_mutates_from_parent(self) -> None:
    from random import Random
    from evolution_sim.mind.evolution import (
        founder_mind_v3_metadata,
        inherit_mind_v3_metadata,
    )

    parent = founder_mind_v3_metadata(agent_id=1, rng=Random(11))
    child = inherit_mind_v3_metadata(
        primary_parent_metadata=parent,
        secondary_parent_metadata=None,
        child_agent_id=2,
        rng=Random(12),
    )

    self.assertTrue(child["inherited_state"])
    self.assertEqual(child["parent_schema_versions"], ["mind_v3_controller_metadata_v1"])
    self.assertNotEqual(child["action_head_bias"], parent["action_head_bias"])
```

- [ ] **Step 2: Run the failing metadata tests**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_mind_v3_founder_metadata_is_bounded_and_deterministic \
  python.tests.test_mind_v1.MindV1Tests.test_mind_v3_child_metadata_mutates_from_parent
```

Expected: fail because `evolution_sim.mind.evolution` does not exist.

- [ ] **Step 3: Implement metadata helpers**

Create `python/evolution_sim/mind/evolution.py` with:

```python
from __future__ import annotations

import math
from collections.abc import Mapping
from random import Random

from evolution_sim.env.runtime.action_contract import ACTION_NAMES

MIND_V3_CONTROLLER_SCHEMA_VERSION = "mind_v3_controller_metadata_v1"
MIND_V3_POLICY_ID = "mind_v3_autonomous_evolution_policy"
MIND_V3_POLICY_VERSION = "mind_v3_autonomous_evolution_policy_v1"
MIND_V3_HIDDEN_UNITS = 16
MIND_V3_MUTATION_SIGMA = 0.035
MIND_V3_WEIGHT_LIMIT = 1.5


def mind_v3_parameter_count() -> int:
    return len(ACTION_NAMES) * MIND_V3_HIDDEN_UNITS + len(ACTION_NAMES)


def founder_mind_v3_metadata(*, agent_id: int, rng: Random) -> dict[str, object]:
    local = Random((agent_id + 1) * 1_000_003 + int(rng.random() * 1_000_000_000))
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "inherited_state": True,
        "state_size": mind_v3_parameter_count(),
        "architecture": "fixed_random_feature_projection_linear_action_head_v1",
        "mutation_policy": "gaussian_head_mutation_v1",
        "parent_schema_versions": [],
        "action_head_weights": {
            action: [
                _round(local.gauss(0.0, 0.08))
                for _ in range(MIND_V3_HIDDEN_UNITS)
            ]
            for action in ACTION_NAMES
        },
        "action_head_bias": {
            action: _round(local.gauss(0.0, 0.04))
            for action in ACTION_NAMES
        },
    }


def inherit_mind_v3_metadata(
    *,
    primary_parent_metadata: Mapping[str, object],
    secondary_parent_metadata: Mapping[str, object] | None,
    child_agent_id: int,
    rng: Random,
) -> dict[str, object]:
    primary_weights = _weights(primary_parent_metadata)
    primary_bias = _bias(primary_parent_metadata)
    secondary_weights = (
        _weights(secondary_parent_metadata)
        if secondary_parent_metadata is not None
        else None
    )
    secondary_bias = (
        _bias(secondary_parent_metadata)
        if secondary_parent_metadata is not None
        else None
    )
    weights: dict[str, list[float]] = {}
    bias: dict[str, float] = {}
    for action in ACTION_NAMES:
        weights[action] = []
        for index, value in enumerate(primary_weights[action]):
            base = value
            if secondary_weights is not None and rng.random() < 0.5:
                base = secondary_weights[action][index]
            weights[action].append(_mutated(base, rng))
        base_bias = primary_bias[action]
        if secondary_bias is not None and rng.random() < 0.5:
            base_bias = secondary_bias[action]
        bias[action] = _mutated(base_bias, rng)
    parent_versions = [
        str(primary_parent_metadata.get("schema_version", "unknown"))
    ]
    if secondary_parent_metadata is not None:
        parent_versions.append(str(secondary_parent_metadata.get("schema_version", "unknown")))
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "inherited_state": True,
        "state_size": mind_v3_parameter_count(),
        "architecture": "fixed_random_feature_projection_linear_action_head_v1",
        "mutation_policy": "gaussian_head_mutation_v1",
        "child_agent_id": int(child_agent_id),
        "parent_schema_versions": parent_versions,
        "action_head_weights": weights,
        "action_head_bias": bias,
    }


def _weights(metadata: Mapping[str, object] | None) -> dict[str, list[float]]:
    if metadata is None or metadata.get("schema_version") != MIND_V3_CONTROLLER_SCHEMA_VERSION:
        return _zero_weights()
    raw = metadata.get("action_head_weights")
    if not isinstance(raw, Mapping):
        return _zero_weights()
    parsed = _zero_weights()
    for action in ACTION_NAMES:
        values = raw.get(action)
        if isinstance(values, list) and len(values) == MIND_V3_HIDDEN_UNITS:
            parsed[action] = [_finite_float(value) for value in values]
    return parsed


def _bias(metadata: Mapping[str, object] | None) -> dict[str, float]:
    if metadata is None or metadata.get("schema_version") != MIND_V3_CONTROLLER_SCHEMA_VERSION:
        return {action: 0.0 for action in ACTION_NAMES}
    raw = metadata.get("action_head_bias")
    if not isinstance(raw, Mapping):
        return {action: 0.0 for action in ACTION_NAMES}
    return {action: _finite_float(raw.get(action, 0.0)) for action in ACTION_NAMES}


def _zero_weights() -> dict[str, list[float]]:
    return {action: [0.0] * MIND_V3_HIDDEN_UNITS for action in ACTION_NAMES}


def _mutated(value: float, rng: Random) -> float:
    return _round(max(-MIND_V3_WEIGHT_LIMIT, min(MIND_V3_WEIGHT_LIMIT, value + rng.gauss(0.0, MIND_V3_MUTATION_SIGMA))))


def _finite_float(value: object) -> float:
    number = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
    if not math.isfinite(number):
        return 0.0
    return max(-MIND_V3_WEIGHT_LIMIT, min(MIND_V3_WEIGHT_LIMIT, number))


def _round(value: float) -> float:
    return round(float(value), 6)
```

- [ ] **Step 4: Verify metadata tests pass**

Run the same unittest command. Expected: `OK`.

## Task 3: Autonomous Policy Scoring

**Files:**
- Modify: `python/evolution_sim/mind/evolution.py`
- Create: `python/evolution_sim/mind/v3_policy.py`
- Test: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write failing policy tests**

Add:

```python
def test_mind_v3_policy_uses_agent_metadata_without_heuristic_source(self) -> None:
    from random import Random
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.mind.evolution import founder_mind_v3_metadata
    from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

    metadata = founder_mind_v3_metadata(agent_id=3, rng=Random(9))
    policy = MindV3EvolutionPolicy(seed=9)
    policy.register_agent_mind(agent_id=3, metadata=metadata)
    observation = {
        "metadata": {"agent_id": 3},
        "observation_input": {"values": [0.0] * 542},
        "action_mask": {action: action in {"stay", "move_east"} for action in ACTION_NAMES},
    }

    decision = policy.decide(observation, dict(observation["action_mask"]))

    self.assertIn(decision.requested_action, {"stay", "move_east"})
    self.assertEqual(decision.policy_id, "mind_v3_autonomous_evolution_policy")
    self.assertEqual(decision.policy_version, "mind_v3_autonomous_evolution_policy_v1")
    self.assertNotIn("heuristic", decision.source)
    self.assertTrue(decision.diagnostics["heuristic_free"])
```

- [ ] **Step 2: Run the failing policy test**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_mind_v3_policy_uses_agent_metadata_without_heuristic_source
```

Expected: fail because `MindV3EvolutionPolicy` does not exist.

- [ ] **Step 3: Add scoring helpers**

In `python/evolution_sim/mind/evolution.py`, add:

```python
def score_mind_v3_metadata(
    *,
    metadata: Mapping[str, object],
    observation_input: list[float],
    action_mask: Mapping[str, bool],
) -> dict[str, float]:
    hidden = _fixed_hidden_features(observation_input)
    weights = _weights(metadata)
    bias = _bias(metadata)
    scores: dict[str, float] = {}
    for action in ACTION_NAMES:
        if not bool(action_mask.get(action, False)):
            continue
        scores[action] = _round(
            bias[action] + sum(weights[action][index] * hidden[index] for index in range(MIND_V3_HIDDEN_UNITS))
        )
    return scores


def _fixed_hidden_features(observation_input: list[float]) -> list[float]:
    values = [_finite_float(value) for value in observation_input]
    if not values:
        values = [0.0]
    hidden: list[float] = []
    for unit in range(MIND_V3_HIDDEN_UNITS):
        total = 0.0
        for offset in range(12):
            index = (unit * 37 + offset * 53) % len(values)
            total += values[index] * _projection_weight(unit, offset)
        hidden.append(_round(math.tanh(total / 4.0)))
    return hidden


def _projection_weight(unit: int, offset: int) -> float:
    return ((unit * 17 + offset * 31) % 23 - 11) / 11.0
```

- [ ] **Step 4: Add `MindV3EvolutionPolicy`**

Create `python/evolution_sim/mind/v3_policy.py`:

```python
from __future__ import annotations

from random import Random
from collections.abc import Mapping

from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.env.runtime.observations import decode_observation_input
from evolution_sim.mind.evolution import (
    MIND_V3_POLICY_ID,
    MIND_V3_POLICY_VERSION,
    founder_mind_v3_metadata,
    inherit_mind_v3_metadata,
    score_mind_v3_metadata,
)


class MindV3EvolutionPolicy:
    policy_id = MIND_V3_POLICY_ID
    policy_version = MIND_V3_POLICY_VERSION

    def __init__(self, *, seed: int) -> None:
        self._rng = Random(seed)
        self._agent_metadata: dict[int, dict[str, object]] = {}

    def register_agent_mind(self, *, agent_id: int, metadata: Mapping[str, object]) -> None:
        self._agent_metadata[int(agent_id)] = dict(metadata)

    def founder_metadata(self, *, agent_id: int) -> dict[str, object]:
        metadata = founder_mind_v3_metadata(agent_id=agent_id, rng=self._rng)
        self.register_agent_mind(agent_id=agent_id, metadata=metadata)
        return metadata

    def child_metadata(
        self,
        *,
        child_agent_id: int,
        primary_parent_id: int,
        secondary_parent_id: int | None,
    ) -> dict[str, object]:
        metadata = inherit_mind_v3_metadata(
            primary_parent_metadata=self._agent_metadata.get(primary_parent_id, {}),
            secondary_parent_metadata=(
                self._agent_metadata.get(secondary_parent_id, {})
                if secondary_parent_id is not None
                else None
            ),
            child_agent_id=child_agent_id,
            rng=self._rng,
        )
        self.register_agent_mind(agent_id=child_agent_id, metadata=metadata)
        return metadata

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        agent_id = _agent_id(observation)
        metadata = self._agent_metadata.get(agent_id)
        if metadata is None:
            metadata = self.founder_metadata(agent_id=agent_id)
        scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=_observation_values(observation),
            action_mask=action_mask,
        )
        requested_action, score = _best_action(scores, action_mask)
        return ActionDecision(
            requested_action=requested_action,
            source="mind_v3_autonomous_evolution_policy_v1",
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics={
                "runtime_mode": "mind-v3-autonomous-evolution",
                "heuristic_free": True,
                "agent_id": agent_id,
                "score": score,
                "legal_action_count": sum(1 for allowed in action_mask.values() if allowed),
            },
        )


def _agent_id(observation: Mapping[str, object]) -> int:
    metadata = observation.get("metadata")
    if not isinstance(metadata, Mapping):
        return -1
    return int(metadata.get("agent_id", -1))


def _observation_values(observation: Mapping[str, object]) -> list[float]:
    payload = observation.get("observation_input")
    if isinstance(payload, Mapping):
        values = payload.get("values")
        if isinstance(values, list):
            return [float(value) for value in values]
        return decode_observation_input(dict(payload))
    return []


def _best_action(scores: Mapping[str, float], action_mask: Mapping[str, bool]) -> tuple[str, float]:
    best_action = "stay"
    best_score = float("-inf")
    for action in sorted(action_mask):
        if not bool(action_mask[action]):
            continue
        score = float(scores.get(action, 0.0))
        if score > best_score:
            best_action = action
            best_score = score
    if best_score == float("-inf"):
        return "stay", 0.0
    return best_action, best_score
```

- [ ] **Step 5: Verify policy test passes**

Run the same unittest command. Expected: `OK`.

## Task 4: Birth-Time Inheritance Hook

**Files:**
- Modify: `python/evolution_sim/env/runtime/reproduction.py`
- Modify: `python/evolution_sim/env/world.py`
- Test: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write failing inheritance integration test**

Add a test that constructs a short world with `MindV3EvolutionPolicy`, runs until at least one birth if possible, and asserts that every child with a parent has `mind_inheritance_metadata.inherited_state == True`. Keep the test bounded to a tiny deterministic config and skip only if no birth occurs in the bounded run.

```python
def test_mind_v3_policy_initializes_founders_and_children(self) -> None:
    from evolution_sim.config import WorldConfig
    from evolution_sim.env.world import SimulationWorld
    from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

    policy = MindV3EvolutionPolicy(seed=7)
    result = SimulationWorld(
        WorldConfig(seed=7, max_ticks=160, initial_agents=12, max_agents=80),
        policy=policy,
    ).run()
    catalog = result.replay["agent_catalog"]
    founders = [agent for agent in catalog.values() if agent["parent_id"] is None]
    children = [agent for agent in catalog.values() if agent["parent_id"] is not None]

    self.assertTrue(founders)
    self.assertTrue(all(agent["mind_inheritance"]["inherited_state"] for agent in founders))
    if children:
        self.assertTrue(all(agent["mind_inheritance"]["inherited_state"] for agent in children))
```

- [ ] **Step 2: Run the failing integration test**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_mind_v3_policy_initializes_founders_and_children
```

Expected: fail because founders still get empty metadata.

- [ ] **Step 3: Extend `ReproductionContext`**

In `python/evolution_sim/env/runtime/reproduction.py`, add a field:

```python
    mind_inheritance_for_child: Callable[[Agent, Agent | None, int], dict[str, object]] | None = None
```

Replace both `mind_inheritance_metadata=empty_mind_inheritance_metadata()` calls in child builders with:

```python
mind_inheritance_metadata=(
    reproduction_context.mind_inheritance_for_child(parent, None, child_agent_id)
    if reproduction_context.mind_inheritance_for_child is not None
    else empty_mind_inheritance_metadata()
),
```

For sexual children, pass `partner` instead of `None`.

- [ ] **Step 4: Wire world policy hooks**

In `python/evolution_sim/env/world.py`, add helper methods:

```python
    def _founder_mind_metadata(self, agent_id: int) -> dict[str, object]:
        founder_metadata = getattr(self.policy, "founder_metadata", None)
        if callable(founder_metadata):
            metadata = founder_metadata(agent_id=agent_id)
            if isinstance(metadata, dict):
                return dict(metadata)
        return empty_mind_inheritance_metadata()

    def _mind_inheritance_for_child(
        self,
        primary_parent: Agent,
        secondary_parent: Agent | None,
        child_agent_id: int,
    ) -> dict[str, object]:
        child_metadata = getattr(self.policy, "child_metadata", None)
        if callable(child_metadata):
            metadata = child_metadata(
                child_agent_id=child_agent_id,
                primary_parent_id=primary_parent.agent_id,
                secondary_parent_id=(
                    secondary_parent.agent_id if secondary_parent is not None else None
                ),
            )
            if isinstance(metadata, dict):
                return dict(metadata)
        return empty_mind_inheritance_metadata()
```

Use `_founder_mind_metadata(self.next_agent_id)` in `_spawn_initial_agents`.

Pass `mind_inheritance_for_child=self._mind_inheritance_for_child` in `_reproduction_context`.

- [ ] **Step 5: Verify inheritance integration passes**

Run the same unittest command. Expected: `OK`.

## Task 5: CLI Surface

**Files:**
- Modify: `python/evolution_sim/cli/run_headless.py`
- Modify: `python/evolution_sim/cli/collect_trajectory.py`
- Test: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write CLI tests**

Add tests that invoke both CLIs with `--mind-v3-autonomous-evolution` and assert output exists, `mind_policy=mind_v3_autonomous_evolution_policy`, and no action source contains `heuristic`.

- [ ] **Step 2: Implement CLI flag**

In both CLIs, add:

```python
parser.add_argument(
    "--mind-v3-autonomous-evolution",
    action="store_true",
    help="Use the experimental Mind v3 inherited autonomous controller without heuristic fallback.",
)
```

When true, reject simultaneous `--mind-artifact` and construct:

```python
from evolution_sim.mind.v3_policy import MindV3EvolutionPolicy

policy = MindV3EvolutionPolicy(seed=args.seed)
```

- [ ] **Step 3: Verify CLI tests pass**

Run the focused CLI tests, then:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m py_compile \
  python/evolution_sim/cli/run_headless.py \
  python/evolution_sim/cli/collect_trajectory.py
```

Expected: `OK` and no compile errors.

## Task 6: Mind v3 Evaluation Report

**Files:**
- Create: `python/evolution_sim/cli/mind_v3_evaluate.py`
- Modify: `package.json`
- Test: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write report test**

Add a test that runs the CLI over `--seeds 5 --ticks 40` and asserts:

```python
report["schema_version"] == "mind_v3_autonomous_evolution_evaluation_v1"
report["policy"]["heuristic_free"] is True
report["comparison"]["heuristic"]["runs"][0]["seed"] == 5
report["comparison"]["mind_v3"]["runs"][0]["seed"] == 5
report["comparison"]["mind_v3"]["aggregate"]["heuristic_action_source_count"] == 0
```

- [ ] **Step 2: Implement the evaluator**

The evaluator should:

1. Parse `--seeds`, `--ticks`, and `--output`.
2. For each seed, run the heuristic baseline through `SimulationWorld(WorldConfig(...)).run(mode=RunMode.SUMMARY_ONLY)`.
3. For each seed, run Mind v3 through `SimulationWorld(WorldConfig(...), policy=MindV3EvolutionPolicy(seed=seed)).run(mode=RunMode.SUMMARY_ONLY)`.
4. Write JSON with alive agents, births, deaths, action-source counts, policy ids, and deltas.

- [ ] **Step 3: Add npm script**

In `package.json`:

```json
"sim:mind:v3:evaluate": "PYTHONHASHSEED=0 PYTHONPATH=python python3 -m evolution_sim.cli.mind_v3_evaluate"
```

- [ ] **Step 4: Verify evaluator**

Run:

```bash
npm run sim:mind:v3:evaluate -- --seeds 5,13 --ticks 120 --output output/mind/mind-v3-autonomous-evolution-smoke.json
jq empty output/mind/mind-v3-autonomous-evolution-smoke.json
```

Expected: JSON exists and parses. Do not expect promotion-level survival yet.

## Task 7: First Large-Scale Milestone Run

**Files:**
- No code changes unless Task 6 exposes missing diagnostics.
- Output: `output/mind/mind-v3-autonomous-evolution-1000tick-report.json`

- [ ] **Step 1: Run a real autonomous-evolution milestone**

Run:

```bash
npm run sim:mind:v3:evaluate -- \
  --seeds 5,13,19,29,37,41 \
  --ticks 1000 \
  --output output/mind/mind-v3-autonomous-evolution-1000tick-report.json
```

- [ ] **Step 2: Parse and inspect**

Run:

```bash
jq '{
  schema_version,
  mind_v3_alive_mean: .comparison.mind_v3.aggregate.alive_agents_mean,
  mind_v3_births_mean: .comparison.mind_v3.aggregate.births_mean,
  heuristic_alive_mean: .comparison.heuristic.aggregate.alive_agents_mean,
  heuristic_births_mean: .comparison.heuristic.aggregate.births_mean,
  heuristic_action_source_count: .comparison.mind_v3.aggregate.heuristic_action_source_count
}' output/mind/mind-v3-autonomous-evolution-1000tick-report.json
```

Expected for the first run: `heuristic_action_source_count == 0`. Survival may be poor; the milestone is to establish honest autonomous learning data, not to claim success.

## Task 8: Documentation And Stop Rules

**Files:**
- Create: `docs/mind-v3-autonomous-evolution.md`
- Modify: `docs/mind-v2-online-neural-roadmap.md`
- Modify: `docs/mind-v2-audit-and-experiment-log.md`

- [ ] **Step 1: Write the v3 doc**

Create `docs/mind-v3-autonomous-evolution.md` with:

```markdown
# Mind v3 Autonomous Evolution

Mind v3 is the first track where action selection is heuristic-free by design.
The simulator still defines legal action primitives and physics. The agent
controller chooses among legal actions without heuristic fallback.

## Non-Goals

- No post-hoc extraction caps.
- No fallback-rate micro-optimization.
- No heuristic guard or confidence delegate inside the Mind v3 runtime.
- No default-runtime promotion until autonomous survival and reproduction are measured.

## First Acceptance Boundary

- Mind v3 action source count contains zero heuristic actions.
- Every founder has bounded inherited controller metadata.
- Every child produced under the v3 runtime inherits bounded controller metadata.
- The v3 evaluator reports survival, births, action-source counts, and lineage/controller metadata.
- The first 1000-tick multi-seed report is recorded even if it fails survival.

## Promotion Boundary

Mind v3 can only replace the default heuristic after it independently sustains
alive agents and births across the extended seed matrix. Guarded fallback rates
are not promotion metrics for this track.
```

- [ ] **Step 2: Update old docs**

Add a short note to the v2 roadmap and audit ledger:

```markdown
Mind v3 is now the primary autonomous-evolution direction. V2 remains the
guarded artifact baseline for comparison and continuity; v3 removes heuristic
action selection and measures autonomous survival/reproduction directly.
```

## Validation Ladder

Run after implementation:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest python.tests.test_mind_v1
npm run sim:test
npm run sim:mind:v3:evaluate -- --seeds 5,13 --ticks 120 --output output/mind/mind-v3-autonomous-evolution-smoke.json
jq empty output/mind/mind-v3-autonomous-evolution-smoke.json
git diff --check
```

Run for the first real milestone:

```bash
npm run sim:mind:v3:evaluate -- \
  --seeds 5,13,19,29,37,41 \
  --ticks 1000 \
  --output output/mind/mind-v3-autonomous-evolution-1000tick-report.json
```

## Stop Rules

Stop and re-plan if:

- Mind v3 uses `ObservationHeuristicPolicy` or any heuristic fallback source during runtime.
- The only improvement is a fallback-rate movement.
- Controller metadata grows beyond a bounded, auditable payload.
- Full replay compatibility changes without matching tests.
- Summary-only runs start retaining full replay payloads.

## Research Alignment

This plan follows the parts of current research that fit this simulator:

- neuroevolution and evolution strategies for scalable population search;
- quality-diversity/open-ended search for maintaining diverse controller lineages;
- world-model control as a later stage, after honest autonomous rollouts exist.

It deliberately does not jump straight to in-tick gradient updates. The current
repo has strong deterministic replay contracts; v3 should earn autonomy first
through bounded inherited controller state and observable selection pressure.
