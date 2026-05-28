# Mind Controllers

Mind Controllers are learned or autonomous policy systems measured against the
Foundation baseline. Current research is Mind v3 autonomous-controller work,
while Mind v1 and v2 remain guarded safety-floor and measurement boundaries.

## Language

**Mind**:
The controller layer responsible for action selection, learned behavior,
lifetime adaptation experiments, and later planning.
_Avoid_: AI, brain

**Heuristic baseline**:
The existing observation-driven controller used as the external comparison and
safety floor for learned-controller work.
_Avoid_: oracle, solution

**Guarded controller**:
A learned controller constrained by guards, fallback, or delegation so it cannot
silently harm Foundation behavior during evaluation.
_Avoid_: promoted autonomous controller

**Mind v1**:
The first guarded learned-controller path built around trajectory collection,
behavior cloning, and readiness gates.
_Avoid_: autonomous mind

**Mind v2**:
The guarded neural/offline-RL continuation that explored neural artifacts,
IQL-style trainers, confidence delegation, and strict gate behavior.
_Avoid_: final learned controller

**Mind v3**:
The current autonomous-controller track. Mind v3 removes heuristic action
selection from the agent runtime while remaining opt-in until broad gates pass.
_Avoid_: heuristic-tuned v2

**Autonomous controller**:
A controller that selects among legal actions without heuristic fallback, hard
guards, or confidence delegation inside the runtime.
_Avoid_: guarded controller

**Heuristic action**:
An action chosen by heuristic fallback, guard intervention, or delegation rather
than by the candidate autonomous controller.
_Avoid_: safe action

**Action collapse**:
A failure mode where a candidate overuses one requested or resolved action and
does not sustain diverse survival, movement, resource, or reproduction behavior.
_Avoid_: specialization

**Held-out seed**:
A seed reserved for validation rather than candidate fitting. Aggregate gains do
not excuse per-seed alive or birth regressions on held-out seeds.
_Avoid_: test seed when used casually

**Fixture**:
A controlled scenario or slice used to expose a specific controller failure
mode, such as carrion movement or recovery behavior.
_Avoid_: benchmark

**Rollout context**:
Policy-owned historical feedback from the same agent's finalized public
trajectory rows, snapshotted before a current decision.
_Avoid_: future context, private world state

**Promotion**:
Moving a controller from opt-in experiment toward default runtime behavior after
strict broad held-out gates, fixture evidence, and replay diagnostics hold.
_Avoid_: best aggregate score

## Example Dialogue

Developer: "This candidate improves mean alive count, but seed 19 loses births."

Domain expert: "That is not promotable. Mind v3 promotion requires strict
per-seed behavior; aggregate gains do not excuse alive or birth regressions."

Developer: "Can we let v3 delegate uncertain actions to the heuristic?"

Domain expert: "No. That turns it back into guarded v1/v2 behavior. For v3, fix
the policy capacity or data path while keeping heuristic actions at zero."
