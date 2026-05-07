# Mind v2 Audit and Experiment Log

Date: 2026-05-06

## Purpose

This document records the current Mind v2 code audit, research checkpoint, and
experiment ledger so future Mind work does not repeat failed trainer or
delegation paths. Any new Mind trainer, calibration rule, or gate probe should
add a row here before the next implementation slice starts.

## Current Control

The current promoted runtime remains heuristic-backed and Mind-disabled by
default. The main comparison target for learned-controller work is:

- default trainer: `contextual-prior`
- model type: `guarded_contextual_local_prior_bc_v2`
- default report: `output/mind/mind-v1-gate-report.json`
- hard guard: `0.1190`
- heuristic delegate: `0.3590`
- total heuristic fallback: `0.4780`
- alive delta: `0.0`
- births delta: `0.0`

Strict promotion target for a learned-controller replacement remains:

- total heuristic fallback below `0.4780`
- hard guard below `0.1190`
- zero per-seed alive regressions
- zero per-seed birth regressions
- default and extended held-out gates green

## Research Checkpoint

The codebase is moving from weighted behavior cloning toward true discrete
offline RL. The current custom PyTorch path now includes a first discrete
IQL-style trainer that still serializes dependency-free runtime artifacts. It
is not promoted: the first corrected gate showed useful hard-guard reduction
but poor delegation and held-out critic calibration.

Relevant primary sources and package docs checked on 2026-05-06:

- TorchRL documents offline RL losses for CQL, Discrete CQL, IQL, Discrete IQL,
  and TD3+BC:
  <https://docs.pytorch.org/rl/main/reference/objectives_offline.html>
- TorchRL is intended as a modular PyTorch-first RL library, with PyPI install
  support and releases aligned with PyTorch:
  <https://docs.pytorch.org/rl/stable/index.html>
- CQL learns conservative Q-functions to reduce offline distribution-shift
  overestimation:
  <https://arxiv.org/abs/2006.04779>
- IQL improves offline policies without directly evaluating unseen actions and
  extracts the policy with advantage-weighted behavior cloning:
  <https://huggingface.co/papers/2110.06169>
- Cal-QL focuses on calibrated conservative values for offline-to-online
  fine-tuning:
  <https://arxiv.org/abs/2303.05479>
- DreamerV3 is a later world-model milestone, not the immediate next step,
  because this repo still needs a stable actor/critic replay loop first:
  <https://arxiv.org/abs/2301.04104>
- Quality-Diversity is relevant to open-ended artificial-life behavior after
  the controller loop is stable:
  <https://arxiv.org/abs/2406.04235>

Package freshness checked with `pip index versions <package>`:

| Package | Latest seen | Local state | Decision |
| --- | ---: | --- | --- |
| `torch` | `2.11.0` | installed `2.11.0` in `.venv` | current |
| `torchrl` | `0.12.0` | installed `0.12.0` in `.venv` | current |
| `tensordict` | `0.12.2` | installed `0.12.2` in `.venv` | current |
| `gymnasium` | `1.3.0` | installed `1.3.0` in `.venv` | current |
| `pettingzoo` | `1.26.1` | installed `1.26.1` in `.venv` | current |
| `minari` | `0.5.3` | installed `0.5.3` in `.venv` | current |
| `d3rlpy` | `2.8.1` | installed `2.8.0` in `.venv` | hold below `2.8.1` because the PyPI metadata still pins `gymnasium==1.0.0`, conflicting with the current Farama stack |

## Code Audit

No immediate runtime contract break was found in the current opt-in neural
slice. The important safety properties are still present:

- learned policy loading requires explicit `enable_mind=True`;
- normal simulator runs remain Mind-disabled by default;
- neural runtime inference uses serialized artifact weights, not a live PyTorch
  dependency;
- live policy scoring re-encodes policy-visible observations through the public
  observation encoder;
- action masks are still applied before choosing a learned action;
- neural acceptance remains guarded and narrow.

The main weaknesses are structural and methodological:

1. `baseline.py`, `artifacts.py`, `diagnostics.py`, and `learned_policy.py` are
   now large Mind hotspots. New trainer work currently touches too many files,
   which increases the chance of metadata drift.
2. Value-supported deviation metadata validation exists in two paths: the
   value-calibrated artifact path and the neural artifact helper. This is mostly
   safe today, but future fields should be added through one shared validator.
3. `load_learned_policy()` still uses `assert isinstance(...)` after
   `load_model_artifact()` validation. This is not the highest risk because the
   artifact loader validates first, but production parsing should use explicit
   exceptions rather than asserts that can be stripped by optimized Python.
4. The current `torch-advantage-actor-critic-bc` trainer is an
   advantage-weighted supervised actor with value heads. It remains useful as a
   comparison, but the real offline-RL path should now build on the
   episode-aware transition adapter and `torch-discrete-iql`.
5. Runtime calibration acceptance is useful as a gateable hook, but widening it
   is the wrong next move. The drink-capable version already showed that
   broader acceptance can regress population outcomes while looking locally
   plausible.
6. Experiment results were previously spread across chat, gate reports, and
   stage-plan prose. This file is now the canonical ledger for repeat-avoidance.

## Experiment Ledger

| Slice | Report | Status | Hard guard | Delegate | Total fallback | Alive delta | Births delta | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Heuristic safety floor | paired comparison baseline in each report | pass | -- | -- | -- | baseline | baseline | reference behavior; all learned rows compare outcomes against this |
| Default contextual prior | `output/mind/mind-v1-gate-report.json` | pass | `0.1190` | `0.3590` | `0.4780` | `0.0` | `0.0` | current control |
| Reward-weighted contextual prior | `output/mind/mind-v1-gate-reward-report.json` | pass | `0.1563` | `0.3242` | `0.4805` | `0.0` | `0.0` | do not promote; worsens hard guard and total fallback |
| Advantage-blended contextual prior | `output/mind/mind-v1-gate-advantage-blended-report.json` | pass | `0.1123` | `0.3650` | `0.4773` | `0.0` | `0.0` | best non-neural fallback result; keep as comparison candidate |
| Advantage-blended strict/extended | `output/mind/mind-v1-gate-strict-report.json` | pass | `0.1123` | `0.3630` | `0.4753` | `0.0` | `0.0` | passes strict threshold; still not a neural/RL replacement |
| Value-calibrated contextual prior | `output/mind/mind-v1-gate-value-report.json` | pass | `0.0966` | `0.3807` | `0.4773` | `0.0` | `0.0` | good guard reduction; delegate rises |
| Torch advantage actor-critic BC | `output/mind/mind-v2-torch-advantage-seedbank-report.json` | pass | `0.2703` | `0.3125` | `0.5828` | `+1.5` | `+1.5` | diagnostics only; not competitive |
| Torch advantage calibrated delegation, broad resource acceptance | `output/mind/mind-v2-torch-advantage-calibrated-delegation-report.json` | fail | `0.1533` | `0.3549` | `0.5082` | `-16.0` | `-15.5` | rejected; do not repeat broad `drink`/resource bypass |
| Torch advantage local-eat-only calibration | `output/mind/mind-v2-torch-advantage-calibrated-resource-report.json` | pass | `0.1641` | `0.4077` | `0.5718` | `+1.5` | `+1.5` | hook works but does not beat control; do not widen until critic improves |
| Torch discrete IQL, value-deviation disabled | `output/mind/mind-v2-torch-discrete-iql-report.json` | pass | `0.0014` | `0.5820` | `0.5834` | `0.0` | `0.0` | not promoted; IQL no longer emits runtime value-deviation metadata, so safe-deviation rate is `0.0` and per-seed outcomes match control, but delegation/fallback are worse than the current control |
| Torch discrete IQL, mask-normalized neural confidence | `output/mind/mind-v2-torch-discrete-iql-report.json` | fail | `0.5772` | `0.0062` | `0.5834` | `0.0` | `0.0` | rejected as controller; legal-action score renormalization fixed confidence accounting, but exposed that the IQL actor is mostly being stopped by the hard safety floor |
| Torch discrete IQL, contextual-prior anchored actor | `output/mind/mind-v2-torch-discrete-iql-report.json` | pass | `0.1224` | `0.3524` | `0.4748` | `0.0` | `0.0` | promotion progress; total fallback beats `0.4780` with zero per-seed deltas, but hard guard still misses the `0.1190` control target |
| Torch discrete IQL, advantage-blended actor anchor | `output/mind/mind-v2-torch-discrete-iql-report.json` | pass | `0.1459` | `0.3304` | `0.4763` | `0.0` | `0.0` | rejected as default IQL anchor; total fallback still beats control, but hard guard worsens materially |
| Torch discrete IQL, contextual anchor plus IQL delegate calibration | `output/mind/mind-v2-torch-discrete-iql-report.json` | pass | `0.1041` | `0.3707` | `0.4748` | `0.0` | `0.0` | first neural/RL strict-control candidate on the default held-out gate; extended strict gate is still required before promotion |
| Torch discrete IQL, contextual anchor plus IQL delegate calibration extended strict | `output/mind/mind-v2-torch-discrete-iql-extended-report.json` | pass | `0.1054` | `0.3661` | `0.4715` | `0.0` | `0.0` | passes the extended strict fallback/outcome matrix; candidate for promotion review, but still heuristic-backed through delegation |

Held-out neural diagnostics for the corrected discrete IQL probe:

- actor top-1 accuracy: `0.4136`
- action-value MAE: `4.8286`
- state-value MAE: `4.6417`
- training transition count: `44,324`
- training TD target MAE: `0.3498`
- training state-value MAE: `0.2507`

The training losses are plausible, but held-out value scale is not calibrated.
The previous IQL run that allowed value-supported local-resource deviations had
safe-deviation rate `0.0168` and seed `5` alive delta `-2.0`. The current IQL
artifact contract therefore forbids those runtime value-deviation fields until
held-out Q/V calibration improves enough to pass strict gates without per-seed
outcome regressions.

The mask-normalized confidence probe is also diagnostic, not promotable. It
renormalizes neural actor scores over legal actions before runtime delegation
and artifact calibration buckets. That collapsed confidence delegation from
`0.5820` to `0.0062`, but hard guard rose from `0.0014` to `0.5772` with the
same `0.5834` total fallback. The learned actor is no longer merely
low-confidence; it is confidently choosing actions that the safety floor
suppresses. The next learner slice must improve actor ranking/safety, not tune
delegation thresholds.

The first promotion-progress slice anchors the mask-renormalized IQL actor to
the current contextual prior at weight `0.9`. This keeps the neural Q/V heads and
masked actor extraction, but regularizes live actor ranking toward supported
behavior before confidence delegation. The initial contextual anchor reduced
total fallback to `0.4748` with zero per-seed alive/birth deltas but missed the
hard-guard target at `0.1224`. An advantage-blended anchor was tested and
recorded; it was rejected because hard guard rose to `0.1459`. The accepted
default-candidate slice keeps the contextual anchor and adds an explicit
IQL-only delegate-margin calibration of `0.25`, reducing hard guard to `0.1041`
while preserving total fallback `0.4748` and zero per-seed outcome deltas. This
is a default strict-control candidate, not a full heuristic replacement: it
still delegates `0.3707` of decisions to the heuristic, so the extended strict
matrix is the required next promotion boundary.

The same candidate now passes the extended strict matrix across seeds
`5,13,19,29,37,41` at `120` and `180` ticks. The extended report records hard
guard `0.1054`, confidence delegation `0.3661`, total fallback `0.4715`,
safe-deviation rate `0.0`, and zero minimum per-seed alive/birth deltas. Treat
this as a promotion-review milestone for the neural/RL artifact path, not the
end state: the runtime still relies on heuristic delegation for more than a
third of decisions and does not learn online during a run.

## Latest Validation Ledger

Fresh checks from the audit slice:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_trainer_writes_real_ml_artifact \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_artifact_rejects_runtime_value_deviation \
  python.tests.test_mind_v1.MindV1Tests.test_torch_advantage_actor_critic_trainer_writes_real_ml_artifact \
  python.tests.test_mind_v1.MindV1Tests.test_torch_actor_critic_trainer_writes_real_ml_artifact
# Ran 4 tests in 2.663s
# OK

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_trainer_writes_real_ml_artifact \
  python.tests.test_mind_v1.MindV1Tests.test_neural_policy_anchors_actor_ranking_to_contextual_prior \
  python.tests.test_mind_v1.MindV1Tests.test_neural_policy_anchors_actor_ranking_to_advantage_blended_prior
# Ran 3 tests in 1.532s
# OK

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_gate \
  --reuse-trajectories \
  --trainer torch-discrete-iql \
  --artifact-output output/mind/mind-v2-torch-discrete-iql-artifact.json \
  --output output/mind/mind-v2-torch-discrete-iql-report.json
# readiness.status = pass
# decision = strict_control_candidate
# hard_guard = 0.1041
# heuristic_delegate = 0.3707
# total_fallback = 0.4748
# safe_deviation = 0.0
# min_alive_delta = 0.0
# min_births_delta = 0.0
# strict_control_target_passed = true

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_gate \
  --reuse-trajectories \
  --trainer torch-discrete-iql \
  --validation-seeds 5,13,19,29,37,41 \
  --validation-ticks 120,180 \
  --artifact-output output/mind/mind-v2-torch-discrete-iql-extended-artifact.json \
  --output output/mind/mind-v2-torch-discrete-iql-extended-report.json \
  --max-alive-agents-mean-regression 0 \
  --max-alive-agents-per-seed-regression 0 \
  --max-births-mean-regression 0 \
  --max-births-per-seed-regression 0 \
  --max-guard-intervention-rate 0.1189 \
  --max-total-heuristic-fallback-rate 0.4779 \
  --fail-on-blockers
# readiness.status = pass
# decision = strict_control_candidate
# hard_guard = 0.1054
# heuristic_delegate = 0.3661
# total_fallback = 0.4715
# safe_deviation = 0.0
# min_alive_delta = 0.0
# min_births_delta = 0.0
# strict_control_target_passed = true

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_mind_v1
# Ran 94 tests in 19.544s
# OK (skipped=4)

PYTHONHASHSEED=0 npm run sim:test
# Ran 368 tests in 170.013s
# OK (skipped=4)

.venv/bin/python -m pip check
# No broken requirements found.

.venv/bin/python -m pip index versions <package>
# Checked individually for torch, torchrl, tensordict, gymnasium, pettingzoo,
# minari, and d3rlpy. Latest/local state still matches the package table above.

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m compileall -q \
  python/evolution_sim/mind python/evolution_sim/cli/mind_gate.py \
  python/evolution_sim/cli/mind_train.py
# pass

git diff --check
# pass
```

## Next Actions

The next implementation slice should not add another heuristic bypass. It
should make the learner stronger and easier to evaluate.

1. Treat `output/mind/mind-experiment-ledger.jsonl` plus this table as the
   repeat-avoidance ledger. Every `mind_gate` probe now writes a compact JSONL
   entry with trainer, artifact path, report path, seeds, ticks, fallback
   metrics, alive deltas, birth deltas, minimum per-seed deltas, strict control
   target pass/fail, and decision.
2. Treat the current IQL artifact as ready for promotion review, not as the
   final Mind architecture. The next review should decide whether to switch the
   opt-in strict candidate into the named strict Mind gate path or keep it as a
   separate neural/RL candidate while more autonomy work continues.
3. Make the next IQL slice actor-safety-first: add guard-aware negative
   examples for actions that the safety floor repeatedly suppresses, and only
   then revisit advantage temperature or CQL-style critic pessimism.
4. Keep the next critic slice calibration-first: normalize or bound Q/V value
   scale and keep the value-supported deviation path disabled until default
   gates have zero per-seed alive/birth regressions and low hard guard.
5. Add held-out critic calibration gates that compare serialized neural Q/V
   values to realized reward scale before runtime deviation thresholds can
   accept additional actions.
6. Re-run default and extended strict gates only after total fallback is below
   `0.4780`, hard guard is below `0.1190`, and every validation seed has zero
   alive/birth regression.
7. Start offline-to-online replay mixing only after the offline IQL/CQL-style
   learner beats the current contextual controls without relying on broader
   heuristic bypasses.
