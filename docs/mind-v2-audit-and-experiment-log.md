# Mind v2 Audit and Experiment Log

Date: 2026-05-08

## Purpose

This document records the current Mind v2 code audit, research checkpoint, and
experiment ledger so future Mind work does not repeat failed trainer or
delegation paths. Any new Mind trainer, calibration rule, or gate probe should
add a row here before the next implementation slice starts.

May 11 direction update: Mind v2 remains the guarded safety-floor and offline
RL comparison path, but it is no longer the main autonomy frontier. The deep
system audit found that Mind v3's no-heuristic boundary is strategically
correct while the current hand-shaped linear v3 controller is underpowered.
Future v2 work should support v3 with labels, diagnostics, and comparison
artifacts rather than widening heuristic delegation or optimizing fallback
metrics as a proxy for autonomous intelligence.

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

Relevant primary sources and package docs checked on 2026-05-08:

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
- ReBRAC shows that strong behavior regularization remains a competitive
  offline-RL baseline and should stay in the comparison set before live updates:
  <https://arxiv.org/abs/2305.09836>
- C2IQL and DiSA-IQL are newer constraint-/domain-shift-aware IQL variants. They
  reinforce the current decision to keep safety constraints explicit and
  auditable instead of mutating neural weights inside the sim tick loop:
  <https://proceedings.mlr.press/v267/liu25ai.html>,
  <https://arxiv.org/abs/2510.00358>
- DreamerV3 is a later world-model milestone, not the immediate next step,
  because this repo still needs a stable actor/critic replay loop first:
  <https://arxiv.org/abs/2301.04104>
- Quality-Diversity is relevant to open-ended artificial-life behavior after
  the controller loop is stable:
  <https://arxiv.org/abs/2406.04235>

Package freshness checked with `.venv/bin/python -m pip index versions
<package>` on 2026-05-07. The project `.venv` has the optional ML stack; bare
system `python3` does not, so npm-script tests skip torch-specific trainer tests
unless the `.venv` interpreter is used explicitly.

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
3. The old `load_learned_policy()` assert-based parsing assumption has been
   removed. Keep future artifact/runtime parsers on explicit exceptions so
   optimized Python cannot bypass validation.
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
7. Stateful online policy evaluation must use fresh policy instances per seed.
   `mind_policy_eval` and `mind_gate` now do this for `autonomous-online`; keep
   that invariant for any future replay-mixed online learner.

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
| Torch discrete IQL, guard-aware learned replay full-weight probe | `output/mind/mind-v2-torch-discrete-iql-guard-aware-replay-report.json` | fail | `0.0948` | `0.3854` | `0.4802` | `0.0` | `0.0` | rejected two-seed probe; learned rollouts as full labels collapsed actor margin and increased delegation |
| Torch discrete IQL, guard-aware weighted learned replay two-seed probe | `output/mind/mind-v2-torch-discrete-iql-guard-aware-weighted-replay-report.json` | pass | `0.0958` | `0.3779` | `0.4737` | `0.0` | `0.0` | weighted replay fixed the full-label failure on seeds `5,13`; not enough for promotion without default/extended coverage |
| Torch discrete IQL, guard-aware weighted learned replay default probe | `output/mind/mind-v2-torch-discrete-iql-guard-aware-weighted-replay-default-report.json` | fail | `0.0987` | `0.3794` | `0.4781` | `0.0` | `0.0` | near miss on default seeds `5,13,19,29`; total fallback misses strict cap by `0.0002`, so do not promote |
| Torch discrete IQL, guard-aware replay strict delegate-margin sweep | `output/mind/mind-v2-torch-discrete-iql-guard-aware-replay-delegate221-default-report.json` | fail | `0.1252` | `0.3530` | `0.4782` | `0.0` | `0.0` | first probe to reduce delegation below `0.3661`, but hard guard violates the strict cap; rejected and not retained as default |
| Torch discrete IQL, shared-encoder CQL replay probe | `output/mind/mind-v2-torch-discrete-iql-cql-replay-report.json` | fail | `0.0943` | `0.3846` | `0.4789` | `0.0` | `0.0` | rejected two-seed probe; CQL reduced hard guard but distorted actor fit and raised delegation past the strict total cap |
| Torch discrete IQL, critic-head-only CQL replay probe | `output/mind/mind-v2-torch-discrete-iql-cql-headonly-replay-report.json` | fail | `0.0928` | `0.3900` | `0.4828` | `0.0` | `0.0` | rejected two-seed probe; detaching the shared encoder protected representation gradients but delegation worsened further |
| Torch discrete IQL, CQL plus behavior anchor replay probe | `output/mind/mind-v2-torch-discrete-iql-cql-anchor-replay-report.json` | fail | `0.0940` | `0.3864` | `0.4804` | `0.0` | `0.0` | rejected two-seed probe; a `0.25` behavior anchor did not offset CQL-induced delegation |
| Torch discrete IQL, light CQL plus strong behavior anchor replay probe | `output/mind/mind-v2-torch-discrete-iql-cql-light-anchor-replay-report.json` | fail | `0.0946` | `0.3890` | `0.4836` | `0.0` | `0.0` | rejected two-seed probe; reducing CQL to `0.005` and raising the behavior anchor to `1.0` still missed total fallback |
| Torch discrete IQL, behavior-anchor-only replay probe | `output/mind/mind-v2-torch-discrete-iql-behavior-anchor-replay-report.json` | pass | `0.0924` | `0.3810` | `0.4734` | `0.0` | `0.0` | passed seeds `5,13`, but delegation rose versus weighted replay; requires default-seed validation |
| Torch discrete IQL, behavior-anchor-only replay default probe | `output/mind/mind-v2-torch-discrete-iql-behavior-anchor-replay-default-report.json` | fail | `0.0950` | `0.3844` | `0.4794` | `0.0` | `0.0` | rejected default probe; active behavior anchor lowers hard guard but misses total fallback, so CQL/anchor loss weights remain `0.0` by default |
| Default contextual prior, autonomous-online runtime comparison | `output/mind/mind-v1-gate-runtime-mode-comparison-report.json` | fail | `0.0000` | `0.0000` | `0.0000` | `-28.5` | `-18.75` | gateable heuristic-free contextual adapter is rejected; zero fallback is meaningless when held-out populations collapse |
| Torch discrete IQL strict candidate, autonomous-online runtime comparison | `output/mind/mind-v2-iql-autonomous-online-default-eval.json` | fail | `0.0000` | `0.0000` | `0.0000` | `-30.25` | `-21.5` | IQL guarded mode remains a strict-control candidate, but heuristic-free autonomous-online mode is rejected; ledger records `10501` replayable online updates and `max_online_update_count=3000` |
| Torch discrete IQL, online-update replay mixed default probe | `output/mind/mind-v2-torch-discrete-iql-online-update-replay-report.json` | pass | `0.1405` | `0.3324` | `0.4729` | `0.0` | `0.0` | not promoted; replayed signed online-update feedback reduces delegation but pushes too many decisions into hard guard, and `autonomous`/`autonomous-online` comparisons still fail outcome gates |
| Viability critic v0 proxy diagnostics on strict IQL candidate | `output/mind/mind-v2-torch-discrete-iql-extended-viability-diagnostics.json` | diagnostic | -- | -- | -- | -- | -- | diagnostics-only; held-out constraint risk is `0.1143`, value-risk AUC is `0.6940`, and `stay` risk is badly underpredicted (`0.7361` realized versus `0.0487` mean score) |
| Torch discrete IQL with trained viability head | `output/mind/mind-v2-torch-discrete-iql-viability-head-diagnostics.json` | diagnostic | -- | -- | -- | -- | -- | trained component head improves held-out risk AUC to `0.7697` and raises `stay` mean risk score to `0.4320`, but worsens Brier to `0.1554`; diagnostics only, not used for actor extraction or runtime decisions |
| Torch discrete IQL with action-conditioned viability head | `output/mind/mind-v2-torch-discrete-iql-action-viability-head-diagnostics.json` | diagnostic | -- | -- | -- | -- | -- | action-conditioned component head improves held-out AUC to `0.9496` and Brier to `0.0572`; calibration split is weaker (`0.8709` AUC, `0.1106` Brier), so keep diagnostics-only |
| Action-conditioned viability on learned-policy suppression probe | `output/mind/mind-v2-torch-discrete-iql-action-viability-learned-suppression-diagnostics.json` | diagnostic | -- | -- | -- | -- | -- | learned guarded seed `43` adds `1464` suppression positives, but aggregate AUC is only `0.6757` and Brier is `0.3323`; suppression component is present but not calibrated |
| Suppression-routed action viability replay diagnostics | `output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-diagnostics.json` | diagnostic | -- | -- | -- | -- | -- | fixes action-head supervision so fallback suppression labels train the suppressed learned action, not the heuristic fallback; held-out AUC/Brier are `0.9443`/`0.0881`, calibration AUC/Brier are `0.8570`/`0.1470`, and learned suppression component improves to `0.6425` AUC and `0.2924` Brier |
| Suppression-routed action viability replay policy eval | `output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-policy-eval.json` | fail | `0.0890` | `0.3910` | `0.4800` | `0.0` | `0.0` | rejected for promotion; hard guard improves but delegation rises past the strict `0.4780` total fallback target, and `autonomous-online` still fails with alive/birth deltas `-35.5`/`-26.0` |
| Observed-mask action viability weights diagnostics | `output/mind/mind-v2-torch-discrete-iql-observed-action-viability-weights-diagnostics.json` | diagnostic | -- | -- | -- | -- | -- | fixes action-viability positive weights to use only observed labels; suppression pos weight is now `1.0576` instead of the global `8.0`, state-head suppression supervision is disabled, held-out Brier/AUC are `0.0555`/`0.9426`, calibration Brier/AUC are `0.1036`/`0.8542`, and learned-suppression component Brier/AUC improve to `0.3788`/`0.6789` but remain overconfident |
| Observed-mask action viability weights policy eval | `output/mind/mind-v2-torch-discrete-iql-observed-action-viability-weights-policy-eval.json` | fail | `0.0880` | `0.3945` | `0.4825` | `0.0` | `0.0` | rejected for promotion; hard guard stays below target but delegation rises, total fallback misses `0.4780`, and `autonomous-online` still fails with alive/birth deltas `-28.25`/`-23.5` despite `9868` replayable update traces |
| Torch discrete IQL, behavior-margin plus pure standardized actor extraction | `output/mind/mind-v2-torch-discrete-iql-calibrated-actor-blend075-policy-eval.json` | pass | `0.0627` | `0.4208` | `0.4835` | `0.0` | `0.0` | rejected; pure batch-standardized IQL advantage weights collapsed survival-critical class fit and worsened total fallback, while `autonomous-online` still failed with alive/birth deltas `-34.75`/`-27.75` |
| Torch discrete IQL, behavior-anchored calibrated actor extraction default gate | `output/mind/mind-v2-torch-discrete-iql-anchored-calibrated-actor-blend075-gate-report.json` | pass | `0.0568` | `0.4172` | `0.4740` | `0.0` | `0.0` | strict-control candidate; behavior-anchored standardized actor weights beat the default contextual-prior fallback target and the prior behavior-margin blend, but `autonomous-online` remains rejected (`-31.25` alive, `-25.5` births) |
| Torch discrete IQL, behavior-anchored calibrated actor extraction extended gate | `output/mind/mind-v2-torch-discrete-iql-anchored-calibrated-actor-blend075-extended-report.json` | pass | `0.0582` | `0.4118` | `0.4700` | `0.0` | `0.0` | strict-control candidate on seeds `5,13,19,29,37,41` at `120,180` ticks; guarded promotion-review signal, not heuristic-free autonomy |
| Torch discrete IQL, discounted-return critic calibration probe | `output/mind/mind-v2-torch-discrete-iql-return-calibrated-actor-blend075-policy-eval.json` | pass | `0.0598` | `0.4179` | `0.4777` | `0.0` | `0.0` | rejected; Q/V discounted-return auxiliary loss records critic-scale diagnostics but misses the current `0.4740` guarded control and worsens autonomous-online to `-35.5` alive, `-29.25` births |
| Torch discrete IQL, constraint-aware actor extraction probe | `output/mind/mind-v2-torch-discrete-iql-constraint-aware-actor-blend075-policy-eval.json` | pass | `0.0575` | `0.4192` | `0.4767` | `0.0` | `0.0` | rejected; observed viability-risk actor filtering touched `12.8%` of logged actions and improved autonomous-online deltas to `-27.0` alive, `-19.25` births, but still misses guarded fallback |
| Torch discrete IQL, risk-adjusted actor distillation probe | `output/mind/mind-v2-torch-discrete-iql-risk-adjusted-actor-blend075-policy-eval.json` | pass | `0.0572` | `0.4187` | `0.4759` | `0.0` | `0.0` | rejected; detached Q-minus-risk actor target improves over the logged-risk filter but still misses the current `0.4740` guarded control and worsens autonomous-online to `-36.25` alive, `-29.0` births |
| Torch discrete IQL, calibrated supported actor extraction v1 | `output/mind/mind-v2-torch-discrete-iql-calibrated-supported-actor-blend075-gate-report.json` | fail | `0.1047` | `0.3712` | `0.4759` | `0.0` | `0.0` | rejected against the current promotion-review control (`0.0568` hard guard, `0.4740` total fallback); calibration-bank Brier/ECE improved (`0.054816`/`0.088814` raw to `0.035617`/`0.032653` calibrated), but extracted targets still over-concentrate on `eat`/`drink`, pushing delegation down by converting too many decisions into hard guard; `autonomous-online` remains rejected (`-34.5` alive, `-28.75` births) |
| Torch discrete IQL, contextual behavior-proximity supported actor extraction v2 cap-fix rerun | `output/mind/mind-v2-torch-discrete-iql-contextual-behavior-fixedcaps-actor-blend075-gate-report.json` | fail | `0.1167` | `0.3550` | `0.4717` | `0.0` | `0.0` | useful but not promoted; after fixing final cap enforcement, the gate completed with artifact diagnostics skipped, target/logged TVD is `0.1800`, final movement/stay-to-resource conversion is capped at `0.1800` (`2,489/13,830`), and total fallback improves, but hard guard remains far above the current promotion-review control `0.0568`; `autonomous-online` fails worse at `-37.0` alive and `-31.5` births. |
| Torch discrete IQL, contextual behavior-prior actor regularization only | `output/mind/mind-v2-torch-discrete-iql-contextual-prior-regularized-actor-blend075-gate-report.json` | fail | `0.0663` | `0.4037` | `0.4700` | `0.0` | `0.0` | useful but not promoted; actor-side contextual behavior-prior cross-entropy sharply reduces hard guard versus v2 caps (`0.1167` to `0.0663`) and beats total fallback (`0.4700`), but still misses the current promotion-review hard-guard control `0.0568`; no contextual target extraction is enabled, so target/logged TVD and movement/stay-to-resource target conversion are not applicable; `autonomous-online` remains rejected (`-35.0` alive, `-28.5` births) |
| Torch discrete IQL, contextual behavior-prior plus v2 target extraction | `output/mind/mind-v2-torch-discrete-iql-contextual-behavior-prior-fixedcaps-actor-blend075-gate-report.json` | fail | `0.1031` | `0.3696` | `0.4727` | `0.0` | `0.0` | rejected; adding the behavior-prior objective to v2 target extraction improves hard guard versus the cap-fix rerun (`0.1167` to `0.1031`) but worsens total fallback (`0.4717` to `0.4727`) and remains far above promotion-review hard guard; extracted target diagnostics are unchanged by design because v2 targets are precomputed before actor finetune: target/logged TVD `0.1800`, movement/stay-to-resource `0.1800` (`2,489/13,830`) |
| Torch discrete IQL, contextual behavior-prior with guard-feedback finetune carryover | `output/mind/mind-v2-torch-discrete-iql-contextual-prior-finetune-guard-carryover-actor-blend075-gate-report.json` | fail | `0.0574` | `0.4199` | `0.4773` | `0.0` | `0.0` | useful but not promoted; carrying runtime-suppressed learned-action margin loss into actor finetune nearly closes the hard-guard gap but shifts too much mass into confidence delegation; `autonomous-online` remains rejected (`-31.5` alive, `-24.25` births) |
| Torch discrete IQL, contextual behavior-prior with finetune guard and behavior-margin carryovers | `output/mind/mind-v2-torch-discrete-iql-contextual-prior-finetune-carryovers-actor-blend075-gate-report.json` | fail | `0.0582` | `0.4155` | `0.4737` | `0.0` | `0.0` | closest non-risk prior-only variant; total fallback beats the `0.4740` control, but hard guard still misses `0.0568`; artifact reports `1,459` finetune guard-feedback rows and `41,263` finetune behavior-margin anchor rows; `autonomous-online` remains rejected (`-36.75` alive, `-30.0` births) |
| Torch discrete IQL, risk-adjusted contextual behavior-prior with finetune carryovers | `output/mind/mind-v2-torch-discrete-iql-risk-adjusted-contextual-prior-finetune-carryovers-actor-blend075-gate-report.json` | fail | `0.0567` | `0.4179` | `0.4746` | `0.0` | `0.0` | current closest actor-objective candidate; hard guard beats the `0.0568` promotion-review control and guarded outcomes stay flat, but total fallback misses by `0.0006`; `autonomous-online` remains rejected (`-38.5` alive, `-32.25` births) |
| Torch discrete IQL, constraint-plus-risk contextual behavior-prior with finetune carryovers | `output/mind/mind-v2-torch-discrete-iql-constraint-risk-contextual-prior-finetune-carryovers-actor-blend075-gate-report.json` | fail | `0.0562` | `0.4236` | `0.4798` | `0.0` | `0.0` | rejected; adding observed-risk actor filtering lowers hard guard but worsens delegation and total fallback; do not stack constraint-aware actor extraction with this prior path by default |
| Torch discrete IQL, risk-adjusted contextual behavior-prior with stronger finetune margin anchor | `output/mind/mind-v2-torch-discrete-iql-risk-adjusted-contextual-prior-strong-finetune-margin-actor-blend075-gate-report.json` | fail | `0.0566` | `0.4237` | `0.4803` | `0.0` | `0.0` | rejected; increasing the finetune behavior-margin anchor weight from `0.15` to `0.20` worsened total fallback, so the code was restored to the base `0.15` finetune weight |
| Torch discrete IQL, risk-adjusted contextual behavior-prior with hard-guard-only finetune carryover | `output/mind/mind-v2-torch-discrete-iql-risk-adjusted-contextual-prior-hard-guard-finetune-carryover-actor-blend075-gate-report.json` | fail | `0.0569` | `0.4189` | `0.4758` | `0.0` | `0.0` | rejected; hard-only finetune carryover reduced the finetune feedback bank from `1,459` to `257` rows but was worse than all runtime-suppressed feedback on both hard guard and total fallback, so the code was restored to all-feedback carryover |
| Torch discrete IQL, risk-adjusted contextual behavior-prior with delegate-margin finetune objective | `output/mind/mind-v2-torch-discrete-iql-risk-adjusted-contextual-prior-delegate-margin-finetune-actor-blend075-gate-report.json` | fail | `0.0577` | `0.4187` | `0.4764` | `0.0` | `0.0` | rejected; adding a runtime-delegate-margin actor finetune term worsened both hard guard and total fallback versus the `0.0567`/`0.4746` risk-adjusted carryover candidate, so the objective was not retained |
| Torch discrete IQL, anchored calibrated actor with runtime-feedback finetune | `output/mind/mind-v2-torch-discrete-iql-anchored-calibrated-finetune-actor-blend075-gate-report.json` | fail | `0.0589` | `0.4203` | `0.4792` | `0.0` | `0.0` | rejected; carrying the finetune guard/behavior-margin losses into the current calibrated control reduced actor top-1 accuracy to `0.229929` and worsened the guarded fallback boundary, so calibrated actor extraction remains outside the finetune loop |
| Torch discrete IQL, anchored calibrated actor with value-supported deviation source fix and strict thresholds | `output/mind/mind-v2-torch-discrete-iql-anchored-calibrated-value-deviation-sourcefix-actor-blend075-gate-report.json` | fail | `0.0568` | `0.4172` | `0.4740` | `0.0` | `0.0` | rejected as no-op; fixing neural-prior score-source handling is retained, but the existing `score_margin >= 0.25` and positive-advantage thresholds produced `safe_deviation_count=0` and did not move the boundary |
| Torch discrete IQL, anchored calibrated actor with relaxed value-supported deviation | `output/mind/mind-v2-torch-discrete-iql-anchored-calibrated-value-deviation-advneg010-actor-blend075-gate-report.json` | fail | `0.0581` | `0.4202` | `0.4783` | `-5.0` | `-5.0` | rejected; hand-count diagnostics showed positive predicted advantage blocked all candidates, but relaxing it to `-0.10` allowed only `46` `eat` safe deviations and regressed guarded survival, so the torch-IQL value-deviation opt-in was not retained |
| Torch discrete IQL, suppression-aware critic calibration plus risk-adjusted contextual behavior prior | `output/mind/mind-v2-torch-discrete-iql-suppression-critic-risk-contextual-prior-finetune-carryovers-actor-blend075-gate-report.json` | fail | `0.0576` | `0.4218` | `0.4794` | `0.0` | `0.0` | rejected; the opt-in Q/V margin loss trained `1,459` runtime-suppressed rows and improved one-step Q MAE, but it pulled the risk-adjusted actor target away from logged behavior and worsened both hard guard and total fallback; autonomous-online improved versus the closest risk-adjusted carryover candidate but still failed (`-31.75` alive, `-26.0` births) |
| Torch discrete IQL, suppression-aware critic calibration plus non-risk contextual behavior prior | `output/mind/mind-v2-torch-discrete-iql-suppression-critic-contextual-prior-finetune-carryovers-actor-blend075-gate-report.json` | fail | `0.0588` | `0.4195` | `0.4783` | `0.0` | `0.0` | rejected; removing detached Q-minus-risk actor distillation did not rescue the suppression critic objective, so the blunt suppressed-action Q/V margin should remain diagnostic-only and off by default |
| Torch discrete IQL, anchored calibrated actor local auto-device rerun | `output/mind/audit/mind-iql-anchored-calibrated-blend075-local-auto-report.json` | pass | `0.0649` | `0.4075` | `0.4724` | `0.0` | `0.0` | not promoted; `--torch-device auto` resolved to MPS on this macOS ARM runner and CUDA was unavailable, but hard guard regressed versus the `0.0568` promotion-review control despite improved total fallback; `autonomous-online` still failed (`-31.5` alive, `-24.0` births) and the full gate took `1099.6261s`, confirming gate evaluation cost is a local bottleneck |
| Torch discrete IQL, anchored calibrated actor local auto-device with evaluation workers | `output/mind/audit/mind-iql-anchored-calibrated-blend075-local-auto-evalworkers2-report.json` | pass | `0.0674` | `0.4081` | `0.4755` | `0.0` | `0.0` | not promoted; `--evaluation-workers 2` recorded two effective evaluation workers and reduced wall time only to `1012.9436s`, because training/artifact diagnostics dominated before the parallel section; the MPS rerun also drifted from the prior guarded metrics, so use MPS/auto for smoke and CPU or controlled CUDA runs for promotion-grade comparisons |
| Torch discrete IQL, anchored calibrated actor local auto-device with diagnostics and evaluation workers | `output/mind/audit/mind-iql-anchored-calibrated-blend075-local-auto-diagworkers2-evalworkers2-report.json` | pass | `0.0643` | `0.4074` | `0.4717` | `0.0` | `0.0` | not promoted; both process pools were active and wall time fell to `812.7058s`, but phase timings show artifact diagnostics still dominate (`660.8851s`) versus training (`37.4958s`) and evaluation (`111.5268s`); `autonomous-online` still failed (`-34.0` alive, `-27.0` births), so this is an infra/profiling win, not a controller win |
| Torch discrete IQL, anchored calibrated actor local auto-device with sharded diagnostics and evaluation workers | `output/mind/audit/mind-iql-anchored-calibrated-blend075-local-auto-shardeddiag8-evalworkers2-report.json` | pass | `0.0657` | `0.4113` | `0.4770` | `0.0` | `0.0` | not promoted; artifact diagnostics now shard by trajectory dataset (`18` diagnostic tasks, `8` effective workers), reducing total wall time to `301.8127s` and artifact diagnostics to `151.8482s`; `autonomous-online` still failed (`-32.25` alive, `-24.0` births), so this is a real local throughput win and still not a controller win |

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
# Ran 95 tests in 20.235s
# OK (skipped=4)

PYTHONHASHSEED=0 npm run sim:test
# Ran 369 tests in 172.781s
# OK (skipped=4)

.venv/bin/python -m pip check
# No broken requirements found.

.venv/bin/python -m pip index versions <package>
# Checked individually for torch, torchrl, tensordict, gymnasium, pettingzoo,
# minari, and d3rlpy. Latest/local state still matches the package table above.

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m compileall -q python
# pass

git diff --check
# pass
```

Fresh checks from the gateable autonomy slice:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_mind_v1
# Ran 104 tests in 22.492s
# OK (skipped=4)

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m compileall -q python
# pass

PYTHONHASHSEED=0 npm run sim:test
# Ran 378 tests in 169.235s
# OK (skipped=4)

PYTHONHASHSEED=0 PYTHONPATH=python npm run sim:mind:evaluate -- \
  --artifact output/mind/mind-v2-torch-discrete-iql-extended-artifact.json \
  --enable-mind \
  --seeds 5,13,19,29 \
  --ticks 120 \
  --compare-runtime-mode autonomous-online \
  --output output/mind/mind-v2-iql-autonomous-online-default-eval.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# guarded IQL status = pass
# guarded hard_guard = 0.1041
# guarded heuristic_delegate = 0.3707
# guarded total_fallback = 0.4748
# guarded alive_delta = 0.0
# guarded births_delta = 0.0
# autonomous-online status = fail
# autonomous-online hard_guard = 0.0
# autonomous-online heuristic_delegate = 0.0
# autonomous-online total_fallback = 0.0
# autonomous-online alive_delta = -30.25
# autonomous-online births_delta = -21.5
# autonomous-online policy_update_trace_count = 10501

.venv/bin/python -m pip check
# No broken requirements found.

.venv/bin/python -m pip index versions <package>
# torch 2.11.0, torchrl 0.12.0, tensordict 0.12.2, gymnasium 1.3.0,
# pettingzoo 1.26.1, and minari 0.5.3 are current; d3rlpy 2.8.1 is latest
# but still pulls gymnasium==1.0.0 in a d3rlpy==2.8.1 dry run, so keep local
# d3rlpy 2.8.0.

git diff --check
# pass
```

Fresh checks from the replay-mixed offline-to-online slice:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_trajectory_writer_can_persist_policy_update_trace_opt_in \
  python.tests.test_mind_v1.MindV1Tests.test_replay_online_update_traces_rejects_invalid_adjustment_bounds
# Ran 2 tests
# OK

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_trainer_writes_real_ml_artifact \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_artifact_rejects_runtime_value_deviation
# Ran 2 tests
# OK

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m evolution_sim.cli.collect_trajectory \
  --seed 1 \
  --ticks 120 \
  --output output/trajectories/mind-v2-iql-autonomous-online-extra/seed1-autonomous-online-ticks120.jsonl.gz \
  --split-id mind-v2-iql-autonomous-online-extra \
  --mind-artifact output/mind/mind-v2-torch-discrete-iql-extended-artifact.json \
  --enable-mind \
  --mind-runtime-mode autonomous-online \
  --include-policy-diagnostics \
  --include-policy-update-trace
# alive_agents = 11
# births = 10
# trajectory_records = 1617

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m evolution_sim.cli.collect_trajectory \
  --seed 2 \
  --ticks 120 \
  --output output/trajectories/mind-v2-iql-autonomous-online-extra/seed2-autonomous-online-ticks120.jsonl.gz \
  --split-id mind-v2-iql-autonomous-online-extra \
  --mind-artifact output/mind/mind-v2-torch-discrete-iql-extended-artifact.json \
  --enable-mind \
  --mind-runtime-mode autonomous-online \
  --include-policy-diagnostics \
  --include-policy-update-trace
# alive_agents = 23
# births = 26
# trajectory_records = 2613

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_gate \
  --reuse-trajectories \
  --trainer torch-discrete-iql \
  --extra-trajectory output/trajectories/mind-v2-iql-autonomous-online-extra/seed1-autonomous-online-ticks120.jsonl.gz \
  --extra-trajectory output/trajectories/mind-v2-iql-autonomous-online-extra/seed2-autonomous-online-ticks120.jsonl.gz \
  --compare-runtime-mode autonomous \
  --compare-runtime-mode autonomous-online \
  --artifact-output output/mind/mind-v2-torch-discrete-iql-online-update-replay-artifact.json \
  --output output/mind/mind-v2-torch-discrete-iql-online-update-replay-report.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# guarded status = pass
# decision = gate_pass_not_promoted
# hard_guard = 0.1405
# heuristic_delegate = 0.3324
# total_fallback = 0.4729
# autonomous alive_delta = -29.0
# autonomous births_delta = -22.25
# autonomous-online alive_delta = -32.25
# autonomous-online births_delta = -26.0
# autonomous-online policy_update_trace_count = 9366

node -e "JSON.parse(require('fs').readFileSync('output/mind/mind-v2-torch-discrete-iql-online-update-replay-artifact.json','utf8'))"
# pass; regenerated artifact is strict JSON with finite training metrics

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m compileall -q python
# pass

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_mind_v1
# Ran 105 tests in 22.128s
# OK (skipped=4)

PYTHONHASHSEED=0 npm run sim:test
# Ran 379 tests in 170.481s
# OK (skipped=4)

git diff --check
# pass
```

Fresh checks from the viability critic v0 diagnostics slice:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_trajectory_writer_rejects_nonfinite_policy_diagnostics \
  python.tests.test_mind_v1.MindV1Tests.test_mind_gate_rejects_unreplayable_extra_update_trace \
  python.tests.test_mind_v1.MindV1Tests.test_neural_actor_critic_trainer_writes_deterministic_artifact \
  python.tests.test_mind_v1.MindV1Tests.test_trajectory_jsonl_loader_rejects_nonfinite_policy_diagnostics
# Ran 4 tests
# OK

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_trainer_writes_real_ml_artifact \
  python.tests.test_mind_v1.MindV1Tests.test_torch_discrete_iql_artifact_rejects_runtime_value_deviation
# Ran 2 tests
# OK

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m compileall -q python
# pass

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_mind_v1
# Ran 107 tests in 22.874s
# OK (skipped=4)

PYTHONHASHSEED=0 npm run sim:test
# Ran 381 tests in 165.354s
# OK (skipped=4)

PYTHONHASHSEED=0 PYTHONPATH=python python3 - <<'PY'
# Built held-out viability diagnostics for
# output/mind/mind-v2-torch-discrete-iql-extended-artifact.json
# against artifact-diagnostic seeds 5,13,19,29.
# constraint_risk_rate = 0.1143
# risk_auc = 0.6940
# stay risk_rate = 0.7361
# stay mean_risk_score = 0.0487
PY

npm run sim:mind:diagnostics -- \
  --artifact output/mind/mind-v2-torch-discrete-iql-extended-artifact.json \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed5-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed13-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed19-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed29-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-extended-viability-diagnostics.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# pass; writes mind_artifact_diagnostics_report_v1 with viability_calibration

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_train \
  --trainer torch-discrete-iql \
  --trajectory output/trajectories/mind-v1-gate/seed1-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed2-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed3-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed4-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed6-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed7-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed8-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed9-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed10-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed11-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed12-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed17-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed23-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed31-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-viability-head-artifact.json
# trained_record_count = 44324

npm run sim:mind:diagnostics -- \
  --artifact output/mind/mind-v2-torch-discrete-iql-viability-head-artifact.json \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed5-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed13-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed19-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed29-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-viability-head-diagnostics.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# risk_score_policy = multi_component_constraint_viability_head_v1
# constraint_risk_rate = 0.1143
# risk_auc = 0.7697
# risk_brier_score = 0.1554
# stay mean_risk_score = 0.4320

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_train \
  --trainer torch-discrete-iql \
  --trajectory output/trajectories/mind-v1-gate/seed1-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed2-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed3-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed4-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed6-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed7-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed8-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed9-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed10-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed11-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed12-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed17-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed23-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed31-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-action-viability-head-artifact.json
# trained_record_count = 44324

npm run sim:mind:diagnostics -- \
  --artifact output/mind/mind-v2-torch-discrete-iql-action-viability-head-artifact.json \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed5-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed13-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed19-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed29-ticks120.jsonl.gz \
  --calibration-trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed37-ticks120.jsonl.gz \
  --calibration-trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed41-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-action-viability-head-diagnostics.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# held-out risk_auc = 0.9496
# held-out risk_brier_score = 0.0572
# held-out stay mean_risk_score = 0.9515
# calibration risk_auc = 0.8709
# calibration risk_brier_score = 0.1106

npm run sim:trajectory -- \
  --seed 43 \
  --ticks 120 \
  --output output/trajectories/mind-v2-action-viability-learned-diagnostics/seed43-guarded-ticks120.jsonl.gz \
  --split-id mind_v2_action_viability_suppression_probe \
  --mind-artifact output/mind/mind-v2-torch-discrete-iql-action-viability-head-artifact.json \
  --enable-mind \
  --include-policy-diagnostics
# trajectory_records = 3008

npm run sim:mind:diagnostics -- \
  --artifact output/mind/mind-v2-torch-discrete-iql-action-viability-head-artifact.json \
  --trajectory output/trajectories/mind-v2-action-viability-learned-diagnostics/seed43-guarded-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-action-viability-learned-suppression-diagnostics.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# suppression target positives = 1464
# risk_auc = 0.6757
# risk_brier_score = 0.3323

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_train \
  --trainer torch-discrete-iql \
  --trajectory output/trajectories/mind-v1-gate/seed1-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed2-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed3-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed4-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed6-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed7-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed8-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed9-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed10-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed11-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed12-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed17-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed23-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/seed31-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v2-action-viability-learned-diagnostics/seed43-guarded-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-artifact.json
# trained_record_count = 47332
# action_viability_suppression_positive_count = 1459
# action_viability_logged_suppression_positive_count = 0

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_artifact_diagnostics \
  --artifact output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-artifact.json \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed5-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed13-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed19-ticks120.jsonl.gz \
  --trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed29-ticks120.jsonl.gz \
  --calibration-trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed37-ticks120.jsonl.gz \
  --calibration-trajectory output/trajectories/mind-v1-gate/artifact-diagnostics/seed41-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-diagnostics.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# held-out risk_auc = 0.9443
# held-out risk_brier_score = 0.0881
# calibration risk_auc = 0.8570
# calibration risk_brier_score = 0.1470

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_artifact_diagnostics \
  --artifact output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-artifact.json \
  --trajectory output/trajectories/mind-v2-action-viability-learned-diagnostics/seed43-guarded-ticks120.jsonl.gz \
  --output output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-learned-suppression-diagnostics.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# suppression component risk_auc = 0.6425
# suppression component risk_brier_score = 0.2924

PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.mind_policy_eval \
  --artifact output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-artifact.json \
  --enable-mind \
  --seeds 5,13,19,29 \
  --ticks 120 \
  --compare-runtime-mode autonomous-online \
  --output output/mind/mind-v2-torch-discrete-iql-suppression-action-viability-policy-eval.json \
  --experiment-ledger-output output/mind/mind-experiment-ledger.jsonl
# guarded hard_guard = 0.0890
# guarded heuristic_delegate = 0.3910
# guarded total_fallback = 0.4800
# autonomous-online alive_delta = -35.5
# autonomous-online births_delta = -26.0

PYTHONHASHSEED=0 PYTHONPATH=python python3 -m unittest python.tests.test_mind_v1
# Ran 109 tests in 24.566s
# OK (skipped=4)

npm run sim:test
# Ran 383 tests in 169.081s
# OK (skipped=4)

git diff --check
# pass
```

## Next Actions

Latest runtime slice: the simulator now has opt-in `autonomous` and
`autonomous-online` Mind modes. `autonomous` removes heuristic guard/delegation
at artifact load time; `autonomous-online` adds
`in_run_contextual_bandit_adapter_v1`, which learns bounded context/action score
offsets from finalized reward records through a policy feedback hook. This is
observable autonomy work, not promotion proof: neural weights remain immutable
during the run, and strict default/extended gates still decide whether any
heuristic-free controller can replace the safety floor. The first smoke outputs
are `output/sim-runs/mind-autonomous-online-viewer-check.json` for viewer
inspection and
`output/trajectories/mind-autonomous-online-seed7-ticks120.jsonl.gz` with
diagnostics; that trajectory wrote `2592` adaptive diagnostics and reached
`online_update_count=2564`.

Gateable autonomy update: `autonomous-online` now emits
`mind_policy_update_trace_v1` when trajectory collection opts in with
`--include-policy-update-trace`. The trace records the reward signal and bounded
context/action offset transition, and `mind_gate --compare-runtime-mode
autonomous-online` evaluates this stateful runtime with a fresh policy instance
per validation run. Runtime-mode comparison summaries are appended to the
experiment ledger entry, including fallback, alive/birth deltas, and update
counts. This makes autonomy probes reproducible enough to reject or promote
through gates, but it still does not authorize in-simulation neural weight
mutation.

Replay-mixed autonomy update: `torch-discrete-iql` now reads replayed
`mind_policy_update_trace_v1` records through the trajectory-to-transition
adapter and applies a signed actor-margin loss for the action that the online
adapter updated. The first default probe mixed two autonomous-online traces
(`4219` non-zero update-feedback records, `2830` positive and `1389` negative)
with the reused gate trajectories. It passed the guarded outcome gate and
lowered total fallback to `0.4729`, but hard guard rose to `0.1405`, so it is
not promotion progress. Runtime comparisons remained rejected:
`autonomous` alive/birth deltas were `-29.0` and `-22.25`;
`autonomous-online` alive/birth deltas were `-32.25` and `-26.0` with `9366`
update traces. Do not add more negative autonomous rollouts blindly; the next
slice needs a stronger constraint/viability model so hard guard and delegation
fall together.

Viability critic v0 update: artifact diagnostics now include
`mind_viability_critic_diagnostics_v0` for neural artifacts. This does not add a
runtime bypass or change actor extraction. It labels existing transitions for
death or short survival horizon, energy/hydration/health floor risk, invalid
actions, reproduction viability, and hard-guard/delegate suppression; then it
calibrates the current neural action-value signal or trained component head with
Brier, AUC, risk buckets, per-action, trophic-role, and meat-mode breakdowns.
`sim:mind:diagnostics` writes the same diagnostics into a durable
`mind_artifact_diagnostics_report_v1` JSON file and appends a compact ledger
entry so probes are not repeated without checking prior evidence.

The strict IQL candidate's held-out proxy baseline is useful but not good enough
for promotion: constraint-risk rate `0.1143`, value-risk AUC `0.6940`, and the
logged `stay` action has risk rate `0.7361` while its mean risk score is only
`0.0487`. The first trained viability head improves ranking signal
(`risk_auc=0.7697`) and makes `stay` visibly risky (`mean_risk_score=0.4320`),
but it worsens calibration (`risk_brier_score=0.1554` versus proxy `0.1078`).
The action-conditioned head is the first useful selector-shaped signal:
held-out `risk_auc=0.9496`, `risk_brier_score=0.0572`, and logged `stay`
`mean_risk_score=0.9515`. On the separate calibration bank, the same head drops
to `risk_auc=0.8709` and `risk_brier_score=0.1106`, so it is still diagnostics
only and not a runtime acceptance policy.

Important caveat: the strict IQL artifact-diagnostic bank is heuristic-collected.
It contains death/floor/invalid labels, but no positive
`hard_guard_or_delegate_suppression` labels. Suppression calibration only becomes
meaningful on trajectories collected with learned-policy diagnostics. The first
guarded learned-policy suppression probe confirms the label exists (`1464`
positives), but aggregate calibration is weak (`risk_auc=0.6757`,
`risk_brier_score=0.3323`) because the head was trained without suppression
positives. Do not use suppression scores for actor extraction yet.

Suppression-routed replay update: the action-conditioned viability loss now
routes hard-guard/delegate suppression labels to the suppressed learned action
from `policy_decision_diagnostics.learned_action`; the resolved heuristic action
still receives the realized death/floor/invalid components, but no longer
inherits the suppression component. The replay-mixed artifact trained on seed
`43` has `1459` action-viability suppression positives and `0` logged-action
suppression positives. This improves the learned suppression component on the
suppression probe from Brier/AUC `0.4867`/`0.4448` to `0.2924`/`0.6425`, but
the held-out aggregate calibration worsens versus the previous action head
(`0.0881` Brier versus `0.0572`) and the guarded policy eval misses strict
control total fallback (`0.4800` versus `0.4780`). Keep this as a rejected
diagnostic checkpoint, not an actor-extraction source.

Learned-only suppression supervision update: the suppression component is now
observed only when a learned-policy decision exists. Heuristic-only trajectories
continue to supervise realized death/floor/invalid components for the resolved
action, but they no longer flood the suppression component with unobserved
negative labels. Artifact diagnostics also mask the suppression component on
heuristic-only rows and report per-component observed counts, so held-out
heuristic banks are not scored as if every row were a true suppression negative.
The v2 replay-mixed artifact trained with `3002` suppression observations and
`1459` positives. It improved guarded fallback from the prior suppression-routed
`0.4800` to `0.4784` total (`0.0885` hard guard plus `0.3899` delegation), but
still misses the strict `0.4780` total target. Held-out viability calibration is
usable (`0.0639` Brier, `0.9374` AUC) and the calibration bank is acceptable but
weaker (`0.1185` Brier, `0.8478` AUC). The learned-suppression probe remains
badly overconfident: aggregate Brier/AUC `0.4175`/`0.5908`, suppression
component Brier/AUC `0.4776`/`0.6693`. A narrower delegate-margin probe at
`0.245` is rejected because it raised hard guard and still missed total fallback
(`0.4785`). Keep v2 label semantics, but do not use the suppression score for
actor extraction yet.

Detached viability-head representation probe: detaching the state/action
viability losses from the shared actor/Q/V hidden representation was measured
as an opt-in torch-IQL experiment, not accepted as the new default. The artifact
`output/mind/mind-v2-torch-discrete-iql-detached-suppression-action-viability-artifact.json`
trained with `viability_representation_policy =
detached_shared_hidden_auxiliary_heads_v1` and the same seed `43`
suppression-replay mix. It preserved the suppression supervision counts
(`1459` action-viability suppression positives and `0` logged-action
suppression positives), but worsened the risk head and policy metrics:
held-out Brier/AUC `0.1590`/`0.8568`, calibration-bank Brier/AUC
`0.2036`/`0.7624`, learned-suppression aggregate Brier/AUC
`0.2778`/`0.5669`, and guarded fallback `0.4825` total
(`0.0939` hard guard plus `0.3886` delegation). The default trainer stays on
`shared_hidden_auxiliary_heads_v1`; detached heads are kept only behind the
`--torch-iql-detach-viability-heads` experiment flag to avoid repeating this
candidate as promotion progress.

Observed-mask action-viability weight fix: action-head positive weights now use
only observed labels for each component, and the state viability head no longer
trains `hard_guard_or_delegate_suppression`. In the replay-mixed seed `43`
artifact this changes the suppression positive weight from the previous global
clamp of `8.0` to the observed learned-policy balance `1.0576`
(`3002` observed, `1459` positive). Held-out/calibration diagnostics improve to
Brier/AUC `0.0555`/`0.9426` and `0.1036`/`0.8542`; learned-suppression
diagnostics improve to aggregate Brier/AUC `0.3609`/`0.6432` and suppression
component Brier/AUC `0.3788`/`0.6789`. This is still rejected for promotion:
guarded fallback worsens to `0.4825` total (`0.0880` hard guard plus `0.3945`
delegation), and `autonomous-online` still fails with alive/birth deltas
`-28.25`/`-23.5`. Keep the weighting fix because it closes the audit hole, but
do not use suppression scores for actor extraction yet.

Behavior-margin actor anchor and neural-prior blend sweep: the viability-safe
logged-action margin anchor is implemented as an opt-in torch-IQL actor loss.
It trains on legal logged actions whose non-suppression viability targets are
all negative, with `behavior_margin_anchor_eligible_rate=0.871778` on the
seed `43` replay mix. The default prior blend remains `0.9`; the sweep made the
blend explicit only for `guarded_torch_discrete_iql_v1` artifacts and measured
`0.85`, `0.75`, and `0.70` without changing guard/delegate thresholds. The
best point was `0.75`: guarded hard guard `0.0576`, delegation `0.4207`, total
fallback `0.4783`, and zero alive/birth deltas. This is real progress from the
observed-mask candidate's `0.4825` total, but it still misses the strict
`0.4780` fallback target and autonomous-online remains rejected
(`-35.0` alive, `-28.5` births). The `0.70` point worsened total fallback to
`0.4785`, so further blend tuning is not the right next promotion slice.
Machine-checkable reports were written for the best candidate:
`mind-v2-torch-discrete-iql-behavior-margin-anchor-blend075-policy-eval.json`,
`mind-v2-torch-discrete-iql-behavior-margin-anchor-blend075-diagnostics.json`,
and
`mind-v2-torch-discrete-iql-behavior-margin-anchor-blend075-learned-suppression-diagnostics.json`.

Actor extraction calibration follow-up: pure batch-standardized IQL advantage
weights were rejected because they collapsed essential class fit and worsened
guarded fallback to `0.4835` total (`0.0627` hard guard plus `0.4208`
delegation). The retained opt-in calibration is behavior-anchored instead:
standardized advantage weights are clipped, normalized, and blended back toward
plain behavior cloning with
`actor_weighting_policy =
behavior_anchored_batch_standardized_iql_advantage_weighted_bc_v2`,
`actor_advantage_calibration_weight_blend=0.35`, and
`actor_advantage_calibration_weight_min=0.25`. The train-set actor weights stay
near behavior support (`0.7141` min, `1.9325` max, `1.0017` mean). This is the
first behavior-margin line to pass both strict fallback targets: default gate
hard guard/delegate/total fallback `0.0568`/`0.4172`/`0.4740`, and extended
gate `0.0582`/`0.4118`/`0.4700`, with zero min per-seed alive/birth deltas in
both reports. Held-out diagnostics are reviewable but not solved:
actor top-1 `0.3823`, Q/V MAE `4.3502`/`4.7762`, held-out viability
Brier/AUC `0.0705`/`0.9543`, calibration-bank Brier/AUC
`0.1393`/`0.8722`, and learned-suppression Brier/AUC
`0.3369`/`0.6746`. Keep this as the new guarded promotion-review artifact, not
as a heuristic-free autonomous controller; `autonomous-online` still fails with
alive/birth deltas `-31.25`/`-25.5`.

The next implementation slice should not add another heuristic bypass. It
should make the learner stronger and easier to evaluate.

1. Treat `output/mind/mind-experiment-ledger.jsonl` plus this table as the
   repeat-avoidance ledger. Every `mind_gate` probe now writes a compact JSONL
   entry with trainer, artifact path, report path, seeds, ticks, fallback
   metrics, alive deltas, birth deltas, minimum per-seed deltas, strict control
   target pass/fail, and decision.
2. Treat the behavior-anchored calibrated IQL artifact as the current guarded
   promotion-review artifact, not as the final Mind architecture. The next
   review should decide whether to switch this opt-in strict candidate into the
   named strict Mind gate path or keep it separate while autonomy work
   continues.
3. Guard-aware replay is now executable but not promotable. CQL-style pessimism,
   pure standardized actor extraction, and runtime blend-only tuning have all
   been measured and rejected as promotion paths. The next IQL slice should
   improve critic scale or constraint-aware actor extraction so delegation falls
   without simply becoming hard-guard intervention.
4. Keep the next critic slice calibration-first: normalize or bound Q/V value
   scale and keep the value-supported deviation path disabled until default
   gates have zero per-seed alive/birth regressions and low hard guard.
5. The first replayed online-update loss is implemented and measured, but not
   promoted. Replace the contextual bandit adapter only after a constraint-aware
   offline-to-online learner beats both guarded fallback targets and
   heuristic-free runtime outcome targets on default and extended gates.
6. Add held-out critic calibration gates that compare serialized neural Q/V and
   viability-risk values to realized reward/constraint scale before runtime
   deviation thresholds can accept additional actions. Viability diagnostics v0,
   a state component head, an action-conditioned component head, and separate
   calibration-bank reporting are now in place; actor extraction still needs
   calibrated suppression-aware training before it can use the signal.
7. The first discounted-return Q/V auxiliary loss is implemented and rejected as
   a promotion path because it worsened guarded total fallback to `0.4777`; keep
   the return MAE diagnostics, but do not enable the loss by default.
8. Keep constraint-aware actor extraction opt-in until it beats the guarded
   promotion-review artifact. The first observed viability-risk actor filter is
   useful because it slightly improves heuristic-free outcome deltas, but it
   still misses total fallback (`0.4767` versus `0.4740`) and remains rejected.
9. The first detached Q-minus-risk actor distillation probe is also rejected. It
   improves guarded total fallback to `0.4759` versus the logged-risk filter, but
   it still misses the `0.4740` promotion-review artifact and worsens
   autonomous-online outcomes. The next actor slice needs calibrated
   action-risk targets or per-action support constraints before using risk
   directly as an actor target.
10. Calibrated Supported Actor Extraction v1 is implemented and rejected as a
    promotion path. It fits action/component risk bias on a separate calibration
    bank, serializes Brier/AUC/ECE/reliability buckets by action and component,
    requires legal positive-advantage supported low-risk candidates, and reports
    rejection reasons. The default gate missed the current promotion-review
    control: hard guard/delegate/total fallback `0.1047`/`0.3712`/`0.4759`.
    The useful result is diagnostic leverage: calibration improved Brier/ECE,
    but `42,685` extracted targets were dominated by `eat` (`23,307`) and
    `drink` (`18,843`), with `4,645` rows rejected for high risk. The next
    actor slice needs contextual support or behavior-proximity constraints so
    Q/V advantage cannot collapse movement and stay behavior.
11. Contextual Behavior-Proximity Supported Extraction v2 is implemented and
   rejected as a promotion path. It adds a calibration-validation bank,
   context/action support with sparse-context-only fallback, target action
   expansion caps, family conversion caps, and movement/stay-to-resource
   conversion diagnostics. The cap-fix rerun enforces the final caps with no
   invariant violations, reducing target/logged TVD to `0.1800` and guarded
   total fallback to `0.4717`, but hard guard remains `0.1167`, well above the
   current promotion-review control `0.0568`. Autonomous-online also regresses
   to `-37.0` alive and `-31.5` births. The next actor slice should train
   behavior-proximity directly into the actor objective, likely through
   contextual KL/behavior-prior regularization and explicit stay/movement
   preservation, instead of relying on post-hoc target rejection.
12. Contextual behavior-prior actor regularization is implemented and rejected
   as a promotion path in both tested arms. The prior-only arm is the better
   result: it uses the calibration-bank contextual action distribution as a
   soft actor target, without contextual supported target extraction, and drops
   hard guard from the v2 cap-fix `0.1167` to `0.0663` while improving total
   fallback to `0.4700`. It still misses the current promotion-review hard
   guard `0.0568`, and autonomous-online remains rejected at `-35.0` alive and
   `-28.5` births. The combined prior-plus-v2-target arm is worse than
   prior-only (`0.1031` hard guard, `0.4727` total fallback), so do not stack it
   by default. Because v2 targets are precomputed before actor finetune, its
   target/logged TVD and movement/stay-to-resource conversion stay unchanged at
   `0.1800`; the actor prior changes the trained actor, not the post-hoc target
   selection diagnostics.
13. Actor action-distribution regularization is implemented and rejected as a
   promotion path in both tested forms. The root audit confirmed the user's
   concern: recent actor/critic losses were only moving a narrow guarded
   fallback frontier, while autonomous survival remained a large failure. The
   current promotion-review control has guarded hard/delegate/total fallback
   `0.0568` / `0.4172` / `0.4740`, but autonomous and autonomous-online still
   fail at `-34.25` / `-28.25` and `-31.25` / `-25.5` alive/birth deltas. The
   learned unguarded policy over-selects `eat` versus the heuristic rollout and
   under-selects movement/resource-balancing actions. A soft logged-action
   marginal KL objective corrected average probability mass but left top-1
   artifact diagnostics unchanged (`0.1595` train prediction/logged TVD) and
   failed the gate at `0.0591` / `0.4156` / `0.4747`; autonomous-online was
   only marginally better at `-31.0` / `-24.75`. A sharpened argmax-proxy KL
   objective was worse, pushing mass into one movement direction and failing at
   `0.0590` / `0.4251` / `0.4841` with autonomous-online `-33.0` / `-26.25`.
   Keep the hook opt-in and diagnostic-only. Do not continue global action-mix
   regularization or critic suppression penalties as the main path; the next
   larger slice needs closed-loop survival credit assignment or context-local
   sequence/navigation supervision that can preserve movement, drinking, and
   reproduction dynamics under autonomous rollout.
14. Mind v3 autonomous evolution is now the primary direction. It is not another
   guarded v2 artifact arm: it removes heuristic action selection from the
   agent runtime, stores bounded inherited controller state on agents, mutates
   controller state through reproduction, applies bounded reward-modulated
   controller updates with a short eligibility trace, and evaluates
   survival/reproduction directly. V2 remains the comparison baseline;
   fallback-rate micro-metrics are no longer the main success metric for v3.
   The first two-seed 120-tick v3 smoke reached zero heuristic action sources
   but collapsed (`0.0` alive mean, `0.0` births mean), versus heuristic
   baseline `49.5` alive mean and `44.5` births mean. Treat that as the first
   honest autonomous baseline and a blocker for the next population-search
   slice, not a reason to reintroduce heuristic fallback. The first
   searched-template slice is now executable through `sim:mind:v3:evolve` and
   reusable through evaluator/trajectory/headless `--founder-template` paths.
   On seeds `5,13` at `80` ticks, raw v3 reached `0.5` alive mean and `0.0`
   births mean; the searched template reached `4.0` alive mean and `0.5` births
   mean, with zero heuristic action-source counts in both cases. This is a
   measured improvement over raw v3, but it is still far below the same
   heuristic baseline (`28.0` alive mean, `18.0` births mean), so it is not a
   promotion claim. The next quality-diversity search slice adds resumable
   checkpoints, archive elites, holdout seeds, richer outcome metrics, and a
   survival-dominant score. It falsified the `120`-tick path at the current
   search scale: searched v3 remains `0.6` alive mean and `0.2` births mean
   across seeds `5,13,19,29,37`, versus heuristic `45.6` / `39.0`. The same
   search at an `80`-tick curriculum horizon is the first useful v3 improvement
   across five seeds: raw v3 is `0.8` alive mean and `0.0` births mean; searched
   v3 is `3.8` alive mean and `0.2` births mean, with zero heuristic action
   sources. This is still not a working mind. Treat it as evidence that
   curriculum search is the next milestone, while direct `120`-tick search with
   the current controller remains too weak.
15. Holdout-gated v3 curriculum search is implemented and falsifies the current
   controller/search scale at horizon extension. The run
   `output/mind/mind-v3-curriculum-80-120-search.json` used train seeds
   `5,13,19`, holdout seeds `29,37`, tick schedule `80,120`, population `12`,
   generations `3`, and search seed `733`. The `80`-tick stage passed the
   holdout floor with alive mean `2.5`, births mean `0.0`, and zero heuristic
   action sources. The `120`-tick stage failed the alive floor with holdout
   alive mean `0.5`, births mean `0.0`, and zero heuristic action sources. The
   final report selected the last passing `80`-tick candidate. Five-seed eval
   of that selected template reached `3.4` alive mean and `0.2` births at
   `80` ticks versus heuristic `27.8` / `16.0`, and `0.4` alive mean and
   `0.2` births at `120` ticks versus heuristic `45.6` / `39.0`. This is not
   better than the previous best `80`-tick searched v3 result (`3.8` / `0.2`);
   treat it as a measured blocker. The next v3 slice should increase
   closed-loop credit assignment or controller capacity before spending more on
   larger curricula.
16. V3 search now preserves runtime lineage and ecological diversity evidence.
   Candidate reports include `controller_lineage_elites` and
   `behavior_descriptors`; promoted candidate metadata comes from the strongest
   runtime lineage elite, while `founder_template_metadata` preserves the
   pre-rollout template. Next-generation mutation draws from lineage elites
   rather than only the balanced parent template, so terrain, trophic,
   movement, and reproduction variants can survive as search parents. This is
   a search-contract fix, not a promotion claim: the immediate goal is diverse
   durable lineages, not one hyper-optimized species.
17. The lineage-selection curriculum rerun falsified the naive next-step claim.
   On the same `80,120` command, the `80`-tick training best improved to score
   `18.5695`, alive mean `4.0`, births mean `0.6667`, movement rate `0.0506`,
   and zero heuristic actions, but holdout failed immediately: alive mean
   `0.5`, births mean `0.0`, movement rate `0.0`, resource rate `1.0`, and
   one unique requested action on average. This is overfit local resource-loop
   behavior, not durable ecology. The search archive now records bounded
   `behavior_niches` keyed by dominant terrain, trophic role, and meat mode,
   and next-generation parent selection can draw from those niche-specific
   lineage templates instead of only metric slots. With niche parents active,
   the same curriculum passes the `80`-tick holdout again (`3.5` alive,
   `0.0` births) but still fails the `120`-tick stage (`0.0` alive,
   `0.0` births). Five-seed eval of the selected niche-archive template is
   `3.6` / `0.4` at `80` ticks and `0.2` / `0.4` at `120` ticks, still far
   below heuristic `27.8` / `16.0` and `45.6` / `39.0`.
   The curriculum gate now also records and can enforce holdout movement rate,
   unique requested actions, and behavior niche count. With explicit floors
   `0.005` movement, `2` unique actions, and `4` behavior niches, the `80`-tick
   stage passes (`3.5` alive, `0.0111` movement, `5.0` actions, `11` niches)
   and the `120`-tick stage still fails (`0.0` alive, `0.0` movement, `1.0`
   actions, `9` niches). This prevents single-action local resource loops from
   being logged as ecological progress.
18. V3 report/gate diagnostics now expose action collapse directly. Aggregates
   include requested/resolved action histograms, dominant requested-action name,
   count, and share, plus active behavior niche count/keys. Active niches require
   live, moving, or reproducing lineage evidence, so static
   terrain/trophic/meat labels cannot hide a one-action holdout loop. Curriculum
   stages can now reject dominant-action collapse with
   `--curriculum-max-holdout-dominant-action-share` and require active niche
   evidence with `--curriculum-min-holdout-active-behavior-niches`.
19. V3 scoring now uses `survival_dominant_quality_diversity_v4`, adding a
   bounded active-niche bonus and bounded dominant-action collapse penalty.
   Strict rerun:
   `output/mind/audit/mind-v3-diversity-scored-strict-curriculum-80-120-search.json`.
   The `80`-tick holdout still passes (`3.5` alive, `0.0111` movement,
   `5.0` unique actions, dominant action share `0.5689`, `8` active niches),
   but the `120`-tick holdout still fails (`0.0` alive, `0.0` movement,
   `1.0` unique action, dominant action share `1.0`, `0` active niches).
   Five-seed eval of the selected template is unchanged from the niche-archive
   level: `3.6` / `0.4` at `80` ticks and `0.2` / `0.4` at `120` ticks.
   This falsifies selection-pressure-only progress and keeps the bottleneck on
   controller capacity/credit assignment.
20. V3 controller capacity has moved from fixed pseudo-random projection to
   `homeostatic_feature_projection_linear_action_head_v2` for new founders.
   The inherited action head remains bounded and heuristic-free, but the eight
   hidden features are now explicit policy-visible survival/reproduction
   signals: energy need, hydration need, injury/risk pressure, reproduction
   drive, local resource context, water context, movement pressure, and
   trophic/mode/signal context. New founders also carry
   `diverse_homeostatic_action_prior_v1` and a bounded specialization profile
   (`forager`, `hydration_seeker`, `disperser`, `reproducer`, or
   `predator_scavenger`). This is the first controller-capacity slice that moves
   the `120`-tick bottleneck: raw v2 founders reached `1.4` alive / `0.2`
   births at `80` ticks and still collapsed at `120`; the curriculum run
   `output/mind/audit/mind-v3-homeostatic-curriculum-80-120-search.json`
   completed both `80` and `120` stages. Stage `120` holdout reached `1.5`
   alive, `0.5` births, movement rate `0.0131`, `4.0` unique requested actions,
   dominant action share `0.5238`, and `7` active behavior niches. Five-seed
   selected-template eval reached `6.0` / `0.4` at `80` ticks and `1.0` /
   `0.4` at `120` ticks, versus heuristic `27.8` / `16.0` and `45.6` / `39.0`.
   This is real progress, not promotion quality.
   A follow-up visibility audit found and removed one small non-ecological
   dependency: the homeostatic projection no longer reads
   `mind_inheritance_available`, and scorer regression tests require invariance
   under patch/navigation-tail changes and under toggling that controller-state
   availability bit. The visibility-hardened selected-template rerun reached
   `6.2` / `0.4` at `80` ticks and `0.8` / `0.4` at `120` ticks, with zero
   heuristic action-source counts, so the controller remains useful but far
   below promotion quality.
   A second audit fixed the v3 transition-credit boundary: reward updates now
   credit a policy-valid `requested_action` when live resolution races convert
   it to `stay`, and fall back to `resolved_action` only for truly invalid
   requests. The trace policy is
   `policy_valid_requested_action_short_eligibility_trace_v2`, later extended
   to `policy_valid_requested_action_horizon_eligibility_trace_v3`. The rerun
   artifacts
   `output/mind/audit/mind-v3-credit-hardened-selected-80tick-eval.json` and
   `output/mind/audit/mind-v3-credit-hardened-selected-120tick-eval.json`
   matched the visibility-hardened metrics (`6.2` / `0.4` and `0.8` / `0.4`)
   with zero heuristic action-source counts.
21. Infrastructure slice for the bottleneck split is implemented. V3 search has
   `--rollout-workers`, with `search.rollout_execution` recording the process
   rollout policy and requested/resolved generation/holdout worker counts.
   Torch training and gates have `--torch-device cpu|cuda|mps|auto`; generated
   torch artifacts record `torch_device_metadata` at the model and
   neural-network levels, including requested/resolved device plus CUDA/MPS
   availability, while serialized weights remain CPU JSON. Local venv smoke on
   May 9, 2026: `auto` resolved to `mps`, CUDA was unavailable. This prepares
   the RTX host path for IQL sweeps without pretending GPU fixes v3 simulator
   rollout cost.
22. Gate profiling and process fan-out are now available for the bottleneck
   split exposed by the local IQL reruns. `mind_gate` accepts
   `--evaluation-workers` and records `protocol.evaluation_execution` with the
   requested/effective process worker count and matrix entry count. It also
   accepts `--artifact-diagnostics-workers`, records
   `protocol.artifact_diagnostics_execution`, and reports
   `timings.phase_wall_seconds` for trajectory preparation, dataset loading,
   training, artifact writing, artifact diagnostics preparation, artifact
   diagnostics, and evaluation. A tiny two-worker smoke wrote
   `output/mind/audit/mind-gate-diagnostics-workers-smoke-report.json` with
   both process pools active and phase timings present; the direct unit check
   confirmed parallel artifact diagnostics match serial diagnostics. The first real
   anchored-calibrated IQL rerun with `--artifact-diagnostics-workers 2
   --evaluation-workers 2` took `812.7058s`; phase timings were
   `37.4958s` training, `660.8851s` artifact diagnostics, and `111.5268s`
   evaluation. The follow-up sharded reducer splits diagnostics by trajectory
   dataset and finalizes raw stats once; the real 8-worker rerun took
   `301.8127s`, with `35.0668s` training, `151.8482s` artifact diagnostics,
   and `112.1603s` evaluation. This proves local gate throughput was dominated
   by Python diagnostics/rollout work, not CUDA tensor kernels.
23. Re-run default and extended strict gates only after total fallback is below
   `0.4780`, hard guard is below `0.1190`, and every validation seed has zero
   alive/birth regression.
24. Continue offline-to-online replay mixing only through `mind_gate
    --extra-trajectory` so every learned-rollout probe appends a ledger entry.
    Do not promote the replay-mixed artifact until it beats the default and
    extended strict matrices, not only the two-seed probe.
25. May 10 audit pass found and fixed a summary diagnostic timing bug:
    animal-resource reachability used the observation-time resource snapshot,
    but opportunity recording recomputed resource presence after all actions.
    Same-tick fresh-kill/carcass deposits could therefore be counted as
    unconsumed policy opportunities that no agent could have observed. The tick
    runtime now passes the same decision-time resource-presence snapshot into
    opportunity recording, with regressions covering both the direct recorder
    and the tick call path. Golden verification then exposed a second hidden
    accounting issue: generated-and-consumed-in-the-same-tick resources were no
    longer counted in `*_consumed_ticks`. Consumption tick counters now derive
    directly from consumption events, while `*_present_ticks` remains a
    decision-time opportunity signal.
26. The same pass verified a real v3 capacity issue: the v2 homeostatic
    controller was invariant to local patch and navigation inputs. New founders
    now use `local_navigation_feature_projection_linear_action_head_v3`, a
    sixteen-feature, observation-only projection over self state, center local
    patch resources/hazard, and bounded water/plant/carrion/prey navigation.
    Legacy fixed-random and homeostatic metadata remain loadable. Smoke results
    are not promotion evidence: raw five-seed founder eval reached `2.2` /
    `0.2` at `80` ticks and `0.0` / `0.2` at `120`, while a small `80`-tick
    search found `3.5` / `1.5` in-sample but collapsed to `eat` on holdout
    (`0.9958` dominant action share, `0.0` births). The next bottleneck is
    objective/credit assignment pressure, not representation visibility.
27. A follow-on v3 audit found that evolved search reports still collapsed
    founder initialization through a single selected controller profile. The
    runtime now accepts founder template pools, search founders are stratified
    across the built-in specialization priors, and promoted search artifacts
    export an `archive_diverse_founder_template_pool_v1` pool filtered by a
    minimal survival floor or births. The current smoke artifact
    `output/mind/mind-v3-eligible-diverse-pool-80-smoke.json` exports
    `hydration_seeker: 6`, `disperser: 1`, and `reproducer: 1`; its holdout
    reached `4.0` alive, `0.0` births, dominant `drink` share `0.5712`, and
    active behavior niches `8`. Five-seed eval reached `3.6` alive and `0.0`
    births at `80` ticks, and `0.6` alive and `0.0` births at `120` ticks.
    This reduces action collapse versus the old single-profile artifact
    (`0.789` dominant `drink` share at `80`) but does not solve reproduction or
    horizon transfer.
28. The next audit found that the v3 score could create fake progress by
    preferring in-sample terminal reproduction viability while the curriculum
    gate did not require any held-out terminal viability. The score policy is
    now `reproductive_viability_quality_diversity_v6`, with stronger terminal
    energy/health/matched-diet viability pressure, an explicit alive
    reproduction-dead-end penalty, and penalties for seed-brittle births or
    one-seed terminal viability. The curriculum gate now records and can enforce
    held-out energy viability, health viability, matched-diet viability, and
    biologically reproduction-ready floors. A dense v3 reward-shaping probe
    (`output/mind/mind-v3-repro-signal-v2-80-smoke.json`) produced one-seed
    in-sample births, then failed holdout with terminal energy/health/diet
    viability all `0.0`; that runtime reward change was not retained. The
    retained v6 smoke
    `output/mind/mind-v3-robust-repro-score-v6-80-smoke.json` selected the
    robustly less-bad `g0-c2` candidate, reduced held-out dominant action share
    to `0.3636`, but still had held-out terminal energy viability `0.0` and
    five-seed eval only `2.4` alive / `0.0` births at `80` ticks and `0.2` /
    `0.0` at `120`. Treat this as stricter falsification and gate hardening,
    not a controller promotion.
29. The evaluator now connects v3 diagnostics to real terminal reproduction
    blockers and controlled ecology fixtures. `mind_v3_evaluate` reports
    `terminal_reproduction_failure_attribution_v1` for default and fixture
    runs, including ready/biologically-ready agents, biological blockers,
    terminal viability shares, and blocker/readiness breakdowns by trophic role
    and meat mode. The opt-in `--fixture-suite basic` runs plant-only,
    carrion-only, prey-rich, and mixed-stable worlds through real
    `SimulationWorld` policy/action/diet/combat/reproduction machinery rather
    than a toy fixture harness. Smoke artifact
    `output/mind/mind-v3-controlled-fixture-v1-80-smoke.json` is another
    falsification: with the current v6 template and seed `5`, v3 is
    heuristic-free but loses open-world `80`-tick survival/births (`3` / `0`
    versus heuristic `37` / `19`) and loses births on every fixture. The fixture
    attribution identifies terminal energy/hydration/matched-diet/health
    readiness collapse, so the next controller work should target credit
    assignment and reproduction-readiness behavior, not more artifact plumbing.
30. The controlled fixture suite is now a report-level gate for both
    `mind_v3_evaluate` and `mind_v3_evolve`, and a curriculum stage stop when
    enabled during tick-curriculum search. The gate records explicit blockers
    for fixture alive/birth floors, mixed-stable births, terminal
    energy/hydration/health/matched-diet viability, and biologically-ready
    terminal agents. The v3 reward update signal changed to
    `outcome_delta_action_conditioned_readiness_signal_v3`, which credits
    observed post-action deltas toward reproduction-ready
    energy/hydration/health and terminal reproduction/death outcomes without
    expanding decision-time FOV. It also penalizes repeated no-gain `eat` and
    only grants extra action-conditioned credit to `eat`, `drink`, or movement
    when the finalized transition improves readiness-relevant state.
    New artifact
    `output/mind/mind-v3-repro-delta-fixture-gated-80-search.json` is the
    first material v3 improvement in this branch: selected candidate `g1-c7`
    reached `11.0` alive / `3.3333` births in train search and `8.5` alive /
    `1.0` births on holdout. Five-seed eval improved to `9.0` alive / `1.4`
    births at `80` and `4.4` alive / `1.8` births at `120`, still with zero
    heuristic actions. It is not promotion: fixture gate failed on
    `carrion_only` alive floor, terminal biologically-ready agents remain
    `0.0`, and open-world dominant requested action is still `eat` around
    `0.64`.
31. Fixture-aware top-K reranking is now wired into v3 search with
    `--fixture-rerank-top-k`. Search score nominates candidates, but final
    selection prefers fixture gate pass, fewer fixture blockers, fixture
    births/survival, holdout births/survival, and lower holdout action
    collapse. Artifact
    `output/mind/mind-v3-fixture-rerank-action-credit-80-search.json`
    selected `g2-c9`: `12.6667` train alive / `4.0` train births and `8.5`
    holdout alive / `3.0` holdout births. The reranker explicitly rejected
    raw top candidate `g2-c2` because it still had a fixture blocker. Five-seed
    eval improved to `9.8` alive / `3.4` births at `80` and `4.8` alive /
    `4.2` births at `120`, zero heuristic actions. Two-seed fixture eval
    passed the configured gate. Not promotion yet: heuristic remains far ahead,
    terminal biologically-ready agents remain effectively absent, and
    mixed-stable energy/health viability is still weak.
32. The v3 search score and reranker now account for a buried blocker that
    previous tests missed: terminal reproduction viability omitted hydration
    even though hydration is a real biological blocker. The score policy is
    now `hydration_sustained_core_readiness_quality_diversity_v8`, and reports
    include sustained core readiness from observed post-action
    energy/hydration/health ratios plus exact reproduction-ready agent/pair
    ticks. Top-K fixture rerank also has a bounded composite repair path:
    failed high-holdout nominees may be evaluated with a real archived
    gate-passing donor founder-template pool. A report-preservation bug was
    found and fixed in the same slice: repaired candidates were initially
    evaluated with a composite pool but saved with the normal archive pool,
    making downstream evals judge the wrong controller. Corrected artifact
    `output/mind/mind-v3-composite-repair-v8-top8-fixture2-80-search.json`
    selected `g2-c2+repair-g0-c4`, with `16.0` train alive / `5.3333` train
    births, `9.5` holdout alive / `3.0` holdout births, and a passing
    two-seed fixture gate. Five-seed eval is mixed: `80` ticks regressed to
    `9.0` alive / `2.2` births, while `120` ticks improved to `7.6` alive /
    `5.0` births. Still not promotion: terminal biologically-ready agents are
    `0.0`, and energy viability remains the primary bottleneck.
33. A deeper energy-readiness audit found another proxy mismatch. V3 search was
    scoring a simple energy-ratio readiness proxy while the real reproduction
    gate uses `reproduction_energy_requirement()`, including trophic breadth
    and animal-mode diet multipliers. Search now records continuous terminal
    energy-requirement satisfaction and energy-gap totals from
    `reproduction_end.energy_readiness_by_meat_mode`. `mind_v3_evaluate` now
    exposes the same fields at aggregate level under
    `terminal_reproduction_failure_attribution_v2`, so five-seed eval can
    independently report the metric being optimized instead of hiding it inside
    nested attribution.
34. The retained controller change is
    `real_energy_proxy_action_conditioned_readiness_signal_v4`: a higher
    observable energy target, stronger energy-progress credit, reduced free
    drink credit, and a no-gain drink penalty. It does not expand decision-time
    FOV; it credits only finalized before/after self-state and action outcome
    data. Artifact
    `output/mind/mind-v3-real-energy-credit-v10-top8-fixture2-80-search.json`
    selected `g0-c1`, passed the two-seed fixture gate, and improved holdout to
    `11.0` alive / `4.5` births. Five-seed eval improved from the v9 `9.0` /
    `2.2` at `80` and `7.6` / `5.0` at `120` to `10.8` / `3.8` and `8.8` /
    `6.8`. Energy satisfaction and energy viability improved, but hydration
    viability regressed and terminal biologically-ready agents remain `0.0`.
    Treat this as real bottleneck movement, not promotion.
35. Balanced bottleneck credit and temporal blocker attribution exposed the next
    failure. Search/eval now report terminal balanced reproduction readiness,
    energy/hydration balance, energy/hydration gap, and temporal core-readiness
    blockers. v11 passed the two-seed fixture gate but five-seed eval still
    collapsed toward `eat` at about `0.76` dominant requested-action share.
    v12 changed the reward to
    `balanced_bottleneck_hydration_guard_readiness_signal_v6`, which credits
    the limiting observed self-state delta, downweights non-limiting positive
    deltas, and penalizes hydration-limiting no-gain `eat` without adding
    decision-time FOV. Artifact
    `output/mind/mind-v3-balanced-bottleneck-v12-top8-fixture2-80-search.json`
    improved five-seed `120` eval versus v10 from `8.8` alive / `6.8` births
    to `14.0` / `8.6`, but hydration regressed and the fixture gate failed on
    `carrion_only`. A larger top-16 rerank still found no fixture-passing
    candidate. The actionable blocker is now carrion/scavenger lane
    preservation, not another generic larger sweep.
36. The v13 scavenger-lane slice made that preservation explicit. Mind v3 now
    has a separate `scavenger` founder profile, archive policy
    `quality_diversity_archive_v2` with a named `scavenger_lane` elite, fixture
    rerank policy `fixture_holdout_top_k_scavenger_lane_rerank_v2`, and
    composite repair policy `fixture_blocker_composite_founder_pool_repair_v2`.
    Lane credit is derived from profiles, behavior descriptors, and lineage
    elites, not from self-asserted tags. Artifact
    `output/mind/mind-v3-scavenger-lane-v13-top8-fixture2-80-search.json`
    selected `g0-c7+repair-g0-c4` using a scavenger-lane donor. It still fails
    the fixture gate, but the `carrion_only` blockers narrowed to energy
    viability only: `1.0` alive, `2.5` births, hydration viability `0.5`,
    matched-diet viability `0.5`, and `14` animal-resource consumption events
    across the two fixture seeds. Five-seed eval improved to `16.6` alive /
    `6.2` births at `80` and `16.0` / `11.8` at `120`; terminal
    biologically-ready agents reached `1.0` mean at `120`. Remaining blockers:
    fixture energy viability, eat dominance around `0.66`, and hydration as the
    primary temporal blocker at `120`.
37. The v14 carrion-rerank slice made fixture rerank candidate evaluation
    process-parallel and changed reward/selection pressure from generic `eat`
    to observed outcome credit. The reward signal is now
    `balanced_bottleneck_observed_carrion_readiness_signal_v7`: carcass and
    fresh-kill `eat` get extra credit only when the action actually consumed an
    animal resource and improved observed self-state readiness; zero-delta
    `eat` is treated as no-gain. Search scoring now records observed
    animal-resource/carcass event rates, and fixture summaries expose
    carcass/fresh-kill consumption plus terminal energy satisfaction. Artifact
    `output/mind/mind-v3-carrion-rerank-v14-top8-fixture2-80-search.json`
    selected `g1-c0` and passed the `80`-tick two-seed fixture gate:
    `carrion_only` improved to `1.5` alive, `4.0` births, energy viability
    `0.5`, energy satisfaction `0.6471`, hydration viability `0.75`,
    matched-diet viability `0.75`, and `11.0` observed animal-resource
    consumption events per fixture run. This is not a global promotion:
    five-seed open-world eval regressed versus v13 to `14.0` alive / `6.0`
    births at `80` and `14.6` / `11.4` at `120`, while dominant `eat` improved
    to about `0.51`. The `120`-tick fixture gate still fails on `carrion_only`
    energy/hydration/matched-diet viability despite `5.0` births, so the next
    blocker is carrion horizon robustness, not hidden FOV or npm/test plumbing.
38. The v16 multi-horizon rerank slice makes that horizon blocker explicit
    instead of relying on a separate post-hoc 120 fixture run. New CLI option
    `--fixture-rerank-ticks` evaluates top-K fixture rerank candidates across
    multiple controlled fixture horizons, combines blockers across horizons,
    and records per-horizon fixture summaries. The retained rerank policy is
    `fixture_holdout_top_k_multi_horizon_scavenger_lane_rerank_v5`: if no
    candidate passes all horizons, partial horizon pass coverage is preferred
    before raw blocker count, so an 80-tick fixture pass is not silently
    regressed while chasing a narrower 120 blocker list. This was not just a
    paper check: v15 briefly selected `g2-c3+repair-g0-c4`, improving broad
    five-seed eval to `14.6` / `6.2` at `80` and `17.4` / `14.0` at `120`, but
    it failed the 80-tick carrion energy fixture and was rejected as a
    promotion-path regression. Corrected v16 artifact
    `output/mind/mind-v3-multihorizon-rerank-v16-top8-80-120-search.json`
    selects `g1-c0`, preserves the 80 fixture pass coverage, and honestly
    fails the combined 80/120 fixture gate on 120-tick `carrion_only`
    energy/hydration/matched-diet viability. Five-seed eval remains `14.0` /
    `6.0` at `80` and `14.6` / `11.4` at `120`. The useful signal is the
    rejected challenger: composite repair can improve open-world 120 survival
    and births, but the controller still cannot satisfy the controlled carrion
    readiness gate across horizons.
39. The v18 bridge-repair slice tested whether that rejected v15 signal could
    be safely reused without lowering the gate. The repair policy is now
    `fixture_blocker_composite_founder_pool_repair_v5`: after standard repair,
    the reranker creates promotion-safe bridge candidates using an
    80-horizon-passing primary and the best unsafe long-horizon donor. v17
    showed that injecting the full donor pool destroys carrion readiness. v18
    then tried bounded donor-template injections with limits `1`, `2`, and
    `4`. Artifact
    `output/mind/mind-v3-bridge-lite-rerank-v18-top8-80-120-search.json`
    evaluated four bridge candidates, but every bridge candidate failed the
    combined fixture gate; even the one-template bridge had `0.0` carrion-only
    survivors at the limiting horizon. The selected candidate stayed `g1-c0`,
    and exact v18 five-seed `120` eval remained `14.6` alive / `11.4` births.
    This rules out simple founder-pool mixing as the path to global promotion;
    the next real controller change must create or learn carrion-ready
    long-horizon behavior directly while keeping the 80 fixture pass as a hard
    precondition.
40. The v19 contextual-founder audit found that template-pool assignment itself
    was a hidden confounder. Previously founders used `agent_id % pool_size`,
    so adding a donor template could reassign unrelated species to different
    controllers and make bridge repair look worse or better for the wrong
    reason. Runtime founder assignment is now
    `contextual_trophic_founder_template_assignment_v1`, selecting templates by
    policy-visible self context: trophic role and meat mode. This is not extra
    FOV and not fixture knowledge. Re-evaluating the prior v18 artifact under
    the new assignment improved broad five-seed `120` eval to `18.0` alive /
    `12.6` births, but the `120` fixture gate still failed on `carrion_only`
    energy and matched-diet viability. A fresh v19 search artifact,
    `output/mind/mind-v3-contextual-template-v19-top8-80-120-search.json`,
    records the corrected score policy
    `contextual_template_scavenger_lane_balanced_bottleneck_quality_diversity_v19`
    and assignment policy. It selected `g1-c0` with `21.6667` search alive /
    `9.3333` search births, but five-seed eval regressed to `14.4` / `4.0` at
    `80` and `11.4` / `7.8` at `120`, and no top-K candidate passed the first
    carrion horizon. Conclusion: contextual assignment is a real bug fix, but
    blind scalar search can still abandon the vetted carrion lane. The next
    slice should add warm-started candidate injection or a fixture archive so
    prior gate-safe behavior remains in the search population.
41. The v20/v21 warm-start slice preserved prior archive behavior in the search
    population and found another provenance bug. The first implementation of
    `--warm-start-report` filled the candidate limit from the first source
    report only, so the intended v18/v14 blend was not actually tested.
    Warm-start import is now round-robin across reports, reports only record
    warm-start provenance when reports are supplied, and fixture rerank forces
    warm-start nominees into the hard regression surface. Round-robin v20
    improved broad eval versus v19 to `15.4` alive / `4.4` births at `80` and
    `11.0` / `8.0` at `120`, but still failed carrion fixtures. The real
    controller improvement is v21:
    `policy_valid_requested_action_horizon_eligibility_trace_v3` increases
    delayed action credit to length `12` with decay `0.84`, so drink/resource
    outcomes can credit longer movement paths without adding hidden FOV.
    Artifact
    `output/mind/mind-v3-delayed-credit-v21-top8-80-120-search.json`
    selected `g2-c2`, reached `21.3333` search alive / `10.3333` search births,
    and five-seed `120` eval improved to `15.8` alive / `12.2` births with
    zero heuristic actions and dominant `eat` near `0.44`. It is still not a
    promotion: controlled `carrion_only` hydration remains `0.0` at `80`, and
    the `120` carrion horizon collapses. The attempted v22 hydration-risk eat
    penalty was falsified and reverted from the default path; artifact
    `output/mind/mind-v3-hydration-risk-v22-top8-80-120-search.json` regressed
    broad search and carrion viability. Next work should put fixture outcomes
    into the search objective/parent selection loop or add a controller
    mechanism that learns water-return behavior after carrion intake.
42. A May 11 audit found a contract and capacity mismatch in Mind v3. The
    implementation consumed local patch and bounded navigation fields, but the
    contract still claimed `policy_visible_self_state_only`. The contract now
    declares `policy_visible_self_local_patch_navigation`, keeps
    `mind_inheritance_available` excluded, and lists the derived interaction
    features. New founders use
    `need_gated_local_navigation_feature_projection_linear_action_head_v4`, a
    twenty-four-feature deterministic projection that adds thirst-gated water
    direction, plant-hunger direction, meat-hunger carrion direction, and
    predatory-hunger prey direction from already policy-visible observation
    fields. This is not a FOV expansion. Search reports also moved founder pool
    metadata to
    `archive_diverse_founder_template_pool_with_identity_diagnostics_v2` and
    record template-pool fingerprints, because the v21 fixture artifact showed
    many top-K candidates evaluating through effectively identical composite
    pools. Artifact
    `output/mind/mind-v3-need-gated-v23-top8-80-120-search.json` is not a
    promotion, but it is not paper progress either: old v3 warm-start
    controllers still dominated scalar search, while a v4 candidate won final
    rerank by fewer fixture blockers. Five-seed eval improved births over v21
    to `7.0` at `80` and `15.2` at `120`, with `16.4` alive at `120`, but the
    combined carrion fixture still fails. Next work should put fixture outcomes
    into parent selection rather than running another scalar-only search.
43. The first v3 neural residual audit closed a different false-progress path.
    The frozen pure-Python neural artifact now runs behind the live online
    linear anchor instead of replacing it, and the residual policy is
    `linear_controller_margin_guarded_neural_residual_v2`: collapsed `eat`
    neural tops are shadowed, and nontrivial linear-anchor margins are not
    overridden. Evaluator reports include
    `mind_v3_neural_anchor_diagnostics_v1`, with neural/linear/anchored action
    counts, score margins, shadow reasons, and transition pairs. Targeted v26
    drilldown
    `output/mind/mind-v3-v26-neural-margin-guard-carrion-120.json` improved
    broad neural-anchor survival over the prior guarded residual from `16.0` to
    `19.0` alive at `120` while births stayed `16.0`; linear still led at
    `23.5` / `18.5`. The carrion-only fixture stayed blocked (`0.0` terminal
    alive, five blockers), so this is diagnostic guardrail progress, not a path
    to promotion. Stop spending iterations on residual-scale tuning unless a
    new learner/training signal reduces the carrion blocker set.
44. A follow-up contextual fixture-bias artifact tested the next obvious
    pure-Python training signal. Neural artifacts now include
    `contextual_fixture_floor_gap_action_bias_v2`, with
    `policy_visible_carrion_water_context_bias_v1` derived only from ecological
    policy input fields: energy/hydration/matched-diet need, center-patch
    carcass/fresh-kill, and visible water/carrion navigation vectors. Artifact
    `output/mind/mind-v3-v26-neural-context-bias-artifact-v3.json` and
    drilldown `output/mind/mind-v3-v26-neural-context-bias-carrion-120.json`
    reproduced the same broad result as the margin-guard slice (`19.0` alive /
    `16.0` births versus linear `23.5` / `18.5`) and left carrion-only
    unchanged (`0.0` alive, `2.5` births, five blockers, primary temporal
    blocker `energy`). The neural top-action distribution barely moved on
    carrion-only, so the current residual/anchor path is not exposing enough
    controllable leverage. Next work should collect controlled fixture
    trajectories for horizon labels or move to torch/vectorized rollout
    training; another hand-shaped residual bias is not justified.
45. The fixture-to-label bridge is now implemented. `mind_v3_evaluate` accepts
    `--trajectory-output-dir` and writes broad plus controlled-fixture runs as
    loadable trajectory JSONL.gz files while keeping summary-only evaluation.
    Reports carry each run's `trajectory_path`. Smoke report
    `output/mind/mind-v3-v27-fixture-trajectory-export-smoke.json` exported
    `output/trajectories/mind-v3-v27-fixture-export-smoke/fixture-carrion-only-mind-v3-29-40.jsonl.gz`,
    and `mind_horizon_labels` generated
    `output/mind/mind-v3-v27-carrion-fixture-horizon-labels-smoke.json` with
    `393` labels at horizons `10,20,40`. This makes the next step concrete:
    include controlled carrion trajectories in artifact training/evaluation,
    rather than asking aggregate blocker labels to stand in for sequence data.
46. The first controlled-trajectory neural training slice produced a partial
    result and a sharper blocker. First,
    `output/mind/mind-v3-v28-broad-carrion-neural-artifact.json` trained on
    broad v3 plus failing carrion-only v3 fixture trajectories. That raised
    label animal-resource contacts to `126` but did not beat the linear anchor:
    `19.0` alive / `9.0` births at `80` and `22.0` / `17.0` at `120`; the
    carrion fixture stayed at `0.0` alive / `2.5` births with five blockers.
    The trainer now supports explicit per-trajectory multipliers through
    `sim:mind:v3:train-neural --trajectory-weight`, recorded in the artifact
    as `trajectory_weight_multipliers` and separate horizon/trajectory/final
    sample-weight summaries. The source-balanced artifact
    `output/mind/mind-v3-v29-source-balanced-neural-artifact.json` used weights
    `1,1,4,4,1,1` for broad v3, heuristic carrion fixture, and failing v3
    carrion fixture trajectories. It improved broad reproduction to `20.0`
    alive / `11.5` births at `80` and `23.0` / `19.5` at `120`, versus linear
    `20.5` / `10.0` and `23.5` / `18.5`. Dominant action share stayed below
    `0.47`, but carrion-only still had `0.0` alive, `2.5` births, and five
    blockers. Diagnostics show the neural top shifted toward movement while
    the linear-margin guard still shadowed most fixture overrides. Next work
    should test one bounded anchor/learner leverage change against the same
    broad-plus-carrion matrix; if carrion terminal alive or blocker count does
    not move, switch to torch/IQL or vectorized rollout training rather than
    another scalar-weight pass.
47. That bounded anchor/learner leverage change was tested and falsified. A
    policy-visible carrion/water context gate that relaxed the residual scale
    and linear-margin guard was too broad: v30 broad eval regressed to `15.5`
    alive / `7.5` births at `80` and `11.0` / `10.0` at `120`, while
    carrion-only still failed with `0.0` alive and five blockers. Tightening the
    gate to direction-aligned carrion/water navigation made it rare in broad
    worlds (`1.21%` active at `80`, `0.86%` at `120`) and recovered broad eval
    to `20.0` / `11.5` at `80` and `22.5` / `19.0` at `120`, but carrion-only
    remained `0.0` alive / `2.5` births with alive, energy, hydration, health,
    and matched-diet blockers. The runtime code was restored to the existing
    `linear_controller_margin_guarded_neural_residual_v2` default. Treat this
    as stop evidence for residual leverage and scalar weighting loops; the next
    useful implementation path is stronger deterministic controller capacity
    against horizon/fixture labels, or torch/IQL/vectorized rollout training.
48. The stronger deterministic controller-capacity probe was implemented as
    v31 and rejected. `deterministic_horizon_fixture_policy_v2` is an opt-in
    direct artifact mode behind
    `sim:mind:v3:train-neural --artifact-mode horizon-fixture`; it uses
    ecological policy inputs, action-conditioned horizon utility, fixture
    pressure, and behavior-support normalization, and it bypasses the
    linear-margin anchor at runtime. The evaluator can now compare linear,
    anchored neural, and direct experimental artifacts in one report via
    `--anchored-neural-artifact`. The comparable v31 reports used the v26
    founder template and same v29 source-balanced trajectories. At `80`, direct
    v31 produced `1.5` alive / `0.0` births with dominant action share
    `0.5155`, versus linear `20.5` / `9.0` and anchored neural `21.0` /
    `10.5`; carrion-only had `0.5` alive / `5.5` births but still five
    blockers. At `120`, direct v31 collapsed to `0.0` alive / `0.0` births
    with dominant action share `0.5136`, while linear reached `23.0` / `17.5`
    and anchored neural `20.5` / `18.0`; carrion-only stayed at `0.0` alive
    with five blockers. Heuristic action sources were zero. Treat this as the
    pure-Python artifact ceiling for now: the next useful track is torch/IQL or
    vectorized rollout training with the same broad-plus-carrion acceptance
    gate.
49. The first carrion post-contact autopsy is complete. The new
    `sim:mind:v3:carrion-autopsy` command builds
    `mind_v3_carrion_failure_autopsy_v1` reports from trajectory JSONL.gz
    files without changing simulator, reward, fixture, policy, or trainer
    behavior. A fresh v31 trajectory export is
    `output/mind/mind-v3-v31-carrion-autopsy-eval-120.json`, with reports
    `output/mind/mind-v3-v31-carrion-autopsy-direct-120.json`,
    `output/mind/mind-v3-v31-carrion-autopsy-anchored-120.json`, and
    `output/mind/mind-v3-v31-carrion-autopsy-linear-120.json`. Direct v31 had
    seven post-contact carrion episodes and all died; its dominant terminal
    path was `low_gain_eat_energy_depletion_after_carrion_contact` (`3/7`),
    with zero drinks, seven post-contact reproduction events, `337`
    post-contact eats, and only `2.2928` total animal-resource gain. Anchored
    neural and linear both died by
    `movement_energy_depletion_after_carrion_contact` in every post-contact
    episode (`7/7` and `10/10`), also with zero drinks. This validates the
    audit direction but changes the task order: first build a counterfactual
    rollout labeler and prove a legal carrion-water recovery sequence can
    survive to 120 ticks; only then feed those labels into constrained
    torch/IQL. If no counterfactual sequence survives, audit fixture/world
    mechanics before doing learner work.
50. The counterfactual carrion-water feasibility slice is complete. The new
    `sim:mind:v3:carrion-counterfactual` command runs deterministic,
    policy-visible scripts against the controlled `carrion_only` fixture and
    writes `mind_v3_carrion_counterfactual_rollout_v1` reports with optional
    trajectory exports. The v32 report
    `output/mind/mind-v3-carrion-counterfactual-v32-120.json` used seeds
    `29,37` at `120` ticks. `hydration_safe_carrion_cycle` kept `3` agents
    alive on both seeds, for `3.0` alive / `11.0` births mean, zero heuristic
    action sources, and dominant action share `0.2528`; `water_first_recovery`
    and `conserve_after_carrion` also produced nonzero terminal alive on at
    least one run. Autopsy of the hydration-cycle trajectories found `22`
    post-contact episodes, `4` survived contact windows, `65` drinks, `87`
    animal-resource events, and `21.825` animal-resource gain, though the
    dominant remaining death path was still
    `movement_energy_depletion_after_carrion_contact`. This resolves the
    feasibility question: the fixture/world mechanics allow survival, and the
    next implementation slice should convert these positive sequences into
    constrained torch/IQL labels rather than continue residual-scale,
    anchor-margin, or scalar-weight tuning.
51. The counterfactual action/value label slice is complete. The new
    `sim:mind:v3:carrion-counterfactual-labels` command writes
    `mind_v3_carrion_counterfactual_labels_v1` reports from scripted
    counterfactual trajectories. The v33 label report
    `output/mind/mind-v3-carrion-counterfactual-v33-hydration-cycle-labels.json`
    was generated from the two successful `hydration_safe_carrion_cycle`
    trajectories with horizons `20,40,80,120` and primary horizon `120`. It
    contains `1155` labels, legal logged-action rate `1.0`, and rollout
    terminal targets preserving the actual end state: `46` terminal agent
    timelines, `6` terminal-alive agents, alive-agent rate `0.130435`, and
    `124` labels whose owning agent survives to terminal. The strict 120-tick
    horizon remains censored for final survivors because the last decision tick
    is `119`; the rollout-terminal target exists specifically to avoid losing
    those positive labels. Next work should wire this report into an opt-in
    torch/IQL data path for row weighting and terminal/homeostatic constraints.
52. The opt-in torch/IQL counterfactual-label data path is implemented.
    `sim:mind:train` and `sim:mind:gate` now accept
    `--torch-iql-counterfactual-labels` for `torch-discrete-iql`, plus a
    non-default row-weight scale override. The hook aligns labels by
    trajectory path and dataset-record index, then adds label-derived row
    weights, rollout-terminal logged-action value targets, state viability
    targets, and logged-action viability targets. Artifact training metrics now
    include matched label counts, action-support failures, terminal-alive label
    count, source scripts, action counts, animal-resource gain, and label
    digest. This is still not a result claim: no torch candidate has been
    trained or evaluated in this slice. The next run should train one bounded
    candidate on broad trajectories plus the v33 hydration-cycle labels and
    accept it only if carrion-only blockers move without broad 120 regression.
53. The labeled torch/IQL acceptance report surface is implemented.
    `sim:mind:v3:labeled-iql-slice` evaluates a candidate learned artifact
    beside the current Mind v3 linear default and optional anchored-neural
    baseline on the same broad seeds and controlled fixtures. The report gate
    enforces the current bounded milestone: broad `120` alive within `1.0` of
    linear, births not worse than linear, dominant action share `<= 0.50`, zero
    heuristic runtime actions, and carrion-only movement by either nonzero
    terminal alive agents or fewer fixture blockers than linear. This closes
    the previous tooling gap where torch/IQL could be trained and broad-evaled
    without producing the carrion-only acceptance comparison in the same
    artifact. No torch candidate result is claimed here; the next actual result
    is the RTX v34 train-plus-slice run.

### GPU / CUDA Boundary

Do not mix GPU plumbing with learner behavior changes. `mind_train` and
`mind_gate` now record torch device provenance, but CUDA is still appropriate
only for torch/IQL learner sweeps on the RTX host. The current Mind v3
curriculum search is dominated by Python simulator rollouts, so GPU will not
fix that runtime; use
`--rollout-workers` for v3 CPU rollout parallelism,
`--artifact-diagnostics-workers` for train/held-out artifact diagnostics, and
`--evaluation-workers` for independent `mind_gate` evaluation matrix entries.
Promote only artifacts that pass the deterministic CPU/Mac gate path, and keep
narrow learner probes attributable to model changes rather than backend
nondeterminism. Switch to the RTX host when running batches of torch/IQL
training sweeps, not for single v3 searches or one-off gate evaluations. Treat
MPS/`auto` reruns as smoke unless their metric drift is bounded against a
deterministic CPU reference.
