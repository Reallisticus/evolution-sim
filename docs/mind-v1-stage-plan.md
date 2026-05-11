# Mind v1 Stage Plan

## End Goal

Mind v1 is ready to move beyond the heuristic safety floor when an offline
policy can run disabled-by-default, preserve Foundation behavior under extended
held-out validation, and demonstrate measurable learned value without relying
on hidden test-specific thresholds.

The phase boundary is not "a model exists." The boundary is:

- trajectory, observation, action, reward, artifact, and gate contracts remain
  stable and versioned;
- `npm run sim:mind:gate` and `npm run sim:mind:gate:extended` pass with no
  blockers or warnings;
- artifact and runtime diagnostics expose imitation quality, action drift,
  contextual coverage, guard fallback share, and per-role/per-mode outcomes;
- Stage 2 guard-share criteria are enforced, including aggregate and per
  role/mode caps plus optional reduction checks against a prior artifact;
- guarded runtime experiments remain opt-in and cannot regress the heuristic
  baseline on held-out seeds;
- a stronger offline baseline reduces heuristic delegation and hard-guard
  fallback while keeping the extended gate green.

## Stages

### Stage 1: Foundation-Backed Offline Safety Floor

Status: complete.

The current guarded contextual behavior-cloning baseline is disabled by default
and sits behind the observation heuristic. It only overrides different heuristic
actions with decisive offline evidence, reports artifact/runtime diagnostics,
and passes the default and extended Mind gates.

Exit criteria:

- `npm run sim:mind:gate -- --reuse-trajectories --fail-on-blockers` passes.
- `npm run sim:mind:gate:extended` passes.
- Guard intervention share is reported for each validation horizon.
- No runtime learned-controller experiment is enabled by default.

### Stage 2: Stronger Offline Baseline

Status: checkpoint complete; keep this gate green while Stage 3 broadens the
matrix.

Improve the offline learner so it can safely reduce guard fallback share without
weakening gates. Candidate implementation work includes reward-weighted action
priors, calibrated per-context support/margin metadata, train/validation
diagnostic splits, and explicit per-role/per-mode performance comparisons.
The current checkpoint adds strict model-artifact validation, held-out artifact
diagnostics, guard-by-action and top guarded-context diagnostics, and
per-role/per-mode guard caps so learner changes can target measured fallback
clusters instead of optimizing against a narrow seed slice. Evaluation also
summarizes the learned actions suppressed by the guard through an in-memory
sidecar, keeping trajectory JSONL schema stable. The first calibrated baseline
slice adds artifact/runtime support and margin metadata so future learner
changes can be evaluated by evidence strength, not just by aggregate guard
fallback share. The paired report slice adds heuristic-vs-learned role/mode
comparisons, keeping Stage 2 model iteration grounded in outcome deltas rather
than aggregate policy averages.
The Stage 2 checkpoint now trains on the default multi-seed bank
`[1,2,3,4,6,7,8,9,10,11,12,17,23,31]` and validates against both the default
held-out seed bank `[5,13,19,29]` at 120 ticks and the extended seed bank
`[5,13,19,29,37,41]` at 120 and 180 ticks. Earlier safe-deviation bypasses for
local eating and plant movement passed the default slice but regressed the
extended matrix, so they are disabled in the current artifact. The active
abstention mechanism is an artifact-versioned confidence delegate: when a
learned action disagrees with the heuristic but the matched training action
prior has score margin below `0.221`, runtime delegates to the heuristic before
the hard safety guard and reports that delegation separately from guard
intervention.
The current observability checkpoint extends the same breakdowns to confidence
delegation: reports now include deferred learned actions, support and margin
buckets, score source, role/mode delegate rates, and top delegated contexts so
the next learner can target high-frequency abstention clusters directly.
Artifact diagnostics also expose action-confusion pairs and per-action
precision/recall, which keeps future learner changes honest about whether they
reduce the broad `eat` prior or merely shift mistakes between movement actions.

The latest calibrated baseline uses feature policy `mind_feature_policy_v2`,
conditional support floor `3`, `smoothed_contextual_action_prior_v1` with
smoothing `0.1`, no prior correction, and safe-deviation bypasses disabled.
It passed the default gate with held-out artifact top-1 accuracy `0.4976`,
held-out action-distribution drift `0.1599`, aggregate guard intervention
`0.1190`, delegate rate `0.3590`, max role guard `0.2166`, max mode guard
`0.1560`, and zero alive/birth deltas on every held-out seed. It passed the
extended gate with held-out top-1 accuracy `0.5002`, held-out drift `0.1683`,
aggregate guard intervention `0.1187` and `0.1228`, delegate rates `0.3562` and
`0.3181`, max role guard `0.2046` and `0.2241`, max mode guard `0.1553` and
`0.1951`, and zero alive/birth deltas at both 120 and 180 ticks. This is still
not a learned-value milestone; it is a stronger abstaining baseline and
diagnostic floor for the next learner.

The first opt-in reward-weighted trainer is implemented but not promoted. It
uses the same contextual-prior structure and weights records by
`max(0.05, 1.0 + reward.total)`, writing model type
`guarded_reward_weighted_contextual_prior_bc_v1` with explicit
`reward_total_shifted_clamp_v1` metadata. On the default reused-trajectory gate
it passed safety with zero alive/birth deltas, held-out artifact top-1 accuracy
`0.5006`, and held-out action-distribution drift `0.2149`. It reduced
confidence delegation from `0.3590` to `0.3242`, but hard guard intervention
rose from `0.1190` to `0.1563` and total heuristic fallback rose from `0.4780`
to `0.4805`. Treat this as an experiment harness, not learned value. The next
learner should use advantage/calibration-aware weighting or per-context action
selection that reduces total fallback, especially hard guards, rather than
merely shifting disagreements from delegate to guard.

The next strict promotion target is now executable through the Mind gate:
candidate models must keep zero per-seed alive and birth deltas, hard guard
below the current rounded control value `0.1190`, and total heuristic fallback
below the current rounded control value `0.4780`. Because gate caps are
inclusive, use `--max-guard-intervention-rate 0.1189` and
`--max-total-heuristic-fallback-rate 0.4779` for a strict local promotion run.
`npm run sim:mind:gate:strict` pins those caps to the current blended candidate
and the extended `5,13,19,29,37,41` seed matrix over `120,180` tick horizons.

The first opt-in advantage-calibrated trainer is implemented but not promoted.
It adjusts contextual action support by within-context reward advantage, writes
model type `guarded_advantage_calibrated_contextual_prior_bc_v1`, and records
`contextual_reward_advantage_adjusted_counts_v1` plus
`contextual_reward_advantage_lift_v1` metadata. The calibration slice added
per-context `delegate_score_margin` metadata so runtime confidence delegation
uses unadjusted support for the learned top action while action ranking can
still use reward-adjusted scores. On the default reused-trajectory strict target
it preserved zero alive/birth deltas, improved held-out artifact top-1 accuracy
to `0.5038`, and reduced hard guard below the strict control target at
`0.0929`. It is still not promoted because confidence delegation rose to
`0.3856` and total heuristic fallback remained `0.4785`, above the strict
`0.4779` cap.

Follow-up probes show the remaining blocker is actual action disagreement, not
confidence partitioning. Changing the delegate threshold only moved decisions
between hard guard and confidence delegation while total fallback stayed near
`0.4785`. Reward-scale sweeps from `0.75` through `4.0` did not beat the current
`2.0` scale. Action-family floors for movement/attack actions reduced the
candidate back near the control total but did not cross the strict fallback cap.
Local-eat safe-deviation probes reduced fallback but changed per-seed
alive/birth outcomes, so safe deviations remain disabled.

The next opt-in trainer is `advantage-blended-contextual-prior`. It blends the
uniform contextual prior with the advantage-calibrated prior at weight `0.2`,
writing model type `guarded_advantage_blended_contextual_prior_bc_v1`,
sample-weight policy `contextual_reward_advantage_blended_counts_v1`, and
`contextual_reward_advantage_score_blend_v1` metadata. This is the first
candidate to beat the strict local control target without relaxing runtime
guards: on the default reused-trajectory gate it passed with zero per-seed
alive/birth deltas, hard guard `0.1123`, confidence delegation `0.3650`, and
total heuristic fallback `0.4773`. On the extended `5,13,19,29,37,41` seed
matrix with `120,180` tick horizons and the same strict zero-delta/fallback caps,
it also passed: at 120 ticks hard guard was `0.1123`, delegation `0.3630`, total
fallback `0.4753`; at 180 ticks hard guard was `0.1163`, delegation `0.3243`,
total fallback `0.4406`.

The next learner step is now contract-visible through artifact diagnostics:
reports include reward calibration by label action, predicted action, and
predicted score-margin bucket. This is the audit surface for a future
calibration-aware value/action model: it lets the trainer compare score
confidence against realized reward while keeping the runtime policy boundary
unchanged and unprivileged.

The first opt-in value-calibrated trainer is implemented. It estimates mean
reward per action for global and contextual buckets, writes full action-value
maps into model artifacts, validates those maps against reward-total bounds,
and blends a small `0.05` value preference on top of the current
advantage-blended prior. The direct `0.2` value blend reduced hard guard but
missed strict promotion because confidence delegation rose too far; the safer
anchored value head passed the strict extended matrix with zero alive/birth
deltas. At 120 ticks hard guard was `0.0966`, delegation `0.3807`, total
fallback `0.4773`; at 180 ticks hard guard was `0.0989`, delegation `0.3488`,
total fallback `0.4477`. This is real progress toward a value learner because
hard guards drop materially, but it is not a heuristic replacement yet: total
fallback is still constrained by confidence delegation and is not better than
the current blended candidate on the extended matrix.

The confidence-calibration slice is implemented for one narrow behavior class:
`positive_value_safe_deviation_v1` lets high-support, positive-value local
`eat` actions override heuristic `stay` decisions when the matched value head
clears explicit artifact thresholds. The strict extended matrix still passes
with zero alive/birth deltas. At 120 ticks hard guard is `0.0964`, delegation
`0.3807`, safe value deviations `0.0002` (`3` actions), and total fallback
`0.4771`; at 180 ticks hard guard is `0.0989`, delegation `0.3488`, safe value
deviations `0.0001` (`3` actions), and total fallback `0.4477`. This is a real
but small controller step: total fallback is now below the strict `0.4779` cap
and the rounded `0.4780` control target, but the system is still mostly
abstaining through confidence delegation.

Rejected tuning remains important. Lowering the value-supported deviation
thresholds to support `16`, value margin `0.02`, and learned value `0.01`
reduced delegation more aggressively, but failed the 180-tick strict gate:
alive mean delta was `-0.8334`, seed `5` alive delta was `-3`, seed `13` alive
delta was `-4`, and seed `5` births delta was `-2`. Keep the accepted
thresholds at support `32`, value margin `0.04`, and learned value `0.02` until
a stronger learner can preserve per-seed outcomes while taking more actions.
Broadening the accepted value path from heuristic `stay` to heuristic movement
was also rejected: it cut total fallback to `0.3857` at 120 ticks and `0.3605`
at 180 ticks, but seed `5` regressed by `-15` and `-33` alive agents and by
`-16` and `-36` births. Immediate positive value is therefore not enough; the
next learner has to model resource depletion and population-level effects.

The next learner should target a richer value/calibration boundary rather than
loosening thresholds: estimate action value under action-conditioned context,
separate resource-consumption externalities from immediate reward, and add a
candidate autonomous move/stay deviation only after it has an explicit gate
target and failure diagnostics.

The next viewer-visible checkpoint is implemented. `sim:run` accepts
`--mind-artifact ... --enable-mind` for explicit full-replay Mind runs, and the
viewer Decision Layer has a `Mind Fallback` mode that renders plain learned
actions, confidence delegation, and hard-guard interventions as distinct map
cues. A seed `5`, 120-tick replay from the calibrated advantage artifact
produced `4,204` trajectory decisions: `2,303` plain learned actions, `1,542`
confidence delegates, `358` hard guards, and `1` passive action. This makes the
remaining fallback clusters inspectable in the browser instead of only in gate
JSON.

The value-head viewer checkpoint is
`output/sim-runs/mind-value-viewer-check.json`. On the same seed and horizon it
produced `4,204` trajectory decisions: `2,313` plain learned actions, `1,522`
confidence delegates, `368` hard guards, and `1` passive action.

The value-deviation gate artifacts are
`output/mind/mind-v1-gate-value-deviation-artifact.json` and
`output/mind/mind-v1-gate-value-deviation-report.json`. They are better suited
for metrics than visual inspection because the accepted value-supported action
count is intentionally tiny.
A seed `41`, 120-tick full replay for this artifact is available at
`output/sim-runs/mind-value-deviation-viewer-check.json`; it contains `1,064`
plain learned actions, `768` confidence delegates, `198` hard guards, and `6`
passive records.

Rejected tuning paths are documented so they are not rediscovered as false
progress: prior-corrected action-lift variants improved some offline metrics
but raised confidence delegation, broad safe local-eat/plant-move deviations
caused alive-agent regressions on held-out seeds, and loose value-supported
local eating also regressed held-out alive/birth outcomes. Those paths should
only be reopened with a stronger feature/model change and the extended gate as
the arbiter.

Exit criteria:

- Default and extended gates remain green with no negative per-seed alive or
  birth deltas.
- Guard fallback share stays below the aggregate `0.45` broad cap, below the
  strict `0.1190` control target for promotion, and below per role/mode `0.50`
  caps.
- Total heuristic fallback is gated explicitly and must fall below the current
  rounded control value `0.4780` before promotion.
- Heuristic delegation share is reported separately and trends downward only
  through genuine learned-policy improvements, not hidden guard reclassification.
- Imitation accuracy and action-distribution drift are reported separately for
  train and held-out artifact datasets.
- Any new trainer remains deterministic under fixed seed and writes a versioned
  artifact.

### Stage 3: Offline Evaluation Hardening

Status: checkpoint complete; keep extending this matrix before runtime
experiments.

Broaden validation beyond the default matrix before using learned actions as a
runtime behavior candidate. This stage now has a documented extended seed
matrix, a longer 180-tick horizon, and compact diagnostics for guarded and
delegated contexts. Continue extending horizons and held-out seeds while the
next learner tries to reduce delegation through real learned value.

Exit criteria:

- A longer-horizon opt-in gate exists and is documented.
- Reports include per-context override/guard summaries for failure analysis.
- Held-out results are reproducible from npm entrypoints.
- Any regression is surfaced as a blocker or warning, not hidden by thresholds.

### Stage 4: Guarded Runtime Experiments

Status: first data-collection and neural artifact slices available; still
blocked for runtime replacement until a neural/offline-to-online model reduces
heuristic delegation while keeping Stages 2 and 3 green.

Run learned-controller experiments only as opt-in probes. The heuristic remains
the safety floor, and learned actions must not be used in release/default paths
until the runtime experiment gate clears.

The first online-oriented boundary is implemented through summary-only learned
trajectory collection:

```bash
npm run sim:trajectory -- \
  --seed 41 \
  --ticks 120 \
  --output output/trajectories/mind-v2-learned-seed41.jsonl.gz \
  --split-id mind-v2-learned-rollout \
  --mind-artifact output/mind/mind-v1-gate-value-deviation-artifact.json \
  --enable-mind
```

This does not train inside a live run. It collects experience from the learned
policy so the next artifact can be trained from mixed heuristic and learned
rollouts, then promoted through the same held-out gates. The neural and
offline-to-online algorithm roadmap is in
`docs/mind-v2-online-neural-roadmap.md`.

The first neural artifact path is implemented as an opt-in trainer:

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer neural-actor-critic-bc \
  --output output/mind/seed7-neural-artifact.json
```

This writes `guarded_neural_actor_critic_bc_v1`, a deterministic pure-Python
actor-critic MLP with actor logits, action-value heads, and a state-value head.
The artifact loader can run it as `mind_v2_neural_policy` through the existing
policy boundary and learned-policy trajectory command. It is a controller
contract milestone, not a promotion: no in-run weight updates are allowed and
the heuristic guard/delegation floor remains active.

The first real ML-backed trainer is also implemented as an opt-in artifact
builder:

```bash
python3 -m pip install -r requirements-mind-ml.txt
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-actor-critic-bc \
  --output output/mind/seed7-torch-artifact.json
```

This writes `guarded_torch_actor_critic_bc_v1`. PyTorch is used only at training
time; the promoted simulator path remains artifact-loaded, deterministic, and
guarded. The artifact records train-set actor/value diagnostics and the exact
optional ML package versions used at training time. The next gate work is to
compare this trainer against the current value-deviation control without
relaxing alive/birth, hard guard, or total fallback thresholds.

The first conservative offline-RL bridge trainer is implemented as another
opt-in artifact builder:

```bash
python3 -m pip install -r requirements-mind-ml.txt
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-advantage-actor-critic-bc \
  --output output/mind/seed7-torch-advantage-artifact.json
```

This writes `guarded_torch_advantage_actor_critic_bc_v1`. It keeps the same
serialized runtime network but uses contextual advantage-weighted actor loss
with a conservative `0.25` blend back toward behavior cloning. `mind_gate`
also now writes held-out neural calibration diagnostics for neural artifacts,
including actor accuracy, action-value/state-value error, score margin buckets,
action-value margin buckets, and predicted-advantage buckets.

Current probe result: the 256-hidden-unit, class-balanced PyTorch trainer is a
real improvement over the first neural scaffold but is not promoted. On
validation seeds `5,13` for `120` ticks it kept per-seed alive/birth deltas at
zero or better and reached total fallback `0.4815`, just above the `0.4780`
control target. Hard guard remained too high at `0.1418` versus the `0.1190`
target, so the next slice is critic/value calibration rather than wider guard
exceptions.

Advantage trainer probe result: the conservative advantage trainer passed the
same two-seed probe with alive delta `+0.5` and births delta `+1.0`, but it is
not promoted. Hard guard was `0.1668` and total fallback was `0.5769`, worse
than both the control and the plain PyTorch behavior-cloning trainer. The useful
signal is diagnostic: held-out neural value error improved (`action_value` MAE
`0.1009`, `state_value` MAE `0.1915`), but actor acceptance still needs
calibration-aware delegation rather than stronger advantage pressure.

The first neural calibration-aware acceptance slice is implemented but not
promoted. Neural artifacts now carry explicit
`value_supported_deviation_min_score_margin` and
`value_supported_deviation_min_predicted_advantage` thresholds. Runtime uses
them only for a narrow high-margin, positive-advantage local `eat` deviation
with safe vitals, local food, no local hazard, and no heuristic attack/drink
suppression. A looser version that accepted `drink` actions was rejected because
it failed the two-seed gate (`alive_delta -16.0`, `births_delta -15.5`). The
accepted thresholds (`score_margin >= 0.25`, predicted advantage `>= 0.24`)
passed the same two-seed probe with `13` safe neural `eat` deviations, hard
guard `0.1641`, confidence delegation `0.4077`, total fallback `0.5718`,
alive delta `+1.5`, and births delta `+1.5`. This proves the calibration path is
wired and gate-measured, but it does not beat the control and is not a promoted
controller.

The first transition-based discrete IQL trainer is implemented as
`torch-discrete-iql`. It trains from episode-aware
`(observation, action, reward, next_observation, done, action_mask,
next_action_mask)` transitions, with a selected-action Q head, expectile V head,
and masked advantage-weighted actor extraction. The corrected default reused
seed-bank gate wrote `output/mind/mind-v2-torch-discrete-iql-report.json` and
passed the broad default gate, but it is not promoted. With runtime
value-supported deviations disabled, hard guard fell to `0.0014`, confidence
delegation rose to `0.5820`, total fallback was `0.5834`, safe-deviation rate
was `0.0`, and every validation seed matched heuristic alive/birth outcomes.

The follow-up mask-normalized confidence probe renormalized neural actor scores
over legal actions before runtime delegation and diagnostics. It is also not
promoted: confidence delegation fell to `0.0062`, but hard guard rose to
`0.5772` and total fallback stayed `0.5834`, failing the broad hard-guard cap
while preserving zero alive/birth deltas. The result is a real offline-RL
harness and a sharper diagnosis, not a controller replacement; the next IQL/CQL
slice must improve actor safety/ranking before any runtime deviation path is
reconsidered.

The first neural/RL strict-control candidate is now implemented for
`torch-discrete-iql`. The candidate keeps value-supported deviations disabled,
renormalizes neural actor scores over the legal action mask, blends the IQL actor
with the contextual prior at weight `0.9`, and raises only the IQL artifact's
confidence-delegate margin to `0.25`. On the default reused seed-bank gate it
passed as `strict_control_candidate`: hard guard `0.1041`, confidence delegation
`0.3707`, total fallback `0.4748`, safe deviation `0.0`, and zero per-seed
alive/birth deltas. This is promotion progress, not a full autonomous
replacement: heuristic fallback is still substantial.

The same IQL candidate also passed the extended strict matrix across seeds
`5,13,19,29,37,41` and tick horizons `120,180`: hard guard `0.1054`,
confidence delegation `0.3661`, total fallback `0.4715`, safe deviation `0.0`,
and zero per-seed alive/birth deltas. This clears the current strict fallback
and outcome gates for promotion review, but it should remain opt-in until the
remaining heuristic delegation is explicitly accepted or reduced by the next
learner slice.

The next offline-to-online slice has started. Learned-policy trajectory
collection can opt in to persisted `policy_decision_diagnostics`, and the
transition adapter now carries `action_source` plus that optional diagnostic
payload beside `next_observation`, `next_action_mask`, and `done`. The
`torch-discrete-iql` trainer consumes those diagnostics as guard-aware actor
feedback: when a learned action was suppressed by the hard safety floor or
confidence delegate, training applies a masked margin penalty so the resolved
action outranks the suppressed learned action in that context. This is still a
frozen offline retrain, not in-run weight mutation. Gate probes can include
learned rollout replay with `--extra-trajectory`, and those probes still append
to `output/mind/mind-experiment-ledger.jsonl`. Learned rollout self-actions are
downweighted during IQL replay so the current policy does not simply imitate its
own unpromoted actions; guard/delegate records keep a stronger training weight
because they carry explicit safety feedback. A stricter `0.221` delegate-margin
sweep did reduce delegation below the current extended candidate, but hard guard
exceeded the strict cap, so the retained replay path keeps the safer IQL `0.25`
delegate margin until actor safety improves further.

CQL-style critic regularization and a stronger replay-weighted behavior anchor
were also added as measured trainer hooks. Active CQL variants reduced hard
guard but increased heuristic delegation enough to miss the strict total
fallback target. A behavior-anchor-only two-seed probe passed, but the default
four-seed probe failed at total fallback `0.4794` against the `0.4779` cap.
Both loss weights therefore remain `0.0` by default; do not promote this path
until a later calibration slice lowers delegation and hard guard together.

An opt-in heuristic-free runtime surface has started. `run_headless` and
`collect_trajectory` accept `--mind-runtime-mode autonomous`, which strips the
artifact's heuristic guard/delegate fields at load time, and
`--mind-runtime-mode autonomous-online`, which adds
`in_run_contextual_bandit_adapter_v1` on top. The adapter learns bounded
context/action score offsets from finalized trajectory reward records via a
policy feedback hook; it is useful for observation and replay collection, but it
is not a promoted controller and does not update neural weights in place.
That path is now gateable as an explicit experiment surface: trajectory
collection can persist `policy_update_trace` with
`--include-policy-update-trace`, and `mind_gate --compare-runtime-mode
autonomous-online` evaluates the stateful runtime with a fresh policy instance
per validation run. Gate reports and experiment ledger rows include runtime-mode
comparison summaries so failed autonomy probes do not get repeated as if they
were unmeasured. The first default comparisons reject the current contextual
adapter: the default contextual prior reaches zero fallback in
`autonomous-online` but regresses alive agents by `-28.5` and births by `-18.75`;
the IQL strict candidate also reaches zero fallback in `autonomous-online` but
regresses alive agents by `-30.25` and births by `-21.5` after `10501`
replayable online updates. This is useful gate evidence, not promotion evidence.

Those replayable updates now feed the offline trainer as a measured experiment:
`torch-discrete-iql` reads `mind_policy_update_trace_v1` through the transition
adapter and applies a signed actor-margin loss for the updated action. The first
default replay-mixed probe lowered total fallback to `0.4729` but raised hard
guard to `0.1405`, so it is explicitly not promoted. The next autonomy slice
should add a stronger constraint/viability value signal before any further
heuristic-free controller promotion attempt.

The first viability slice is diagnostics-only. `mind_viability_critic_diagnostics_v0`
labels existing transitions for short-horizon survival risk, floor-pressure
risk, invalid actions, reproduction viability, and hard-guard/delegate
suppression, then reports value-risk calibration by bucket, action, trophic
role, and meat mode. On the strict IQL candidate, held-out constraint risk is
`0.1143`, risk AUC is `0.6940`, and logged `stay` risk is underpredicted
(`0.7361` realized versus `0.0487` mean risk score). The next slice added a
serialized `multi_component_constraint_viability_head_v1` to the discrete IQL
network and a standalone `sim:mind:diagnostics` report writer. On the same
held-out bank, the trained state head improves risk AUC to `0.7697` and raises
the logged `stay` mean score to `0.4320`, but worsens Brier to `0.1554`. The
following slice adds an
`action_conditioned_multi_component_constraint_viability_head_v1` component
head. It improves held-out AUC to `0.9496`, Brier to `0.0572`, and `stay` mean
risk score to `0.9515`, while a separate calibration bank reports weaker but
usable AUC/Brier (`0.8709` / `0.1106`). The heuristic-collected diagnostic bank
has no positive hard-guard/delegate suppression labels; the first learned-policy
diagnostic probe supplies `1464` positives. The follow-up suppression-routed
replay slice trains those positives on the suppressed learned action rather than
the resolved fallback action, improving the suppression component to `0.6425`
AUC and `0.2924` Brier, but the guarded policy eval still misses strict total
fallback (`0.4800` versus `0.4780`). Actor extraction still must not use this
signal until suppression calibration improves and strict gates pass. The next
label-policy fix observes suppression only for learned-policy decisions and
masks unobserved suppression components in diagnostics. That v2 artifact moves
guarded total fallback closer to control (`0.4784`) with hard guard `0.0885`,
but still misses the `0.4780` target; its learned-suppression probe remains
overconfident (aggregate Brier/AUC `0.4175`/`0.5908`, suppression component
`0.4776`/`0.6693`). A `0.245` delegate-margin probe is rejected because total
fallback remains `0.4785` after hard guard rises. A detached
viability-head representation probe is also rejected as a default: it worsened
held-out Brier/AUC to `0.1590`/`0.8568`, calibration-bank Brier/AUC to
`0.2036`/`0.7624`, learned-suppression Brier/AUC to `0.2778`/`0.5669`, and
guarded total fallback to `0.4825`. The trainer default remains
`shared_hidden_auxiliary_heads_v1`; detached heads are available only through
the opt-in `--torch-iql-detach-viability-heads` experiment flag.
The follow-up observed-mask action-viability weight fix is retained as a
training correctness change: action-head positive weights are now computed only
from observed labels, so learned suppression uses weight `1.0576` instead of the
global `8.0`, and state-head suppression supervision is disabled. It improves
held-out/calibration Brier/AUC to `0.0555`/`0.9426` and `0.1036`/`0.8542`, and
learned-suppression component Brier/AUC to `0.3788`/`0.6789`. It is still not
promotion progress because guarded fallback worsens to `0.4825` total
(`0.0880` hard guard plus `0.3945` delegation) and `autonomous-online` still
fails alive/birth deltas (`-28.25`/`-23.5`). The next learner slice should focus
on reducing delegation without pushing decisions back into hard guard.

The following actor-confidence slice added an opt-in viability-safe logged
action margin anchor and a torch-IQL-only neural-prior blend override. This
produced measurable guarded progress but not promotion. The best blend sweep
candidate used `--torch-iql-behavior-margin-anchor` and
`--torch-iql-neural-actor-prior-blend-weight 0.75`, reaching hard guard
`0.0576`, delegation `0.4207`, total fallback `0.4783`, and zero guarded
alive/birth deltas on the default held-out policy eval. The adjacent `0.70`
blend worsened total fallback to `0.4785`, so further runtime-blend tuning is
not enough. This line should now move back into actor/Q calibration and
confidence extraction rather than runtime blend tuning.

The follow-up actor/Q calibration slice replaced pure standardized actor
weights with behavior-anchored standardized IQL advantage extraction. Pure
batch standardization is recorded as rejected (`0.4835` total fallback and
autonomous-online alive/birth deltas `-34.75`/`-27.75`). The anchored variant
keeps training weights near behavior support (`0.7141` min, `1.9325` max,
`1.0017` mean) and passes guarded strict-control targets on both default and
extended gates: default hard guard/delegation/total fallback
`0.0568`/`0.4172`/`0.4740`, extended `0.0582`/`0.4118`/`0.4700`, with zero
min per-seed alive/birth deltas. This is the current guarded promotion-review
candidate. It is not the autonomous Mind endpoint because `autonomous-online`
still fails outcomes and held-out Q/V MAE remains high (`4.3502`/`4.7762`).

The next hard-guard/delegation tradeoff batch did not beat that boundary.
A delegate-margin actor finetune term worsened the risk-adjusted contextual
prior candidate to `0.0577` hard guard and `0.4764` total fallback. Applying
runtime-feedback finetune to the anchored calibrated actor control worsened it
to `0.0589` hard guard and `0.4792` total fallback. A value-supported deviation
probe verified a real source-handling bug for neural actor prior blends and
retains that fix, but the strict thresholds were a no-op (`safe_deviation_count=0`)
and the relaxed `-0.10` predicted-advantage floor regressed guarded outcomes
(`-5.0` alive, `-5.0` births). The retained promotion-review control therefore
remains the anchored calibrated actor artifact, and the next work should target
critic/state-value calibration before any further safe-deviation or delegation
bypass.
The first suppression-aware critic calibration implementation is also rejected
as a promotion path. It adds an opt-in Q/V margin loss on `1,459`
runtime-suppressed learned-action rows, but the full risk-adjusted contextual
prior gate worsened to `0.0576` hard guard and `0.4794` total fallback. Removing
the detached risk-adjusted actor distillation did not rescue it (`0.0588` hard
guard and `0.4783` total fallback). The useful signal is diagnostic: blunt
suppressed-action Q/V margins improve some autonomous-online deltas but damage
guarded fallback, so the next learner work should focus on calibrated critic
ranking and constraint-aware offline-to-online actor extraction before another
autonomy attempt.

The first two follow-up probes are recorded and rejected. Discounted-return Q/V
calibration added durable return-target and return-MAE diagnostics, but the
auxiliary loss missed the guarded control (`0.4777` total fallback) and worsened
autonomous-online deltas to `-35.5` alive and `-29.25` births. Observed
viability-risk actor filtering touched `12.8%` of logged actions and improved
autonomous-online deltas to `-27.0` alive and `-19.25` births, but guarded total
fallback was still `0.4767`, above the current promotion-review artifact
(`0.4740`). Keep both flags opt-in and continue with calibrated critic/actor
quality, not threshold relaxation.
The next risk-adjusted actor distillation probe used detached action viability
risk to target `Q advantage - risk` over legal actions. It improved guarded
fallback versus logged-risk filtering (`0.4759`) but still missed the
promotion-review artifact and worsened autonomous-online outcomes to `-36.25`
alive and `-29.0` births. This keeps the direction diagnostic-only: raw
action-risk scores are useful for ranking pressure but not calibrated enough to
drive actor targets directly.

Calibrated Supported Actor Extraction v1 is now implemented as the bounded
follow-up rather than another one-off probe flag. It requires a separate
calibration bank via `--calibration-trajectory`, reports Brier/AUC/ECE and
reliability buckets by action and component, applies per-action support
constraints, and extracts actor targets only for legal actions with positive
calibrated advantage, calibrated risk below threshold, and enough support. The
default gate rejected the first artifact against the current promotion-review
control: hard guard/delegation/total fallback
`0.1047`/`0.3712`/`0.4759`, with zero alive/birth deltas. Calibration improved
the risk probability signal (`0.054816` raw Brier and `0.088814` raw ECE to
`0.035617` Brier and `0.032653` ECE), but the extracted targets still collapsed
toward `eat` and `drink` (`42,150` of `42,685` selected targets), which reduced
delegation by raising hard guard. The next learner slice should add contextual
support or behavior-proximity constraints before letting Q/V advantage choose
free legal actions.

Contextual Behavior-Proximity Supported Extraction v2 is implemented and
recorded as another rejected-but-useful actor extraction slice. It adds a
separate `--calibration-validation-trajectory` bank, context/action support,
sparse-context-only fallback to coarser feature keys, target action expansion
caps, family conversion caps, and movement/stay-to-resource conversion
diagnostics. After the final cap-enforcement fix, the gate artifact improves
guarded total fallback to `0.4717` with zero alive/birth deltas and reduces
target/logged TVD to `0.1800`, with movement/stay-to-resource conversion capped
at `0.1800` (`2,489/13,830`). Hard guard is still `0.1167`, far above the
current promotion-review control `0.0568`, and autonomous-online fails at
`-37.0` alive / `-31.5` births. Keep it opt-in and rejected; the next actor
slice should train contextual behavior proximity directly rather than relying on
post-hoc target rejection.

Contextual behavior-prior actor regularization is now implemented as that first
trained behavior-proximity objective. The prior-only arm, which does not enable
contextual supported target extraction, is useful but still rejected: guarded
hard/delegate/total fallback is `0.0663` / `0.4037` / `0.4700` with zero
guarded alive/birth deltas, while autonomous-online still fails at `-35.0`
alive / `-28.5` births. The combined behavior-prior plus v2 target-extraction
arm is worse than prior-only (`0.1031` / `0.3696` / `0.4727`) and leaves v2
target diagnostics unchanged because targets are precomputed before actor
finetune. The next actor-objective batch added finetune carryovers for the
existing runtime-suppressed-action margin and viability-safe logged-action
margin. The non-risk prior variant now beats total fallback but still misses
hard guard (`0.0582` / `0.4155` / `0.4737`). Adding the existing detached
Q-minus-risk actor loss is the closest result so far: hard guard clears the
promotion-review control at `0.0567`, guarded alive/birth deltas stay at zero,
but total fallback misses by `0.0006` (`0.4746` versus `0.4740`).
Constraint-aware stacking, a stronger finetune margin-anchor weight, and
hard-guard-only finetune carryover were all tested and rejected. Keep the prior
objective and finetune carryovers opt-in; the remaining blocker is no longer
target extraction caps but the hard-guard/delegation tradeoff in the actor
objective.

Keep GPU/CUDA as a separate infrastructure slice. Switch to NVIDIA CUDA only
when repeated training grids, longer seed banks, or extended gates make CPU
iteration the bottleneck. The GPU slice should add `--torch-device` with
`cpu|cuda|mps|auto` choices, default to `cpu`, record requested/resolved device
and torch/CUDA/MPS metadata in artifacts, serialize weights back to CPU JSON,
and require CPU/Mac gate validation before promotion.

The rejected advantage-blended neural anchor is also recorded. It kept total
fallback below control at `0.4763`, but hard guard rose to `0.1459`; do not use
that anchor as the IQL default unless a future actor-safety change fixes the
hard-guard regression.

The actor action-distribution objective is also recorded and rejected. The soft
logged-action marginal KL variant corrected probability mass but not top-1
policy behavior and failed the guarded gate at `0.0591` / `0.4156` / `0.4747`.
The sharpened argmax-proxy KL variant worsened total fallback to `0.4841` and
autonomous-online to `-33.0` alive / `-26.25` births. Keep the hook opt-in for
diagnostics only. The next large actor milestone should stop treating the
guarded fallback boundary as the main learning signal and instead add
closed-loop survival or context-local sequence/navigation supervision that can
survive without the heuristic safety floor.

Exit criteria:

- Runtime experiment command is opt-in and documented.
- Learned-policy summary-only trajectory collection remains opt-in and writes
  the same `mind_trajectory_v1` contract as heuristic collection.
- Neural actor-critic artifacts declare backend, architecture, input size,
  hidden width, value heads, and deterministic seed metadata.
- PyTorch-backed actor-critic training is optional, documented, and serializes
  dependency-free runtime artifacts.
- Advantage-weighted actor-critic training is optional, documented, and gated as
  an experiment harness.
- Discrete IQL training is optional, documented, transition-based, and gated as
  an experiment harness until strict fallback and zero per-seed outcome targets
  pass.
- The torch-IQL behavior-margin anchor and neural actor prior blend override are
  optional promotion-probe controls only. They are not release defaults because
  the best measured candidate still misses total fallback by `0.0006`.
- Neural actor confidence is mask-renormalized before runtime delegation, and
  diagnostics report that normalization policy.
- The current IQL strict-control candidate has passed the extended strict
  matrix, but remains opt-in pending promotion review.
- Heuristic-free and autonomous-online runtime modes are executable only as
  opt-in observation/probe surfaces; they are not release defaults or promotion
  proof.
- Online update traces are serializable and replayable before any autonomous
  runtime-mode comparison can be treated as gate evidence.
- Runtime-mode ledger rows include fallback, alive/birth deltas, and update counts
  so zero-fallback failures cannot be mistaken for progress.
- Learned policy stays disabled by default in normal simulator runs.
- Runtime experiment reports compare heuristic, guarded learned, and any
  stronger offline model on the same seeds and horizons.
- No release/Foundation gate depends on learned-controller behavior.

### Stage 5: Mind v1 Phase Boundary

Status: future.

Mind v1 can move to the next phase only after the stronger offline model and
guarded runtime experiments are repeatably green. The next phase can then focus
on broader model classes or learned-controller integration without weakening
Foundation guarantees.

Exit criteria:

- Full validation ladder passes after the Mind runtime experiment slice.
- Extended Mind gates show no Foundation regression.
- Benchmarks quantify runtime cost of learned evaluation.
- The default simulator remains deterministic and heuristic-backed unless a
  Mind runtime flag is explicitly enabled.
