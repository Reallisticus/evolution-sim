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

Status: first data-collection slice available; still blocked for runtime
replacement until a neural/offline-to-online model reduces heuristic delegation
while keeping Stages 2 and 3 green.

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

Exit criteria:

- Runtime experiment command is opt-in and documented.
- Learned-policy summary-only trajectory collection remains opt-in and writes
  the same `mind_trajectory_v1` contract as heuristic collection.
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
