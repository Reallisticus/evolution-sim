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
confidence partitioning. Reward-scale sweeps from `0.75` through `4.0` did not
beat the current `2.0` scale. Action-family floors for movement/attack actions
reduced the candidate back near the control total but did not cross the strict
fallback cap. Local-eat safe-deviation probes reduced fallback but changed
per-seed alive/birth outcomes, so safe deviations remain disabled.

Rejected tuning paths are documented so they are not rediscovered as false
progress: prior-corrected action-lift variants improved some offline metrics
but raised confidence delegation, and safe local-eat/plant-move deviations
caused alive-agent regressions on held-out seeds. Those paths should only be
reopened with a stronger feature/model change and the extended gate as the
arbiter.

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

Status: blocked until a stronger offline model reduces heuristic delegation
while keeping Stages 2 and 3 green.

Run learned-controller experiments only as opt-in probes. The heuristic remains
the safety floor, and learned actions must not be used in release/default paths
until the runtime experiment gate clears.

Exit criteria:

- Runtime experiment command is opt-in and documented.
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
