# Mind v1 Data Contract

Current contract version: `mind_v1_data_contract_v1`

Mind v1 is still disabled by default. Runtime learned-policy inference requires
an explicit enable flag, and Stage 2+ policy changes remain offline-gated until
they preserve Foundation release behavior and held-out Mind gates.

## Schemas

- Shared summary schema: `foundation_summary_v1`
- Trajectory schema: `mind_trajectory_v1`
- Observation schema: `mind_observation_v3`
- Observation encoder: `mind_observation_encoder_v2`
- Policy interface: `mind_policy_interface_v1`
- Action contract: `mind_action_contract_v1`
- Reward schema: `mind_reward_v1`
- Action outcome schema: `mind_action_outcome_v2`
- Model artifact schema: `mind_model_artifact_v1`

The canonical runtime view is exposed by
`evolution_sim.mind.contracts.mind_v1_data_contract()`.

## Trajectory JSONL

Trajectory JSONL is independent of replay. A valid dataset has:

- a header row with `format=evolution_sim_trajectory_jsonl_v1` and the current
  trajectory, observation, policy, action, reward, reproductive-group, and
  recombination versions;
- record rows that include every field in `TRAJECTORY_RECORD_FIELDS`;
- a footer row with the run summary and trajectory summary.

`evolution_sim.mind.dataset.load_trajectory_jsonl()` validates the header,
records, and footer before returning records for offline training.

## Offline Baseline

`evolution_sim.mind.baseline.train_behavior_cloning_baseline()` trains a tiny
guarded contextual action-prior behavior-cloning model from strict trajectory
records. It buckets policy-visible vitals, trophic mode, local resource hints,
navigation hints, and action masks. Runtime loading enables a heuristic safety
floor for survival/navigation conflicts, so this baseline proves dataset,
artifact, adapter, and gate plumbing without claiming learned-controller
quality. The default trainer is `contextual-prior`; it writes model type
`guarded_contextual_local_prior_bc_v2`, uses `uniform_v1` sample weighting, and
requires `sample_weight_total == trained_record_count` in artifact validation.

`npm run sim:mind:train -- --trajectory <path> --output <artifact>` validates
the trajectory file before writing an artifact. Repeat `--trajectory` to train a
single seed-bank baseline from multiple JSONL/JSONL.GZ streams. Combined
artifacts preserve first-seen source seed order, source trajectory paths, total
record count, source dataset count, and a combined config digest; all sources
must share the same data-contract digest.

The training CLI also accepts
`--trainer reward-weighted-contextual-prior`. This opt-in trainer keeps the same
contextual prior structure but weights each record by
`max(0.05, 1.0 + reward.total)` before normalizing action scores. It writes
model type `guarded_reward_weighted_contextual_prior_bc_v1`, records
`reward_total_shifted_clamp_v1` sample-weight metadata, and keeps unweighted
support counts separate from weighted counts so diagnostics still expose how
much raw data backed each decision. This path is experimental until it reduces
guard/delegate fallback on held-out gates without alive or birth regressions.

The CLI also accepts `--trainer advantage-calibrated-contextual-prior`. This
trainer is an opt-in experiment harness that keeps the contextual-prior model
shape but adjusts per-action support by reward advantage inside the matched
context: action mean reward minus context mean reward. The adjustment is
clamped to `[0.25, 2.0]`, uses scale `2.0`, and only applies when at least two
actions have support `>= 2` in that context. It writes model type
`guarded_advantage_calibrated_contextual_prior_bc_v1`, sample-weight policy
`contextual_reward_advantage_adjusted_counts_v1`, and explicit
`contextual_reward_advantage_lift_v1` metadata. Each advantage-calibrated score
metadata block must also carry `delegate_score_margin`: a conservative runtime
confidence margin for the learned top action, bounded by the adjusted
`score_margin` and computed from unadjusted support. This keeps reward-lifted
action scores from being treated as high-confidence evidence unless raw support
also backs the learned top action. This path is not promoted by default; the
same held-out gates decide whether it is better than the current control.

The CLI also accepts `--trainer advantage-blended-contextual-prior`. This
candidate keeps the same reward-advantage calculation but blends the normalized
uniform contextual prior with the normalized advantage-calibrated prior at
weight `0.2`. It writes model type
`guarded_advantage_blended_contextual_prior_bc_v1`, sample-weight policy
`contextual_reward_advantage_blended_counts_v1`,
`contextual_reward_advantage_lift_v1`, and
`contextual_reward_advantage_score_blend_v1` metadata. Runtime guard and
delegate behavior is unchanged; the blend only changes learned action scores so
advantage evidence is anchored to the behavior-cloning prior unless reward lift
is strong enough to move the ranking.

The CLI also accepts `--trainer value-calibrated-contextual-prior`. This
opt-in learner adds a deterministic value head without changing the runtime
policy boundary: it estimates mean `reward.total` per action for the global and
matched contextual buckets, converts supported value differences into a
normalized value preference, and blends that value preference at weight `0.05`
on top of the current `0.2` advantage-blended prior. It writes model type
`guarded_value_calibrated_contextual_prior_bc_v1`, sample-weight policy
`contextual_value_calibrated_score_blend_v1`,
`mean_reward_action_value_v1`, and `prior_value_score_blend_v1` metadata, plus
full-vocabulary `action_value_estimates` and
`conditional_action_value_estimates`. It also declares the opt-in
`positive_value_safe_deviation_v1` runtime policy with support and value-margin
thresholds. Runtime may bypass confidence delegation and the hard heuristic
guard only for local-resource `eat` actions over heuristic `stay` decisions when
the matched bucket has at least `32` records, the learned action value is at
least `0.02`, the learned-over-heuristic value margin is at least `0.04`, vitals
are safe, and the center tile has a compatible local resource. Artifact
validation requires those value maps to cover the complete action vocabulary
with finite values inside the declared reward-total bounds and requires the
value-supported deviation thresholds to be explicit. This remains guarded and
disabled by default.

The CLI also accepts `--trainer neural-actor-critic-bc`. This is the first
neural artifact path. It writes model type
`guarded_neural_actor_critic_bc_v1`, sample-weight policy
`neural_actor_critic_bc_uniform_v1`, backend
`pure_python_deterministic_v1`, architecture
`fixed_random_feature_mlp_actor_critic_v1`, and training policy
`one_pass_hidden_prototype_actor_critic_bc_v1`. The model payload includes a
fixed 8-unit `tanh` hidden layer over the current policy input vector, actor
output weights/biases for every action, action-value output weights/biases for
every action, and a state-value head. Artifact validation requires all neural
weight matrices to match the declared observation size, hidden width, and full
action vocabulary with finite values. Runtime inference re-encodes the live
policy-visible observation through the public observation encoder before
feeding the neural network, so it does not receive privileged world state. The
artifact remains guarded, immutable during a run, and disabled by default.

The CLI also accepts `--trainer torch-actor-critic-bc` after installing
`requirements-mind-ml.txt`. This writes model type
`guarded_torch_actor_critic_bc_v1`, sample-weight policy
`torch_actor_critic_bc_uniform_v1`, backend `pytorch_optional_v1`,
architecture `torch_mlp_actor_critic_v1`, and training policy
`adamw_balanced_cross_entropy_td0_actor_critic_v1`. Training uses PyTorch with
AdamW, class-balanced cross-entropy actor loss, TD(0)-style selected-action
value regression, and state-value regression. Artifact inference does not
import PyTorch: the trained weights are serialized into the same finite
actor/action-value/state-value schema and validated against the declared hidden
width and action vocabulary. Torch artifacts include train-set actor accuracy,
confidence margin, action-value/state-value absolute error, Python version, and
installed optional ML package versions for reproducibility. This is the first
real ML-backed Mind trainer, but it remains guarded, immutable during a run, and
disabled by default until strict gates beat the control.

The CLI also accepts `--trainer torch-advantage-actor-critic-bc` after
installing `requirements-mind-ml.txt`. This writes model type
`guarded_torch_advantage_actor_critic_bc_v1`, sample-weight policy
`torch_advantage_actor_critic_bc_contextual_advantage_weighted_v1`, backend
`pytorch_optional_v1`, architecture `torch_mlp_actor_critic_v1`, and training
policy `adamw_contextual_advantage_weighted_actor_critic_v1`. The actor loss is
class-balanced cross entropy multiplied by conservative contextual
advantage weights and blended back toward uniform behavior cloning. The value
heads, runtime inference path, artifact immutability, and disabled-by-default
guarding remain the same as the PyTorch behavior-cloning trainer.

The CLI also accepts `--trainer torch-discrete-iql` after installing
`requirements-mind-ml.txt`. This writes model type
`guarded_torch_discrete_iql_v1`, sample-weight policy
`torch_discrete_iql_transition_expectile_awbc_v1`, backend
`pytorch_optional_v1`, architecture `torch_mlp_actor_critic_v1`, and training
policy `adamw_discrete_iql_expectile_advantage_weighted_v1`. Training uses an
episode-aware trajectory-to-transition adapter with
`observation_input`, `action`, `reward_total`, `next_observation_input`,
`done`, `action_mask`, `next_action_mask`, and `episode_id`. The critic trains
a selected-action Q head against a TD(0) target, trains the V head with
expectile regression, and extracts a masked discrete actor with
advantage-weighted behavior cloning. Runtime inference still imports no
PyTorch and remains artifact-loaded, guarded, immutable, and disabled by
default. Neural actor scores are renormalized over the current legal action
mask before runtime confidence delegation and neural calibration score-margin
buckets. The current IQL artifact also declares
`neural_actor_prior_policy=contextual_prior_score_anchor_v1` and
`neural_actor_prior_blend_weight=0.9`; runtime blends the mask-renormalized
neural actor distribution with the matched contextual/global action prior before
ranking learned actions. IQL artifacts use an explicit trainer-scoped
`heuristic_delegate_max_training_score_margin` of `0.25`, leaving the promoted
contextual-prior baseline at `0.221`. Because held-out Q/V calibration is not
yet trustworthy enough for runtime bypass decisions, `torch-discrete-iql`
artifacts must not declare `positive_value_safe_deviation_v1` or any
value-supported deviation thresholds; artifact validation rejects those fields
for IQL until a future calibration contract explicitly enables them.

For neural artifacts, `mind_gate` now reports `neural_calibration` in both
train and held-out artifact diagnostics. These diagnostics include actor
top-1 accuracy, action-value and state-value absolute error, score-margin
buckets, action-value-margin buckets, and predicted-advantage buckets. They are
diagnostic only; promotion is still decided by held-out policy rollouts and
Mind gates. Neural calibration reports declare
`score_normalization_policy=mask_renormalized_neural_actor_scores_v1` so
confidence buckets match the same legal-action scoring used by runtime
delegation.

Neural and value-calibrated artifacts that enable
`positive_value_safe_deviation_v1` must also declare
`value_supported_deviation_min_score_margin` and
`value_supported_deviation_min_predicted_advantage`. Runtime uses these fields
as an explicit calibration contract before any neural value-supported local
resource action can bypass the heuristic floor. The current neural path is
limited to high-margin local `eat` actions with safe vitals and local food; it
does not authorize broad movement, attack, or drink deviations. The
`torch-discrete-iql` model type is explicitly excluded from this runtime
deviation path for now.

Artifacts must include a manifest with the current schema versions before
runtime inference is allowed. `load_learned_policy(..., enable_mind=True)` is
required; without the explicit flag, loading fails.

## Evaluation Gates

`npm run sim:mind:evaluate -- --artifact <path> --enable-mind` compares the
heuristic and learned policy over summary-only seeds with trajectory recording
enabled. The Mind v1 gate report checks:

- schema compatibility through the loaded artifact manifest;
- policy-visible invalid-action rate from trajectory summaries, with
  resolution-conflict rate reported separately;
- survival through viable-run share;
- reproduction through births per run;
- resource pressure through plant energy available per land tile.

`npm run sim:mind:gate` is the reproducible local gate for this baseline. By
default it collects train seeds `1,2,3,4,6,7,8,9,10,11,12,17,23,31` for 120
ticks, trains the guarded seed-bank artifact, evaluates validation seeds
`5,13,19,29`, and writes one report with trajectory collection summaries,
artifact provenance, evaluation output, explicit gate criteria, and readiness
status. The default gate criteria permit at most `0.5` terminal alive-agent mean
regression versus the heuristic over the held-out seed bank, at most `2.0`
alive-agent regression on any single validation seed, and at most `1.0` births
regression before review on any single validation seed. Policy-visible invalid
action rate is capped at `0.02`, aggregate guard intervention is capped at
`0.45`, total heuristic fallback (`guard_intervention_rate +
heuristic_delegate_rate`) defaults to the broad cap `1.0`, and each role/mode
guard intervention rate is capped at `0.50`. The CLI supports
`--fail-on-blockers` for hard gate failures and `--fail-on-review` when review
warnings should also produce a non-zero exit. Evaluation reports include paired
per-seed heuristic versus learned deltas for alive agents, births, and deaths so
aggregate regressions can be traced to individual validation seeds.

The current strict promotion target for the next learner is intentionally tighter
than the default broad gate. To require a model to beat the current control's
rounded default fallback metrics, run the gate with zero per-seed regression
allowance and inclusive caps just below the documented control values:

```bash
npm run sim:mind:gate:strict
```

The strict entrypoint is pinned to the current candidate trainer
`advantage-blended-contextual-prior`, the extended validation seed matrix
`5,13,19,29,37,41`, the `120,180` tick horizons, zero alive/birth regression,
hard-guard cap `0.1189`, total fallback cap `0.4779`, and
`--fail-on-blockers`. Candidate experiments can still use `npm run
sim:mind:gate -- --trainer <candidate-trainer> ...` directly, but the strict
script is the local promotion check.

The Mind gate accepts the same `--trainer` option and records the chosen trainer,
sample-weight policy, any reward-advantage blend metadata, and any value-head
metadata, including value-supported deviation score-margin and
predicted-advantage thresholds, in both `protocol` and `artifact` report
sections. It also appends a compact JSONL experiment ledger entry by default to
`output/mind/mind-experiment-ledger.jsonl`; use
`--experiment-ledger-output <path>` to redirect it or pass no path from the
Python API to skip ledger writes. Ledger entries include broad gate status,
strict control-target pass/fail, minimum per-seed alive/birth deltas, fallback
rates, and a decision label so rejected probes are searchable later. The gate
report also includes artifact diagnostics computed from both training
trajectories and a held-out artifact-diagnostic seed bank before runtime
evaluation:

- imitation top-1 accuracy against the behavior-cloning label;
- predicted versus label action distribution drift, per-action precision/recall,
  and top action-confusion pairs;
- contextual feature coverage, fallback depth, support buckets, and score-margin
  buckets.
- reward calibration by behavior-cloning label action, predicted action, and
  predicted score-margin bucket so value-aware learners can compare confidence
  against realized reward without reading runtime-private state.

Runtime evaluation diagnostics are computed from trajectory records without
changing the trajectory schema. They report action-source counts, guard
intervention share, confidence-delegation share, per-trophic-role and
per-meat-mode mean reward, action counts, guard intervention rates, and
confidence-delegation rates. They also report guard and delegation breakdowns by
final requested action, learned action suppressed or deferred, matched score
source, training-support bucket, score-margin bucket, and the top policy-visible
context buckets where either safety floor took control. Context buckets use the
vitals/role/action-mask feature depth from the contextual baseline. During
in-process evaluation, learned policies may attach decision diagnostics that are
summarized in the report but are not written into trajectory JSONL records and
do not change `mind_trajectory_v1`. The same sidecar carries the matched score
source, feature-key depth, training support, and learned score margin so guard
interventions and delegates can be grouped by calibration evidence.
Paired evaluation reports compare heuristic and guarded learned outcomes by
trophic role and meat mode, including terminal count deltas and role/mode policy
diagnostic deltas, without changing trajectory records.

The current Stage 2 guarded baseline carries explicit artifact parameters for a
confidence-delegation policy. If the learned action disagrees with the
observation heuristic and the matched training action prior has score margin
below `0.221`, runtime delegates to the heuristic before evaluating the hard
safety guard. The resulting action source uses
`observation_heuristic_confidence_delegate_v1`, and reports count this
separately from `observation_heuristic_safety_floor_v1` guard interventions.
This keeps low-confidence learned disagreements visible without treating them as
learned-controller value. Artifacts may provide a conservative
`delegate_score_margin` beside `score_margin`; when present, runtime uses that
margin for confidence delegation while leaving learned action ranking based on
the artifact action scores.

The artifact still carries nullable safe-deviation parameters for local eating
and plant movement, but the current baseline writes them as `null`. Those
bypasses remain disabled unless a future artifact explicitly supplies finite
thresholds and passes the extended gate.

Value-calibrated artifacts use a separate value-supported deviation policy
instead of those nullable heuristic-safe fields. The value path is narrower:
only high-support, positive-value local-resource `eat` actions over heuristic
`stay` decisions can bypass confidence delegation, and the selected value
estimates are surfaced in policy diagnostics as `learned_action_value` and
`heuristic_action_value`.

Full-replay runs can now opt into a Mind artifact for viewer inspection:

```bash
npm run sim:run -- \
  --seed 5 \
  --ticks 120 \
  --output output/sim-runs/mind-viewer-check.json \
  --mind-artifact output/mind/mind-v1-gate-advantage-artifact.json \
  --enable-mind
```

The explicit `--enable-mind` flag is required here for the same reason it is
required during evaluation: learned-policy inference remains disabled unless a
caller deliberately opts in. The resulting full replay keeps the existing
`mind_trajectory_v1` payload and records Mind action sources in trajectory
records. The viewer's Decision Layer includes a `Mind Fallback` mode that colors
plain learned actions, confidence delegation, and hard-guard interventions
separately so fallback clusters can be inspected on the map and in each agent
dossier.

Summary-only trajectory collection can also opt into a Mind artifact:

```bash
npm run sim:trajectory -- \
  --seed 41 \
  --ticks 120 \
  --output output/trajectories/mind-v2-learned-seed41.jsonl.gz \
  --split-id mind-v2-learned-rollout \
  --mind-artifact output/mind/mind-v1-gate-value-deviation-artifact.json \
  --enable-mind
```

This is the current online-learning boundary. The simulator may collect
experience from a learned policy, but policy weights are still immutable during a
run. In-simulation online updates remain disallowed until a future contract can
make update logs replayable, deterministic, and gateable. The machine-readable
online-learning ladder is exposed by `mind_online_learning_contract()`.

The guarded contextual baseline uses feature policy `mind_feature_policy_v2`.
It materializes conditional action priors once the feature context has at least
3 training records, then falls back through coarser feature keys or the global
prior. Conditional scores use `smoothed_contextual_action_prior_v1` with
additive smoothing `0.1`; the artifact also records the prior-correction
exponent, currently `0.0`, to make this a plain smoothed local prior rather than
a hidden action-lift model. This checkpoint improves held-out action drift and
guard fallback while preserving zero alive/birth deltas on the extended matrix.
Global-prior correction and safe-deviation runtime bypasses were tested but
left disabled because they did not preserve the gates.

The learned adapter also requires decisive evidence before overriding a
different heuristic action. First, low-margin training contexts delegate to the
heuristic. Then the hard guard requires the learned action's score to exceed the
heuristic action's score by `1.0` unless a validated safe-deviation path exists.
Because scores are normalized action frequencies and safe deviations are
currently disabled, nearly all ambiguous disagreements remain heuristic-backed.

Longer-horizon validation is supported through a validation matrix without
changing the training horizon:

```bash
npm run sim:mind:gate -- --reuse-trajectories --validation-ticks 120,240
```

The first horizon remains available as `evaluation` for compatibility, and all
horizons are listed under `evaluation_matrix`. Overall readiness fails or enters
review if any validation horizon emits blockers or warnings.

The broader opt-in entrypoint is:

```bash
npm run sim:mind:gate:extended
```

It evaluates held-out seeds `5,13,19,29,37,41` at 120 and 180 ticks, reusing the
seed-bank training trajectories where possible, and writes
`output/mind/mind-v1-gate-extended-report.json`.

The phase boundary and staged path beyond this guarded offline baseline are
tracked in `docs/mind-v1-stage-plan.md`.
