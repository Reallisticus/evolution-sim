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
quality.

`npm run sim:mind:train -- --trajectory <path> --output <artifact>` validates
the trajectory file before writing an artifact. Repeat `--trajectory` to train a
single seed-bank baseline from multiple JSONL/JSONL.GZ streams. Combined
artifacts preserve first-seen source seed order, source trajectory paths, total
record count, source dataset count, and a combined config digest; all sources
must share the same data-contract digest.

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
`0.45`, and each role/mode guard intervention rate is capped at `0.50`. The CLI
supports `--fail-on-blockers` for hard gate failures and `--fail-on-review` when
review warnings should also produce a non-zero exit. Evaluation reports include
paired per-seed heuristic versus learned deltas for alive agents, births, and
deaths so aggregate regressions can be traced to individual validation seeds.

The gate report also includes artifact diagnostics computed from both training
trajectories and a held-out artifact-diagnostic seed bank before runtime
evaluation:

- imitation top-1 accuracy against the behavior-cloning label;
- predicted versus label action distribution drift, per-action precision/recall,
  and top action-confusion pairs;
- contextual feature coverage, fallback depth, support buckets, and score-margin
  buckets.

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
learned-controller value.

The artifact still carries nullable safe-deviation parameters for local eating
and plant movement, but the current baseline writes them as `null`. Those
bypasses remain disabled unless a future artifact explicitly supplies finite
thresholds and passes the extended gate.

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
