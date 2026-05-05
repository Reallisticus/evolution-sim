# Mind v1 Data Contract

Current contract version: `mind_v1_data_contract_v1`

Mind v1 is data tooling only at this stage. Runtime learned-policy inference is
disabled by default and requires an explicit enable flag. Do not add Stage 2/3
semantics on top of Mind tooling until Foundation release gates are clean.

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
default it collects train seeds `3,7,11,17` for 120 ticks, trains the guarded
seed-bank artifact, evaluates validation seeds `5,13,19,29`, and writes one
report with trajectory collection summaries, artifact provenance, evaluation
output, explicit gate criteria, and readiness status. The default gate criteria
permit at most `0.5` terminal alive-agent mean regression versus the heuristic
over the held-out seed bank, at most `2.0` alive-agent regression on any single
validation seed, and at most `1.0` births regression before review on any single
validation seed. Policy-visible invalid action rate is capped at `0.02`; the
current baseline reports `0.0`. The CLI supports `--fail-on-blockers` for hard
gate failures and `--fail-on-review` when review warnings should also produce a
non-zero exit. Evaluation reports include paired per-seed heuristic versus
learned deltas for alive agents, births, and deaths so aggregate regressions can
be traced to individual validation seeds.

The gate report also includes artifact diagnostics computed from the training
trajectories before runtime evaluation:

- imitation top-1 accuracy against the behavior-cloning label;
- predicted versus label action distribution drift;
- contextual feature coverage and fallback depth.

Runtime evaluation diagnostics are computed from trajectory records without
changing the trajectory schema. They report action-source counts, guard
intervention share, and per-trophic-role/per-meat-mode mean reward, action
counts, and guard intervention rates. They also report guard intervention
breakdowns by final requested action plus the top policy-visible context
buckets where the guard fired, using the vitals/role/action-mask feature depth
from the contextual baseline.

The guarded contextual baseline only materializes conditional action priors once
the feature context has at least 12 training records. Lower-support contexts
fall back to coarser feature keys or the global prior, which keeps offline
imitation from overfitting sparse seed-bank states before runtime experiments
are allowed beyond the heuristic safety floor.

The learned adapter also requires a decisive offline margin before overriding a
different heuristic action: the learned action's score must exceed the
heuristic action's score by `1.0`. Because scores are normalized action
frequencies, this only permits unanimous contextual overrides; lower-confidence
or ambiguous disagreements remain on the heuristic safety floor.

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

It evaluates held-out seeds `1,2,4,5,6,8,9,10,12` at 120 and 180 ticks, reusing
the seed-bank training trajectories, and writes
`output/mind/mind-v1-gate-extended-report.json`.

The phase boundary and staged path beyond this guarded offline baseline are
tracked in `docs/mind-v1-stage-plan.md`.
