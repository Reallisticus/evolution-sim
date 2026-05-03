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
global action-prior behavior-cloning model from strict trajectory records. It is
intentionally small: it proves dataset, artifact, adapter, and gate plumbing
without claiming learned-controller quality.

Artifacts must include a manifest with the current schema versions before
runtime inference is allowed. `load_learned_policy(..., enable_mind=True)` is
required; without the explicit flag, loading fails.

## Evaluation Gates

`npm run sim:mind:evaluate -- --artifact <path> --enable-mind` compares the
heuristic and learned policy over summary-only seeds with trajectory recording
enabled. The Mind v1 gate report checks:

- schema compatibility through the loaded artifact manifest;
- invalid-action rate from trajectory summaries;
- survival through viable-run share;
- reproduction through births per run;
- resource pressure through plant energy available per land tile.
