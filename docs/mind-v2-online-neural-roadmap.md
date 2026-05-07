# Mind v2 Online Neural Roadmap

## Purpose

The current Mind is not a neural RL agent. It is a deterministic, guarded
offline learner with a value head and an opt-in learned-policy runtime path.
This document defines the path from that safe baseline to neural policies,
online learning, and open-ended emergent behavior without silently weakening
Foundation gates.

The current audit, package/research checkpoint, and durable experiment ledger
live in `docs/mind-v2-audit-and-experiment-log.md`. Check that ledger before
starting another trainer or calibration experiment so rejected paths are not
repeated.

## Offline Versus Online

Offline means training from fixed trajectory files. Online means collecting new
experience from a policy that is already acting in the simulator, then using
that experience to improve the next policy.

This repo should not start with in-tick online weight updates. Updating a model
inside a simulation run would make replay provenance, deterministic debugging,
and held-out gates harder to trust. The safe online boundary is therefore
offline-to-online:

1. Train an artifact from fixed trajectories.
2. Run that artifact in summary-only mode to collect learned-policy
   trajectories.
3. Retrain outside the simulation tick loop.
4. Promote only if strict held-out gates pass.

The executable slices are now:

```bash
npm run sim:trajectory -- \
  --seed 41 \
  --ticks 120 \
  --output output/trajectories/mind-v2-learned-seed41.jsonl.gz \
  --split-id mind-v2-learned-rollout \
  --mind-artifact output/mind/mind-v1-gate-value-deviation-artifact.json \
  --enable-mind
```

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer neural-actor-critic-bc \
  --output output/mind/seed7-neural-artifact.json
```

```bash
python3 -m pip install -r requirements-mind-ml.txt
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-actor-critic-bc \
  --output output/mind/seed7-torch-artifact.json
```

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-advantage-actor-critic-bc \
  --output output/mind/seed7-torch-advantage-artifact.json
```

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-discrete-iql \
  --output output/mind/seed7-torch-iql-artifact.json
```

The trajectory command collects learned-policy experience without full replay
surfaces and keeps the same explicit `--enable-mind` safety gate used by full
replay and policy evaluation. The neural trainer writes
`guarded_neural_actor_critic_bc_v1`: a deterministic pure-Python actor-critic
MLP artifact with frozen weights, explicit backend/architecture metadata, masked
action scores, state value, and action-value estimates. The torch trainer writes
`guarded_torch_actor_critic_bc_v1`: the same runtime artifact contract, but
trained with PyTorch/AdamW. The advantage trainer writes
`guarded_torch_advantage_actor_critic_bc_v1`, an opt-in conservative
advantage-weighted variant. The discrete IQL trainer writes
`guarded_torch_discrete_iql_v1`: an opt-in offline-RL artifact trained from
episode-aware transitions with Q, expectile V, and masked advantage-weighted
actor extraction. None of these neural artifacts are promoted yet.

## Algorithm Stack

There is no single SOTA algorithm for this simulator. The problem mixes
partially observed multi-agent ecology, delayed population effects, sparse
reproduction outcomes, local resource externalities, and non-stationarity from
evolution. The implementation should use a staged stack instead of betting on
one method.

Tooling checkpoint: `requirements-mind-ml.txt` is the optional training stack.
PyTorch is the first executable dependency because it gives direct control over
masked actor/value losses and serializable weights. TorchRL/TensorDict,
Gymnasium/PettingZoo, Minari, and d3rlpy are declared for the next environment,
multi-agent, dataset, and conservative offline-RL slices, but they should enter
runtime behavior only through explicit artifact contracts and gates.

Freshness checkpoint, May 6, 2026: `pip index versions` reported the installed
optional stack current for `torch`, `torchrl`, `tensordict`, `gymnasium`,
`pettingzoo`, and `minari`. `d3rlpy` has a newer `2.8.1` release, but that
release pins `gymnasium==1.0.0` and conflicts with the current Farama stack, so
the optional requirement keeps an explicit `d3rlpy>=2.8,<2.8.1` compatibility
hold until d3rlpy releases against newer Gymnasium.

```bash
for package in torch torchrl tensordict gymnasium pettingzoo minari d3rlpy; do
  .venv/bin/python -m pip index versions "$package"
done
```

Current literature checkpoint, May 2026: recent offline RL work keeps moving
toward constraint-aware and distribution-shift-aware variants of IQL-style
training. That matches this simulator better than a raw online PPO jump because
the main failure mode is not just action accuracy; it is unsafe extrapolation
under sparse, delayed population effects. Treat newer C2IQL/DiSA-IQL-style
ideas as candidate critic and replay-weighting designs, not as dependencies to
import wholesale. A candidate only enters the repo after it can be expressed in
the existing masked discrete action contract and compared against heuristic,
current learned, and held-out learned-policy rollouts.

### 1. Neural Actor-Critic Behavior Cloning

First neural target: a small deterministic MLP actor-critic over the existing
542-float observation vector, trained from the seed-bank trajectories.

Outputs:

- masked action logits for the existing discrete action vocabulary;
- state value estimate;
- action-value estimates for calibrated delegation;
- calibration metrics by action, trophic role, meat mode, score bucket, and
  held-out seed.

Why first: it gives us a real neural policy while preserving the current offline
contract and deterministic artifact promotion path. It also creates the value
head needed by later offline RL.

Current status: the schema, loader, learned-policy trajectory path, and four
trainers exist. `--trainer neural-actor-critic-bc` is a dependency-free
contract trainer. `--trainer torch-actor-critic-bc` uses the optional
`requirements-mind-ml.txt` stack and serializes the trained weights back into
the same dependency-free runtime artifact. The current PyTorch trainer uses a
256-unit actor-critic hidden layer, capped class-balanced actor loss, and
artifact diagnostics for train-set actor accuracy, confidence margin,
action-value absolute error, state-value absolute error, Python version, and
installed optional ML package versions. `--trainer
torch-advantage-actor-critic-bc` is the first conservative offline-RL bridge: it
keeps the same runtime artifact but trains the actor with conservative
contextual advantage weights blended back toward behavior cloning.
`--trainer torch-discrete-iql` is the first Bellman-style discrete offline-RL
trainer: it builds episode-aware transitions, trains selected-action Q targets,
fits an expectile V head, and extracts a masked actor with IQL-style advantage
weights.

On a two-seed probe (`5,13`, 120 ticks) trained from the existing 14-seed bank,
the plain PyTorch behavior-cloning trainer kept per-seed alive/birth deltas
non-negative and reduced total fallback to `0.4815`, just above the `0.4780`
control target. Hard guard was still `0.1418`, above the `0.1190` target.

The conservative advantage trainer also passed alive/birth safety on the same
probe (`alive +0.5`, `births +1.0`) and improved held-out neural value error
(`action_value` MAE `0.1009`, `state_value` MAE `0.1915`), but it worsened
fallback (`hard guard 0.1668`, `total fallback 0.5769`). It is not promoted.
The first held-out-calibration acceptance slice is now implemented for one
narrow action family. Neural artifacts declare score-margin and
predicted-advantage thresholds, and runtime can accept high-margin,
positive-advantage local `eat` actions under safe vitals and local-food
constraints. The clean two-seed probe passed with `13` such neural deviations
and total fallback `0.5718`, a small improvement over the advantage trainer but
still worse than the `0.4780` control target. It is not promoted. The next neural
work is to improve the policy/critic itself, not to widen heuristic bypasses.

### 2. Conservative Offline RL

Next target: conservative offline RL over the same replay bank. Use algorithms
that explicitly resist out-of-distribution actions:

- IQL-style advantage-weighted policy extraction for stable improvement from
  logged behavior.
- CQL-style pessimistic critics if value overestimation causes unsafe actions.
- TD3+BC/ReBRAC ideas where behavior-cloning regularization is needed to keep
  online fine-tuning stable.
- Constraint-conditioned or distribution-shift-aware IQL variants if gates show
  that plain value estimates take actions that look locally good but regress
  alive/birth outcomes.

For the current discrete action space, implement the ideas as masked
actor-critic losses rather than importing a continuous-control implementation
verbatim.

Current executable slice: `torch-discrete-iql` implements the first
transition-based discrete IQL-style trainer. The corrected default gate report
is `output/mind/mind-v2-torch-discrete-iql-report.json`. The IQL artifact now
uses mask-renormalized neural actor scores, a `0.9` contextual-prior actor
anchor, and an explicit IQL-only confidence delegate margin of `0.25`. On the
default held-out seed bank it is the first neural/RL strict-control candidate:
hard guard `0.1041`, delegation `0.3707`, total fallback `0.4748`,
safe-deviation `0.0`, and zero per-seed alive/birth deltas. Runtime
value-supported deviations are still disabled for IQL artifacts, so the
candidate earns the fallback improvement through actor ranking and abstention,
not broad heuristic bypasses.

The same candidate also passed the extended strict matrix across seeds
`5,13,19,29,37,41` at `120` and `180` ticks: hard guard `0.1054`, delegation
`0.3661`, total fallback `0.4715`, safe-deviation `0.0`, and zero per-seed
alive/birth deltas. This is promotion-review progress for the neural/RL path,
but it is still not a final autonomous controller: the policy is frozen during a
run and delegates more than a third of decisions to the heuristic.

The next confidence-calibration probe renormalized neural actor scores over the
current legal action mask before runtime delegation. This fixed the confidence
accounting but did not improve total fallback: delegation fell to `0.0062`,
hard guard rose to `0.5772`, total fallback remained `0.5834`, and the probe
failed the broad hard-guard cap. That is a useful diagnosis: the IQL actor is
confidently choosing actions the safety floor suppresses. The next
conservative offline-RL slice should improve actor ranking with guard-aware
training or contextual-prior anchoring before any runtime deviation path is
reconsidered.

The advantage-blended neural anchor was also tested as a candidate actor anchor.
It was rejected as the default IQL anchor: total fallback remained below control
at `0.4763`, but hard guard rose to `0.1459`. The default candidate therefore
uses the plain contextual actor anchor plus trainer-scoped delegate calibration.

### 3. Offline-To-Online Fine-Tuning

Once neural offline gates pass, add online fine-tuning as a separate batch job:

- collect learned-policy summary-only trajectories;
- mix heuristic, prior learned, and current learned rollouts in replay;
- train with conservative replay weighting;
- evaluate on held-out seeds never used for the online collection batch.

PPO/SAC-style online updates are candidates here, but they should run outside
the simulation tick loop. The promoted artifact remains immutable during a run.

### 4. Open-Ended Population Search

Emergent behavior should not be optimized only by mean reward. Add
quality-diversity archives after the neural policy is stable:

- behavior descriptors: trophic role mix, migration radius, reproduction timing,
  predation pressure, carrion reliance, habitat preference, lineage persistence;
- archive axes: survival quality plus diversity of ecological strategies;
- viewer surfaces: lineage strategy labels, archive niche, controller lineage,
  and behavioral novelty over time.

This is the path toward visible variety: agents should discover different viable
ecological strategies, not converge to one high-reward exploit.

### 5. World Models

Dreamer-style world models are attractive for this simulator because they learn
predictive dynamics and can imagine future rollouts. They are not the first
implementation step because the current contract still needs neural actor,
critic, replay mixing, and online promotion gates. The world-model milestone is
appropriate after the neural offline-to-online loop is stable.

## Promotion Gates

Every neural or online artifact must satisfy the current strict Mind promotion
criteria before it can replace more heuristic behavior:

- zero per-seed terminal alive regression;
- zero per-seed birth regression;
- hard guard below `0.1189`;
- total heuristic fallback below `0.4779`;
- default and extended held-out seed matrices remain green;
- learned-policy trajectory collection remains opt-in;
- no in-simulation weight updates unless a future contract version explicitly
  makes update logs replayable and gateable.

## Implementation Order

1. Keep learned-policy summary-only trajectory collection green.
2. Add a neural artifact schema extension with explicit backend, architecture,
   input normalization, action mask policy, and deterministic seed metadata.
   Status: implemented for `guarded_neural_actor_critic_bc_v1`.
3. Add a small neural BC actor-critic trainer and loader behind new trainer and
   model type names. Status: implemented for `neural-actor-critic-bc`.
4. Add a PyTorch-backed trainer that keeps runtime inference artifact-based.
   Status: implemented for `torch-actor-critic-bc`.
5. Add artifact diagnostics for calibration, value error, held-out action drift,
   and fallback deltas.
   Status: implemented for neural artifacts and Mind gates.
6. Add a transition-based discrete IQL trainer with Q, expectile V, and masked
   advantage-weighted actor extraction. Status: implemented for
   `torch-discrete-iql`; not promoted.
7. Renormalize neural confidence over legal action masks before delegation and
   diagnostics. Status: implemented; it exposed IQL actor safety/ranking
   failures and is not a promotion.
8. Add guard-aware actor ranking or contextual-prior anchoring before relaxing
   thresholds. Status: implemented for `torch-discrete-iql` as a contextual
   actor anchor plus IQL-only delegate calibration; default and extended strict
   gates now pass as a strict-control candidate.
9. Decide the promotion-review boundary for the strict-control IQL artifact and
   keep it opt-in until the review explicitly accepts the remaining heuristic
   delegation.
10. Add offline-to-online replay mixing only after the neural offline gate passes.
11. Add quality-diversity archives and viewer diagnostics once policy lineages
   are stable enough to compare.

## References

- PPO: https://arxiv.org/abs/1707.06347
- CQL: https://arxiv.org/abs/2006.04779
- IQL: https://arxiv.org/abs/2110.06169
- TD3+BC: https://arxiv.org/abs/2106.06860
- ReBRAC: https://arxiv.org/abs/2305.09836
- C2IQL: https://proceedings.mlr.press/v267/liu25ai.html
- DiSA-IQL: https://arxiv.org/abs/2510.00358
- DreamerV3: https://arxiv.org/abs/2301.04104
- MAP-Elites: https://arxiv.org/abs/1504.04909
- POET: https://arxiv.org/abs/1901.01753
- Open-ended generally capable agents: https://arxiv.org/abs/2107.12808
