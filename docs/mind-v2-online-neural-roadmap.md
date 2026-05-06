# Mind v2 Online Neural Roadmap

## Purpose

The current Mind is not a neural RL agent. It is a deterministic, guarded
offline learner with a value head and an opt-in learned-policy runtime path.
This document defines the path from that safe baseline to neural policies,
online learning, and open-ended emergent behavior without silently weakening
Foundation gates.

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

The executable slice is now:

```bash
npm run sim:trajectory -- \
  --seed 41 \
  --ticks 120 \
  --output output/trajectories/mind-v2-learned-seed41.jsonl.gz \
  --split-id mind-v2-learned-rollout \
  --mind-artifact output/mind/mind-v1-gate-value-deviation-artifact.json \
  --enable-mind
```

This collects learned-policy experience without full replay surfaces and keeps
the same explicit `--enable-mind` safety gate used by full replay and policy
evaluation.

## Algorithm Stack

There is no single SOTA algorithm for this simulator. The problem mixes
partially observed multi-agent ecology, delayed population effects, sparse
reproduction outcomes, local resource externalities, and non-stationarity from
evolution. The implementation should use a staged stack instead of betting on
one method.

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
3. Add a small neural BC actor-critic trainer and loader behind new trainer and
   model type names.
4. Add artifact diagnostics for calibration, value error, held-out action drift,
   and fallback deltas.
5. Run strict gates against the neural artifact without relaxing thresholds.
6. Add offline-to-online replay mixing only after the neural offline gate passes.
7. Add quality-diversity archives and viewer diagnostics once policy lineages
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
