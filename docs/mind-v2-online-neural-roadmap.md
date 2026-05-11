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

## Mind v3 Boundary

Mind v3 is now the primary autonomous-controller direction, not another guarded
artifact tweak. V2 remains the guarded offline artifact baseline for continuity
and comparison. V3 removes heuristic action selection from the agent runtime and
measures survival, births, deaths, inherited controller state, and action-source
counts directly. Do not use v2 fallback-rate micro-optimizations as the main
path for v3.

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

The current strongest guarded IQL candidate adds the opt-in behavior-margin
actor anchor plus behavior-anchored calibrated IQL actor extraction. The
extraction weights use standardized advantages only after clipping,
normalizing, and blending back toward behavior support. With the learned seed
`43` suppression diagnostic trajectory mixed into training and
`neural_actor_prior_blend_weight=0.75`, the default gate passes as a
`strict_control_candidate`: hard guard `0.0568`, delegation `0.4172`, total
fallback `0.4740`, safe-deviation `0.0`, and zero min per-seed alive/birth
deltas. The extended matrix also passes: hard guard `0.0582`, delegation
`0.4118`, total fallback `0.4700`, and zero min per-seed alive/birth deltas.
This supersedes the earlier behavior-margin blend-only near miss for guarded
promotion review. It does not supersede the autonomy roadmap:
`autonomous-online` still fails with alive/birth deltas `-31.25`/`-25.5`, and
held-out Q/V scale remains poorly calibrated.

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

Current guard-aware replay slice: `sim:trajectory` can now persist scalar
learned-policy decision diagnostics with `--include-policy-diagnostics`, and
`torch-discrete-iql` reads those diagnostics through the transition adapter.
`mind_gate` also accepts repeated `--extra-trajectory` inputs so learned rollout
replay probes still write the experiment ledger. Suppressed learned actions from
hard guards and confidence delegates become a masked actor margin penalty, with
hard-guard feedback weighted above delegate feedback. Unpromoted learned
self-actions are downweighted in replay so the trainer uses learned rollouts
mainly as safety feedback until they earn promotion. A stricter `0.221`
confidence-delegate margin is recorded as a rejected sweep: it reduced
delegation, but hard guard exceeded the strict cap. The retained replay path
therefore keeps the safer IQL `0.25` margin while actor safety improves. This is
the first executable bridge from frozen learned rollouts to offline-to-online
replay mixing; in-run weights remain immutable until the diagnostic rollout loop
beats the strict fallback targets.

The conservative critic/actor-anchor slice is implemented as an experiment
harness, but not active by default. `torch-discrete-iql` now records a CQL-style
masked legal logsumexp selected-action gap and a replay-weighted behavior
cross-entropy anchor in training metrics. Active CQL variants reduced hard guard
but consistently increased confidence delegation, and the behavior-anchor-only
default probe still missed the strict total fallback cap (`0.4794` versus
`0.4779`). The retained trainer therefore leaves both loss weights at `0.0`.
The next promotion-oriented work is not more CQL pressure; it is calibration
that can reduce delegation without simply moving those decisions into hard
guard.

The first in-run autonomy surface is now deliberately small:
`autonomous-online` runs the loaded artifact without heuristic fallback and
updates bounded context/action score offsets from finalized reward records. This
is a contextual bandit adapter for observation, learned-rollout collection, and
viewer-visible behavior probes. It is not the final online learner: neural
weights remain immutable during the run, and promotion still requires the
default and extended gates to beat the strict hard-guard and total-fallback
controls.

The gateable form of that surface now serializes
`mind_policy_update_trace_v1`. This is deliberately aligned with the recent
offline-to-online RL lesson from Cal-QL-style calibration, CQL/IQL-style
conservatism, and constraint-conditioned IQL variants: make every online update
auditable and replayable before allowing policy weights to change in the live
sim. The contextual adapter is therefore a measured bridge, not the final RL
algorithm.

The first default held-out autonomous-online comparisons are intentionally
negative. The contextual-prior autonomous-online run reached zero fallback but
failed with alive delta `-28.5` and births delta `-18.75`. The guarded IQL strict
candidate still passes at hard guard `0.1041`, delegation `0.3707`, and total
fallback `0.4748`, but its autonomous-online runtime comparison also fails with
alive delta `-30.25` and births delta `-21.5` despite `10501` replayable online
updates. Treat this as a gateable rejection of the contextual adapter, not a
reason to relax the safety floor.

The first offline-to-online replay-mixed probe is also implemented and rejected.
`torch-discrete-iql` can now consume replayed `mind_policy_update_trace_v1`
records as a signed actor-margin loss, which is the first real bridge from
online adaptation traces back into the offline actor/critic trainer. The default
probe mixed two autonomous-online traces into the reused gate bank and reached
hard guard `0.1405`, delegation `0.3324`, total fallback `0.4729`, and zero
guarded alive/birth deltas. It is not promotable because hard guard exceeds the
control cap and heuristic-free `autonomous`/`autonomous-online` comparisons
still collapse populations. The next online learner must learn a viability or
constraint-conditioned value signal, not merely imitate signed contextual
updates.

Viability critic v0 is now a diagnostics-only surface. Neural artifact
diagnostics label existing transitions for short-horizon survival risk,
death/near-death floor pressure, invalid actions, reproduction viability, and
heuristic suppression, then report Brier/AUC plus per-action and per-role/mode
breakdowns for the current neural value-risk proxy or the serialized
`multi_component_constraint_viability_head_v1` state head or
`action_conditioned_multi_component_constraint_viability_head_v1` component
head. The strict IQL candidate's held-out proxy baseline confirms the gap:
overall constraint risk `0.1143`, risk AUC `0.6940`, and `stay` risk `0.7361`
with only `0.0487` mean predicted risk. The first trained state head improves
ranking to `0.7697` AUC and moves `stay` mean score to `0.4320`, but worsens
Brier to `0.1554`. The action-conditioned head is stronger on the same held-out
bank (`0.9496` AUC, `0.0572` Brier, `0.9515` `stay` mean score), but the
separate calibration bank is weaker (`0.8709` AUC, `0.1106` Brier). The
heuristic-collected diagnostic bank has no positive hard-guard/delegate
suppression labels; the first learned-policy diagnostic probe adds `1464`
suppression positives but aggregate calibration remains weak (`0.6757` AUC,
`0.3323` Brier). Suppression-routed replay now trains those positives on the
suppressed learned action instead of the resolved heuristic fallback and improves
the suppression component to `0.6425` AUC and `0.2924` Brier, but strict policy
eval still rejects the candidate (`0.4800` total fallback and failed
autonomous-online outcomes). Learned-only suppression supervision plus observed
diagnostic masks moved guarded fallback closer (`0.4784` total) but left the
suppression head overconfident. The observed-mask positive-weight fix then
changes learned suppression weighting from the global `8.0` clamp to `1.0576`,
disables state-head suppression training, and improves held-out/calibration
Brier/AUC to `0.0555`/`0.9426` and `0.1036`/`0.8542`. It is still rejected as a
controller-quality step because guarded fallback worsens to `0.4825` total and
autonomous-online alive/birth deltas remain negative (`-28.25`/`-23.5`). This
signal is still diagnostics-only.

### 3. Offline-To-Online Fine-Tuning

Once neural offline gates pass, add online fine-tuning as a separate batch job:

- collect learned-policy summary-only trajectories;
- collect them with `--include-policy-diagnostics` so suppressed learned
  actions become guard-aware actor feedback;
- collect `autonomous-online` probes with `--include-policy-update-trace` so the
  in-run adaptation stream can be replayed;
- mix heuristic, prior learned, and current learned rollouts in replay;
- train with conservative replay weighting;
- consume `mind_policy_update_trace_v1` only as a replayable feedback signal,
  not as a live neural-weight mutation inside the tick loop;
- evaluate on held-out seeds never used for the online collection batch.

PPO/SAC-style online updates are candidates here, but they should run outside
the simulation tick loop. The promoted artifact remains immutable during a run.
The current in-run contextual adapter is allowed only as an experiment surface
even though its update log is serialized and replayable. It has now failed the
default held-out autonomous-online comparison, so the next online work must use
offline-to-online replay mixing and critic-calibrated policy improvement rather
than larger bounded score offsets.

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
10. Add gateable autonomous-online runtime comparisons and replayable online
    update traces. Status: implemented for `mind_policy_update_trace_v1`; the
    contextual adapter is rejected on default held-out outcomes.
11. Replace the contextual adapter with real offline-to-online replay mixing only
    after the neural offline gate passes and the replayed update dataset can train
    a calibrated, constraint-aware actor/critic outside the sim tick loop.
    Status: the first signed update-trace actor loss is implemented and rejected;
    viability critic v0 diagnostics, a state component head, an
    action-conditioned component head, separate calibration reporting, and a
    learned-policy suppression probe are implemented. Suppression-aware
    action-head routing is also implemented and rejected as a promotion
    candidate because total fallback remains above strict control. Learned-only
    suppression supervision and observed-component diagnostics move guarded
    total fallback closer to the strict target (`0.4784` versus `0.4780`) but
    still leave the suppression score overconfident on learned-policy probes.
    Detaching
    viability heads from the shared actor/Q/V representation is also rejected
    as a default because it worsened held-out/calibration/suppression Brier and
    guarded total fallback; it remains available only as the
    `--torch-iql-detach-viability-heads` experiment flag. Observed-mask
    action-viability positive weights are now the default correctness fix
    (`1.0576` suppression weight on the seed `43` replay mix), but the measured
    policy still misses fallback (`0.4825` total). A follow-up opt-in
    viability-safe behavior-margin actor anchor plus torch-IQL neural-prior
    blend sweep moved the best guarded total fallback to `0.4783` at blend
    `0.75` with hard guard `0.0576` and zero guarded alive/birth deltas, but it
    still misses the strict `0.4780` target and autonomous-online remains
    rejected. The next step is actor/Q training that improves confidence
    calibration directly instead of continuing runtime blend sweeps. The first
    discounted-return Q/V auxiliary loss and observed viability-risk actor
    extraction probes are implemented and rejected for promotion
    (`0.4777` and `0.4767` total fallback respectively), so the next learner
    slice should improve action-conditioned constraint calibration and critic
    ranking rather than merely downweighting risky logged labels. A first
    detached Q-minus-action-risk actor distillation probe improved guarded total
    fallback to `0.4759`, but it still missed the guarded control and worsened
    autonomous-online outcomes. Treat raw action-risk actor targets as
    diagnostic until risk calibration and action support are stronger. Calibrated
    Supported Actor Extraction v1 now fits action/component risk calibration on a
    separate calibration bank and reports support/rejection reasons, but its
    first default gate is rejected: `0.1047` hard guard, `0.3712` delegation,
    `0.4759` total fallback, zero guarded outcome deltas, and autonomous-online
    deltas `-34.5` alive / `-28.75` births. The next promotion-relevant actor
    work should constrain extraction by contextual support or behavior proximity
    so the critic cannot collapse movement/stay targets into mostly `eat` and
    `drink`. Contextual Behavior-Proximity Supported Extraction v2 adds a
    calibration-validation bank, context/action support, sparse-context-only
    fallback, target expansion caps, family conversion caps, and conversion
    diagnostics. After the final cap-enforcement fix, the default gate has
    total fallback `0.4717`, target/logged TVD `0.1800`, and zero guarded
    outcome deltas, but hard guard remains `0.1167` versus the current
    promotion-review control `0.0568`; autonomous-online fails worse at `-37.0`
    alive / `-31.5` births. The next promotion-relevant actor slice should use
    contextual KL/behavior-prior regularization or another trained
    behavior-proximity objective instead of post-hoc extraction caps.
12. Contextual behavior-prior actor regularization is implemented as the first
    trained behavior-proximity objective. The prior-only arm is the useful
    result: `0.0663` hard guard, `0.4037` delegation, `0.4700` total fallback,
    zero guarded outcome deltas, and autonomous-online deltas `-35.0` alive /
    `-28.5` births. It is still rejected against the current `0.0568` hard-guard
    control. Combining the prior with v2 target extraction is worse
    (`0.1031` / `0.3696` / `0.4727`) and should not be stacked by default.
    Finetune carryovers for suppressed-action guard feedback and
    viability-safe behavior margins are now implemented. With those carryovers,
    the non-risk prior-only path beats total fallback but still misses hard
    guard (`0.0582` / `0.4155` / `0.4737`). The closest current actor-objective
    candidate adds detached Q-minus-risk actor distillation and reaches
    `0.0567` hard guard, `0.4179` delegation, `0.4746` total fallback, and zero
    guarded outcome deltas. This clears hard guard but misses total fallback by
    `0.0006`; autonomous-online still fails (`-38.5` alive / `-32.25` births).
    Constraint stacking, stronger finetune margin weighting, and hard-guard-only
    carryover were tested and rejected. A follow-up delegate-margin actor
    finetune term (`0.0577` / `0.4187` / `0.4764`) and calibrated-control
    runtime-feedback finetune (`0.0589` / `0.4203` / `0.4792`) were also
    rejected. A value-supported deviation probe retained a neural-prior
    score-source fix, but strict thresholds were a no-op and a relaxed
    predicted-advantage floor regressed guarded outcomes. A suppression-aware
    critic calibration hook now trains Q/V margins directly on
    runtime-suppressed learned-action rows, but both full follow-up gates are
    rejected: the risk-adjusted contextual-prior stack worsens to
    `0.0576` / `0.4218` / `0.4794`, and the non-risk contextual-prior stack
    worsens to `0.0588` / `0.4195` / `0.4783`. Keep that hook opt-in and
    diagnostic-only. A root audit then separated the guarded promotion boundary
    from true autonomous capability: the current control is near the guarded
    frontier, but autonomous and autonomous-online still fail by `-34.25` /
    `-28.25` and `-31.25` / `-25.5` alive/birth deltas. Actor
    action-distribution regularization was implemented to test the obvious
    over-`eat` collapse. The soft marginal KL arm failed at `0.0591` /
    `0.4156` / `0.4747`; the sharpened argmax-proxy KL arm was worse at
    `0.0590` / `0.4251` / `0.4841`. Both retain zero guarded outcome deltas but
    fail autonomous-online. Keep the action-distribution hook opt-in and
    diagnostic-only; next work should target closed-loop survival credit or
    context-local sequence/navigation supervision rather than another global
    distribution penalty, safe-deviation relaxation, or extraction cap.
13. Mind v3 autonomous evolution has started as the main no-heuristic track.
    The first search slice evaluates inherited controller templates with
    `sim:mind:v3:evolve`, reuses the best template through evaluator,
    trajectory, and headless runtime flags, and preserves zero heuristic action
    sources. The initial two-generation smoke improves raw v3 from `0.5` alive
    mean / `0.0` births mean to `4.0` alive mean / `0.5` births mean at `80`
    ticks on seeds `5,13`, but remains far below the heuristic baseline
    (`28.0` / `18.0`). Continue with scaled selection pressure,
    quality-diversity, or curriculum-style survival pressure before returning
    to any v2 actor-objective micro-optimization.
14. The first quality-diversity v3 search milestone is implemented. The search
    report is resumable, carries archive elites for balanced/survival/births/
    resource/movement/low-death behavior, writes holdout evaluation, and uses a
    survival-dominant score after a real run showed resource loops could
    otherwise outrank sustained survival. At `120` ticks the current controller
    still collapses (`0.6` alive mean, `0.2` births mean across seeds
    `5,13,19,29,37`). At `80` ticks, curriculum search improves raw v3 from
    `0.8` / `0.0` alive/births to `3.8` / `0.2`, with zero heuristic action
    sources. Next v3 work should make the curriculum explicit and only extend
    horizons after holdout survival clears, then address reproduction directly.
15. Holdout-gated v3 curriculum search is implemented. The first bounded run
    (`output/mind/mind-v3-curriculum-80-120-search.json`) passed the `80`-tick
    holdout floor (`2.5` alive mean, `0.0` births, zero heuristic actions) but
    failed the `120`-tick holdout floor (`0.5` alive mean, `0.0` births, zero
    heuristic actions). The selected last-passing template evaluates at `3.4`
    alive mean / `0.2` births at `80` ticks and `0.4` / `0.2` at `120` ticks
    across seeds `5,13,19,29,37`, so it does not beat the previous best
    `80`-tick searched v3 result (`3.8` / `0.2`). The next v3 work should
    improve closed-loop survival credit or controller capacity before scaling
    this curriculum further.
16. V3 search now carries runtime lineage elites and behavior descriptors
    forward. This aligns the search with the actual product goal: multiple
    durable ecological niches, not one globally dominant species. Treat terrain
    occupancy, trophic role, meat mode, movement/resource action mix, lineages,
    and runtime species/ecotype counts as first-class diagnostics for future v3
    work. The first lineage-selection curriculum rerun improved the `80`-tick
    training best but failed holdout immediately (`0.5` alive, `0.0` births,
    zero movement, resource rate `1.0`), so the archive now carries bounded
    behavior-niche cells keyed by dominant terrain, trophic role, and meat mode
    and can use those niche-specific lineage templates as future parents. That
    restores the `80`-tick holdout pass (`3.5` alive, `0.0` births) but still
    fails the `120`-tick stage (`0.0` alive, `0.0` births), so the next real
    bottleneck is controller credit assignment/capacity, not archive plumbing.
    The curriculum gate now has optional holdout movement, action-diversity,
    and behavior-niche floors. With floors `0.005` movement, `2` unique
    actions, and `4` behavior niches, the `80`-tick stage passes with `11`
    niches, while the `120`-tick stage still collapses to `0.0` alive,
    `0.0` movement, and `1.0` unique action.
    The report contract now surfaces requested/resolved action histograms,
    dominant requested-action share, and active behavior niches. Active niches
    require live, moving, or reproducing lineage evidence, so static
    terrain/trophic/meat variety can no longer mask a one-action holdout loop.
    Curriculum stages can reject dominant-action collapse via
    `--curriculum-max-holdout-dominant-action-share` and require live behavior
    diversity via `--curriculum-min-holdout-active-behavior-niches`.
    The score policy is now `survival_dominant_quality_diversity_v4`, with a
    bounded active-niche bonus and bounded dominant-action collapse penalty.
    The strict v4 rerun still passes only the `80`-tick stage (`3.5` alive,
    `0.5689` dominant action share, `8` active niches) and fails `120` ticks
    (`0.0` alive, `1.0` dominant action share, `0` active niches). Five-seed
    eval remains `3.6` / `0.4` at `80` and `0.2` / `0.4` at `120`, so scoring
    pressure alone is not the bottleneck fix.
17. V3 controller capacity now uses
    `homeostatic_feature_projection_linear_action_head_v2` for new founders
    instead of fixed pseudo-random projection. The action head is still bounded,
    inherited, and heuristic-free, but the hidden features expose
    policy-visible survival and reproduction context: energy need, hydration
    need, injury/risk pressure, reproduction drive, resource context, water
    context, movement pressure, and trophic/mode/signal context. Founder
    initialization also records `diverse_homeostatic_action_prior_v1` with a
    bounded specialization profile, so search starts with multiple ecological
    biases rather than one uniform random controller family. The measured
    `80,120` curriculum run
    `output/mind/audit/mind-v3-homeostatic-curriculum-80-120-search.json`
    completed both stages. The selected template evaluates across
    `5,13,19,29,37` at `6.0` alive / `0.4` births for `80` ticks and `1.0` /
    `0.4` for `120` ticks, still far below the heuristic baseline but materially
    better than the prior `120`-tick collapse.
    A visibility hardening pass then removed `mind_inheritance_available` from
    the v2 homeostatic projection and added scorer tests proving patch,
    navigation, and controller-state availability changes do not affect this
    controller's self-state feature scores. The rerun after that fix reached
    `6.2` alive / `0.4` births for `80` ticks and `0.8` / `0.4` for `120`
    ticks, so the next blocker remains credit assignment and reproduction
    pressure rather than representation visibility.
    The next audit pass also fixed a v3 online-credit bug: valid requested
    actions now receive reward credit even if live resolution races convert the
    outcome to `stay`; invalid requested actions still credit the resolved
    action. The selected-template rerun after this fix held the same measured
    outcome: `6.2` / `0.4` at `80` ticks and `0.8` / `0.4` at `120` ticks with
    zero heuristic action-source counts.
18. GPU/CUDA support is now a clean infrastructure-only path, not a learner
    behavior change. `mind_train` and `mind_gate` accept
    `--torch-device cpu|cuda|mps|auto` defaulting to `cpu`; torch artifacts
    record requested/resolved device plus CUDA/MPS visibility metadata and
    still serialize CPU JSON weights. Local smoke on May 9, 2026 resolved
    `auto` to MPS on this macOS ARM runner; CUDA remains unavailable here and
    belongs on the RTX host. V3 curriculum search separately has
    `--rollout-workers` for CPU process-level rollout
    parallelism because that bottleneck is simulator-rollout bound, not tensor
    kernel bound. `mind_gate` now also reports phase timings and has
    `--artifact-diagnostics-workers` for artifact diagnostics plus
    `--evaluation-workers` for independent held-out/runtime-mode evaluation
    matrix entries; use these before moving one-off local gates to the RTX
    host. The first real local IQL gate with both worker pools improved wall
    time from `1099.6261s` to `812.7058s`, and the phase timing map showed
    artifact diagnostics still dominated (`660.8851s`) over training
    (`37.4958s`) and evaluation (`111.5268s`). The sharded artifact-diagnostics
    reducer now splits work by trajectory dataset and reduced the same local
    profile to `301.8127s`, with diagnostics at `151.8482s`. CUDA should wait
    for heavier tensor-training sweeps; this local gate path is still mostly
    Python diagnostics and simulator evaluation.
19. May 10 audit hardening fixed an animal-resource opportunity timing bug:
    reachability and resource presence are now recorded from the same
    decision-time snapshot, so resources created later in the tick no longer
    inflate the policy-visible opportunity denominator. Same-tick generated
    resources still count as consumption ticks when they are consumed, so
    consumption observability is not coupled to opportunity visibility. New v3
    founders also moved from self-only homeostatic features to
    `local_navigation_feature_projection_linear_action_head_v3`, which uses
    only policy-visible local patch and navigation tensor fields. A later May
    11 slice extends this again to
    `need_gated_local_navigation_feature_projection_linear_action_head_v4`,
    adding thirst/hunger-gated navigation interactions without widening FOV.
    This fixes a representation blind spot but does not by itself prove horizon
    transfer.
20. The next v3 hardening slice fixed a founder-template collapse in the
    evolved-artifact path. Search founders are now stratified across
    specialization priors, evolved reports can carry an eligible founder
    template pool, and evaluator/runtime paths consume that pool instead of
    cloning all founders from one selected controller. The current best local
    smoke keeps three viable profiles (`hydration_seeker`, `disperser`,
    `reproducer`) and lowers five-seed `80`-tick dominant-action share to
    `0.5887`, but births remain `0.0` and `120`-tick survival is only `0.6`.
    Next work should move to credit assignment and reproduction pressure after
    preserving this diversity contract.
21. Reproduction-pressure scoring now has a held-out gate. The v6 scoring
    policy strengthens terminal energy, health, and matched-diet viability and
    penalizes alive candidates with zero terminal reproduction viability, plus
    one-seed births and one-seed terminal viability. The curriculum gate now has
    optional held-out floors for those terminal viability shares plus
    biologically reproduction-ready mean. A dense reward-shaping probe produced
    one-seed in-sample births but failed holdout, so the runtime reward change
    was not retained. The retained v6 smoke
    `output/mind/mind-v3-robust-repro-score-v6-80-smoke.json` lowers held-out
    dominant action share to `0.3636`, but terminal energy viability remains
    `0.0`, births remain `0.0`, and five-seed `120`-tick survival is only
    `0.2`. Use nonzero held-out viability floors before claiming reproduction
    progress.
22. V3 evaluation now has a real controlled-ecology diagnostic path. The
    evaluator reports terminal reproduction failure attribution from the same
    `reproduction_end` summary used by runtime gates, and `--fixture-suite
    basic` runs herbivore/hunter/scavenger/omnivore founders through plant-only,
    carrion-only, prey-rich, and mixed-stable arenas with real policy,
    diet/combat/resource, and reproduction mechanics. The 80-tick smoke
    `output/mind/mind-v3-controlled-fixture-v1-80-smoke.json` shows this is not
    controller progress: current v3 loses open-world survival/births (`3` /
    `0` versus heuristic `37` / `19`) and fixture births, with terminal
    viability failures concentrated in energy, hydration, matched diet, and
    health depending on the arena. Keep this suite as a regression/falsification
    surface for the next credit-assignment/reproduction-pressure work.
23. The controlled fixture suite is now an explicit v3 gate in evaluation and
    search, and curriculum search stops a stage when the enabled fixture gate
    fails. The first reward-signal change after this gate uses observed
    post-action reproduction-readiness deltas rather than adding new
    decision-time features: `outcome_delta_action_conditioned_readiness_signal_v3`
    credits energy/hydration/health movement toward readiness, terminal
    reproduction/death outcomes, and action-conditioned outcomes. Repeated
    no-gain `eat` is penalized; `eat`, `drink`, and movement get extra credit
    only when the finalized transition improves readiness-relevant state. New
    80-tick search artifact
    `output/mind/mind-v3-repro-delta-fixture-gated-80-search.json` improved to
    `8.5` held-out alive / `1.0` held-out births, with five-seed eval at `9.0`
    alive / `1.4` births for `80` and `4.4` alive / `1.8` births for `120`.
    This is real progress over the prior v3 horizon-transfer collapse, but not
    promotion: `carrion_only` still fails fixture alive floor, no terminal
    biologically-ready population is sustained, and action collapse remains
    biased toward `eat`.
24. Search now has fixture-aware top-K reranking through
    `--fixture-rerank-top-k`. Artifact
    `output/mind/mind-v3-fixture-rerank-action-credit-80-search.json` selected
    a fixture-passing candidate over a higher raw-score candidate with a
    fixture blocker. Five-seed eval reached `9.8` alive / `3.4` births at `80`
    and `4.8` alive / `4.2` births at `120`; two-seed fixture eval passed the
    configured fixture gate. This is the first v3 result in this sequence with
    multi-birth transfer at both `80` and `120`, but it is still below the
    heuristic and still lacks sustained terminal biological readiness.
25. V3 search now includes hydration in terminal reproduction viability and
    records sustained core readiness from observed post-action
    energy/hydration/health state. The reranker can also evaluate composite
    founder-template repair candidates, combining a high-holdout nominee with
    a real archived fixture-passing donor pool. Corrected artifact
    `output/mind/mind-v3-composite-repair-v8-top8-fixture2-80-search.json`
    selected `g2-c2+repair-g0-c4`, passed the two-seed fixture gate, and
    improved five-seed `120`-tick eval to `7.6` alive / `5.0` births. The
    `80`-tick eval regressed to `9.0` alive / `2.2` births, so this is a
    horizon-transfer milestone rather than a promotion.
26. Real-energy readiness is now an explicit search/evaluation metric instead
    of a hidden terminal blocker. The evaluator reports
    `terminal_reproduction_failure_attribution_v2` and surfaces terminal
    energy-requirement satisfaction, energy shortfall, and terminal viability
    shares at aggregate level. The v4 online reward signal raises the
    observable energy target and penalizes no-gain drinking without adding new
    decision-time features. Artifact
    `output/mind/mind-v3-real-energy-credit-v10-top8-fixture2-80-search.json`
    selected `g0-c1`; five-seed eval improved to `10.8` alive / `3.8` births
    at `80` and `8.8` / `6.8` at `120`. Energy viability improved, but
    hydration viability fell and terminal biologically-ready agents remain
    `0.0`; the next learner target is balancing energy and hydration readiness
    rather than more search plumbing.
27. Balanced bottleneck credit is now wired through v3 search/evaluation as
    `balanced_bottleneck_hydration_guard_readiness_signal_v6` with
    `balanced_bottleneck_anti_collapse_quality_diversity_v12` scoring. It adds
    terminal balanced readiness, energy/hydration balance, energy/hydration
    gap, and temporal blocker attribution, all from observed before/after
    self-state rather than extra decision-time perception. v12 improved the
    five-seed `120` open-world result to `14.0` alive / `8.6` births, but it is
    not a promotion: terminal biologically-ready agents remain `0.0`,
    hydration regressed versus v10, and both top-8 and top-16 rerank searches
    failed the `carrion_only` fixture lane. The next controller target is an
    explicit carrion/scavenger quality-diversity lane with fixture-preserving
    archive selection.
28. v13 adds that explicit carrion/scavenger lane: a distinct `scavenger`
    founder prior, `quality_diversity_archive_v2` with a `scavenger_lane`
    elite, top-K plus lane fixture rerank, and scavenger-lane composite repair
    when no fixture-passing donor exists. Artifact
    `output/mind/mind-v3-scavenger-lane-v13-top8-fixture2-80-search.json`
    selected `g0-c7+repair-g0-c4`. The hard fixture gate still fails, but
    `carrion_only` now has real animal-resource use (`14` events over two
    fixture seeds), `1.0` alive, `2.5` births, hydration viability `0.5`, and
    matched-diet viability `0.5`; the remaining hard blocker is energy
    viability.
29. v14 parallelizes fixture rerank across CPU candidate evaluations and adds
    observed animal-resource intake to scoring and fixture summaries. The
    controller reward is
    `balanced_bottleneck_observed_carrion_readiness_signal_v7`, which credits
    carcass/fresh-kill `eat` only from observed action outcomes and readiness
    deltas, and penalizes zero-delta `eat`. Artifact
    `output/mind/mind-v3-carrion-rerank-v14-top8-fixture2-80-search.json`
    passes the hard `80`-tick fixture gate with `carrion_only` energy viability
    `0.5` and dominant `eat` around `0.47` in that fixture. The tradeoff is
    explicit: five-seed open-world eval is lower than v13 (`14.0` / `6.0` at
    `80`, `14.6` / `11.4` at `120`), and the `120`-tick fixture gate still
    fails on `carrion_only` horizon robustness. Next work should search on
    multi-horizon fixtures or add survival/reproduction credit that preserves
    carrion readiness past the initial 80-tick window.
30. v16 adds multi-horizon fixture rerank through `--fixture-rerank-ticks`.
    Top-K candidates can now be evaluated on both `80` and `120` controlled
    fixture horizons during selection, with combined blockers and per-horizon
    summaries in the report. A first v15 run exposed a tempting open-world
    challenger, `g2-c3+repair-g0-c4`, with five-seed `120` eval at `17.4`
    alive / `14.0` births, but it regressed the 80-tick carrion fixture energy
    gate. The retained v16 selector therefore prefers partial horizon pass
    coverage before smaller blocker count when all candidates fail the combined
    gate. Corrected artifact
    `output/mind/mind-v3-multihorizon-rerank-v16-top8-80-120-search.json`
    preserves the v14 selected `g1-c0` and records the real blocker:
    `carrion_only` still fails the `120` horizon on energy, hydration, and
    matched-diet viability. The next controller work should combine the v15
    open-world improvement with hard carrion readiness preservation rather
    than choosing one side of that tradeoff.
31. v18 tests that combination as a bounded bridge repair instead of changing
    the gate. The reranker now creates promotion-safe bridge candidates from
    an 80-horizon-passing primary plus an unsafe long-horizon donor, with
    donor-template limits `1`, `2`, and `4` to avoid swamping the safe primary.
    Artifact
    `output/mind/mind-v3-bridge-lite-rerank-v18-top8-80-120-search.json`
    evaluated four such bridge candidates. All failed the combined fixture
    gate, and even the smallest donor injection collapsed carrion-only
    readiness at the limiting horizon. The selected candidate remains `g1-c0`,
    with v18 `120` eval at `14.6` alive / `11.4` births. This is a useful
    negative result: simple founder-pool composition cannot transfer the v15
    open-world improvement safely.
32. v19 fixes a hidden founder-pool assignment confounder. Template pools are
    no longer assigned by `agent_id % pool_size`; Mind v3 now records and uses
    `contextual_trophic_founder_template_assignment_v1`, selecting templates
    from policy-visible trophic role and meat mode self context. The prior v18
    artifact improves under this assignment (`18.0` alive / `12.6` births at
    `120`), but a fresh v19 scalar search regresses on five-seed eval and still
    fails carrion fixtures. This confirms the method is below the standard
    needed for open-ended ecological diversity: the next slice should preserve
    vetted fixture-safe artifacts through warm-start/fixture-archive search
    before returning to larger controller or IQL sweeps.
33. v20/v21 adds fixture-archive warm-starting and longer delayed credit.
    Warm-start candidate import is now round-robin across supplied reports,
    and fixture rerank includes warm-start nominees even when scalar score
    would exclude them. This closed a hidden false-progress path where only the
    first source archive was actually sampled. The controller credit assignment
    is now `policy_valid_requested_action_horizon_eligibility_trace_v3`
    (`12` steps, `0.84` decay), which keeps credit for longer
    movement-to-water/resource paths without expanding policy FOV. Artifact
    `output/mind/mind-v3-delayed-credit-v21-top8-80-120-search.json`
    improved five-seed `120` eval to `15.8` alive / `12.2` births and reduced
    dominant `eat` to `0.4388`, but the multi-horizon carrion fixture still
    fails, especially hydration. The rejected v22 hydration-risk eat penalty
    regressed both broad search and carrion viability, so the next learner
    target should be fixture-in-loop parent selection or an architecture that
    can sequence carrion intake followed by water return.
34. v23 adds need-gated navigation capacity and honest pool diagnostics. The
    Mind v3 contract now declares `policy_visible_self_local_patch_navigation`
    instead of the stale self-only scope, keeps private mind-inheritance
    availability excluded, and lists derived interaction features. Founder pool
    reports record template-pool fingerprints so top-K fixture rerank can no
    longer hide that multiple nominees are evaluating through identical
    composite pools. This is a capacity and observability slice; promotion still
    depends on measured search/eval artifacts. The first v23 artifact improved
    broad five-seed births over v21 (`7.0` at `80`, `15.2` at `120`) but still
    failed the combined carrion fixture, so the next task is fixture-in-loop
    parent selection/search pressure.
35. Add quality-diversity viewer diagnostics once policy lineages
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
