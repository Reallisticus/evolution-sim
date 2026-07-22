# Mind v3 Frontier Research Audit

Date: 2026-07-21
Evidence update: 2026-07-22
Scope: controller-learning strategy, not implementation authorization
Research policy: primary papers, official proceedings, and first-party project sources only

## Executive verdict

The project has not yet produced a learned autonomous controller that clears its
survival, reproduction, carrion, action-diversity, and per-seed boundaries. It
has, however, produced enough negative evidence to identify a much narrower
problem than “reinforcement learning does not work here.”

The dominant failure is an experimental and architectural mismatch:

1. The target behavior is closed-loop, history-dependent, and multi-agent.
2. The latest training data are a small set of selected successful trajectories,
   not broad policy-induced experience with matched failures and counterfactuals.
3. The v195 “training” operation aggregates terminal support utility into exact
   public-feature-key/action tables. It does not fit a recurrent function
   approximator.
4. v195 constructs every training key with empty rollout history while runtime
   scores with live per-agent history, breaking feature parity on `96.08%` of
   the source rows.
5. The deployed candidate then encounters states induced by its own actions,
   where exact keys are mostly absent and present keys are action-incomplete.
6. Strict seeds have been repeatedly observed and some have supplied support
   provenance, so they cannot still serve as clean evidence of generalization.

The current v202 move toward a public, action-masked model-capacity harness is
directionally correct. It will only be a real change of direction if the harness
hosts a compact **recurrent, closed-loop learner trained with fresh simulator
interaction**, and if exact branch replay is used to query difficult
learner-induced states. If v203 merely scaffolds another lookup table,
feed-forward scorer, or offline extraction rule, it will preserve the failure
mechanism under a new version number.

The highest-probability breakthrough route is therefore:

> A small parameter-shared recurrent actor-critic with hard masked categorical
> actions, trained outside the tick loop from vectorized on-policy rollouts,
> augmented by exact-simulator counterfactual queries on states visited by the
> learner. Use a GRU or LRU before a Transformer; use the exact simulator before
> a learned world model; use quality diversity to generate curricula and data,
> not as a substitute for policy capacity.

This does not relax the promotion gates. It separates a **learnability gate**
from the much stricter **promotion gate**, so a training loop can be falsified
without rewarding a candidate merely for making no changes.

## What the repository is actually trying to build

The durable project objective is not a guarded Mind v1/v2 controller. Mind v3
must select legal primitive actions without heuristic fallback, hard-guard
action replacement, or confidence delegation. It must remain deterministic and
replayable for a frozen artifact, use only the public policy surface at runtime,
and eventually sustain survival and reproduction over broad held-out ecology
and controlled fixtures. See the local
[Mind controller vocabulary](../../python/evolution_sim/mind/CONTEXT.md),
[Mind v3 plan](../superpowers/plans/2026-05-09-mind-v3-autonomous-evolution.md),
and [experiment ledger](../mind-v3-autonomous-evolution.md).

The environment is not a conventional single-agent Markov benchmark:

- an agent receives local/self ecology, engineered navigation features, a
  utility-shaped public action affordance mask, and bounded prior public history;
- other agents act in the same tick, so an observation-valid movement can become
  resolution-invalid because occupancy changed;
- births and deaths change the population and therefore the future transition
  distribution;
- delayed hydration, energy, carrion contact, reproduction, and survival make
  current action value depend on prior trajectory context;
- the policy must generalize across deterministic worlds generated from seeds,
  while training itself remains stochastic unless deliberately controlled.

That combination is best viewed as a multi-agent POMDP with a deterministic
underlying simulator, not as a finite contextual bandit over exact feature keys.

## Local evidence: what has and has not been proved

| Evidence | What it proves | What it does not prove |
| --- | --- | --- |
| The original autonomous linear policy produced zero heuristic actions but collapsed on survival and births. | The no-heuristic runtime and measurement path are real. | The controller representation or learning rule is adequate. |
| v53-v55 PyTorch IQL candidates produced broad alive/birth gains and nonzero carrion survivors, but requested-action share remained collapsed around `0.63-0.66`; the actor score was blended with a `0.90` contextual prior and support data included evaluation seeds. v56-v63 then exhausted coefficient, prior-blend, extraction, and global-bias repair. | There is at least partial behavioral learning signal, and scalar repair of that family is not promising. | A direct neural actor, clean held-out generalization, or all offline-to-online/recurrent RL is closed. |
| v154-v176 exhausted the tiny archive / nearest-neighbor scorer lane. | Exact local similarity is not sufficient policy capacity. | Sequence memory or learned generalization is unnecessary. |
| v180, v186, and v195 consumed three training slices and failed shadow acceptance. | The three artifacts are not promotable and should not be rerun. | Three independent modern learning paradigms were tested. The latest artifacts share a table/scorer lineage. |
| v193 found replay-valid terminal support for six target fixture seeds, and v194 serialized 3,599 rows from selected support. | Successful behavior exists in the deterministic simulator. | The public input is Markov, the successful actions are causally identified, or the rows cover a learned policy's state distribution. |
| v195 collapsed to `eat` at requested-action share `0.9422`, with zero carrion survivors and broad regressions. | Positive support rows did not distill into a safe closed-loop policy. | The terminal-survival objective itself is impossible to learn. |
| v196 identified mask-only / low-specificity feature coverage as the immediate collapse mechanism. | Low-specificity overrides must be rejected. | Abstaining from all overrides is improvement. |
| v197 applied zero overrides and removed broad regressions. | A specificity gate can block a known collapse. | A learned policy improved anything; carrion survivors remained zero. |
| v198-v201 found that all present high-specific candidate occurrences were action-incomplete; v201 counted 2,714 missing current-valid action demands, 87.03% with only lower-specificity support. | Exact-table completeness is structurally poor at deployment states. | The answer is to fill every exact table cell. |
| v202 defines public observation, current mask, and public history as the capacity-harness boundary. | The next architecture can be leakage-safe and replayable. | A model architecture, optimizer, rollout protocol, credit assignment method, or learnability test now exists. |
| The current v203 scaffold audit passes 31 executable checks over a 604-input recurrent actor-critic and recurrent IPPO surface. | A leakage-safe, hard-masked, no-heuristic trainable implementation now exists. | Training was run by v203, slice 4 was consumed, or any learned artifact is promotable. |
| The repaired three-arm development canary ran base recurrent PPO, exact-branch auxiliary, and context-keyed shuffled-label auxiliary arms. | The exact auxiliary has a positive sampled broad-world signal over both controls under one matched learner initialization. | Carrion terminal survival improved, deterministic behavior improved, or the effect generalizes across learner seeds. |

The campaign baseline was commit `f2ab4bf`; v202 is the latest completed,
durable campaign step and authorizes only
`v203_public_masked_model_capacity_harness_scaffold_no_training`. The recurrent
implementation and development evidence below were collected from a dirty,
unpinned local research tree. The compatible reviewed source is now preserved
at commit `6ce1f41eab74cc28d2b61811618bc267dfb1f46d`, and the named evidence is
archived separately. This does not turn the work into a campaign slice,
retained policy artifact, runtime change, candidate, validation result, or
promotion result.

### Implementation and first matched recurrent-learning evidence (2026-07-21)

The proposed minimum learner now exists as a real simulator-integrated research
path rather than a table/scorer scaffold:

- a parameter-shared public recurrent actor-critic consumes the 541-value safe
  ecological observation, the current 20-action mask, and 43 values of public
  previous-action/resolution/reward feedback (604 inputs total);
- a shared encoder and GRU produce a hard-masked 20-action categorical actor
  and scalar value estimate, with one hidden state per living agent and explicit
  reset/death handling;
- recurrent IPPO/PPO collects policy-induced trajectories from canonical
  `SimulationWorld` instances with weights frozen for each rollout batch, GAE,
  complete per-agent sequences, full-sequence TBPTT, world-balanced losses,
  deterministic seed roles, and ordered process-parallel rollout collection;
- evaluation runs multiple replay-deterministic sampled policy streams per
  environment, separates policy decisions from passive terminal records, and
  repeats every candidate run to verify its exact behavior digest;
- the runtime actor has no heuristic fallback, fixture/seed input, action
  override, or table-based abstention path.

The current no-training v203 evidence is
`output/mind/mind-v3-v203-carrion-survivor-continuation-public-masked-model-capacity-harness-scaffold-20260722.json`.
It passes the pinned v202 source audit and all `31` current scaffold checks,
has logical exact digest
`e3a2875fb15fc64d64b8b122df42247ba377e1aa1095ad317038c946f60a6cb4`,
file SHA256
`c86e36302e887d222c1d880e2ddd4a7ce5d6fe2cdc144eec2de363e0f0df8271`,
and size `14,926` bytes. Its classification is
`m3_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold_ready_for_future_explicit_recurrent_ippo_training_no_training`.
Training, optimizer updates, slice-4 consumption, dataset creation or mutation,
training/runtime artifacts, runtime/default action-selection changes, gate
relaxation, and promotion all remain false. The older default-path report,
`output/mind/mind-v3-v203-carrion-survivor-continuation-public-masked-model-capacity-harness-scaffold.json`,
is retained only as pre-repair provenance. Its self-consistent digest
`753f0d264a18e9b38548733dfa8c6a1604fccb7b229da0241e939beb478d0065`
covers `30` checks and is not current validation evidence; the dated rerun adds
the now-required `reward_components` rollout-alignment field.

A source-frozen matched development experiment compared a 128-unit GRU with a
true feed-forward history ablation from the same initialization, learner seed,
80-world schedule, 143k-agent-transition scale, and evaluation sampling stream.
The reports were intentionally collected as unpinned development evidence:

- GRU report:
  `output/mind/mind-v3-public-recurrent-ippo-development-paired-gru-seed-1777057840.json`,
  exact digest
  `949439125b7be841b91003aa53475ba6caa50ae44777d1999719be509a2842ae`;
- feed-forward report:
  `output/mind/mind-v3-public-recurrent-ippo-development-paired-ff-seed-1777057840.json`,
  exact digest
  `2ff581fe428c6241d71d39ff4fa85412c2e65151159e02d87ab6c0fc22925cfa`;
- sealed paired analysis:
  `output/mind/mind-v3-public-recurrent-ippo-development-paired-memory-analysis-seed-1777057840.json`,
  logical exact digest
  `c6bfd4d60ea65fbdc99fb6707b07359ccc37ed32874db9ed173199e7bd29c427`.

The sampled GRU moved the carrion fixture from `0/24` to `7/24` non-extinct
environment-by-policy-stream run cells and from `0` to `12` total terminal-live
agents across those 24 cells. Its mean terminal alive count rose to `0.5`, mean
births to `7.5417`, and mean shaped return from `-24.894` to `-17.1158`. On
broad development worlds it reached `24/24` non-extinct run cells, mean terminal
alive `11.8333`, births `11.2083`, and shaped return `71.4075`. The matched
feed-forward arm reached only `1/24` non-extinct carrion cells and mean terminal
alive `0.0833`; the paired before/after difference-in-differences favored the GRU
by `+0.25` carrion run-cell survival, `+0.4167` terminal alive, `+0.8333` births,
and `+2.1616` return.

This is the first concrete evidence in this lane that a learned recurrent
policy can produce terminal carrion survivors through ordinary masked action
sampling. It is not a breakthrough claim yet. The 24 cells cross only six
environment seeds with four reused sampling streams under one learner seed, so
they are correlated repeated measurements rather than 24 independent learners
or environments. More importantly, deterministic argmax evaluation collapsed
to `eat` (`1.0` carrion share) and produced zero carrion survivors for both
architectures. Even the sampled GRU exceeded the pre-registered `0.50` action
share cap (`eat=0.5655` on carrion and `0.5427` on broad). The evidence therefore
shows a real learnable signal and a recurrent advantage, while also localizing
the next blocker to policy ranking/credit and action concentration rather than
basic gradient flow or simulator integration.

### Completed exact-counterfactual three-arm canary (2026-07-22)

The formerly proposed controlled comparison is complete. It used the same
128-unit GRU, learner seed `1777057840`, frozen initial model, eight-update
schedule, `10` worlds per update, `128` rollout ticks, training/evaluation
streams, and MPS runtime for all three arms. Every arm ran `80` real simulator
worlds. Evaluation used six canonical selection-role environment seeds, four
replay-deterministic sampling streams per environment, separate broad and
`carrion_only@120` contexts, and a deterministic argmax pass.

The implementation was adversarially hardened before the repaired runs:

- whole updates are atomic across PPO, exact collection, the auxiliary Adam
  step, model/optimizer state, and update counters; a failed attempt rolls back
  while its one-use evidence remains consumed;
- scheduled runs are single-use, prevalidated, and terminal after partial
  failure, and training tasks accept only canonical `train` or `curriculum`
  environment roles with globally unique task and policy-sampling identities;
- exact branches query every currently valid focal action from one copied
  simulator checkpoint, bind rows to the exact current model, verify replay,
  and keep environment, source-policy, and selection seeds separate;
- the auxiliary performs one shared-Adam step at `0.1` of the PPO learning
  rate, clips gradients at `0.5`, enforces mean/max forward-KL limits of
  `0.002`/`0.010`, and has no retry path after acceptance or rejection;
- shuffled labels use a public-context-keyed permutation over currently valid
  action values, not one globally reused positional shuffle;
- evaluator v4 revalidates nested reports and role-bound canonical selection
  seeds; external validation and one-use lockbox authorization are unavailable,
  so candidate, validation, lockbox, runtime, and promotion paths fail closed;
- branch returns are explicitly one exact rollout on one sequential simulator
  RNG tape. They do not estimate an expected causal effect or outcome
  uncertainty, and replay proves determinism rather than independence.

The first shuffled-label attempt failed closed after `594.68` seconds at update
4 and created no report. The retained log is
`output/mind/mind-v3-public-recurrent-ippo-development-cf-shuffled-1777057840.failure.log`
with SHA256
`ec88ed22bb20ed5252b20702f949dbcebf170dd38809ff38ed0aa3b490c072fe`.
The collector had required a mapping for every tick trajectory record even
though passive records correctly have no policy-decision diagnostics. The
repair aligns passive rows with explicit `None`, requires mappings for recurrent
decisions, and rejects diagnostics on passive or unknown-action-source rows.
The integrated new recurrent/counterfactual/comparison regression set then
passed `183` tests in `128.703` seconds. That was a focused repaired-surface
gate, not the repository-wide suite; its exact invocation was not persisted.

The repaired reports are:

| Arm | Report and logical exact digest | Real scale |
| --- | --- | --- |
| Base recurrent PPO | `output/mind/mind-v3-public-recurrent-ippo-development-cf-repaired-base-1777057840.json` — `7f5702e3c007a85d433404d7a05daf5d5c05b5433c9e32983798cdffe5928fcc` | `80` worlds, `81,130` transitions, `696.231` seconds |
| Exact branch auxiliary | `output/mind/mind-v3-public-recurrent-ippo-development-cf-repaired-exact-1777057840.json` — `4bf04d4f04cb7126d5c290fbb60c27b3e0d4ef1184265d539e02c5ec5d462ca7` | `80` worlds, `81,059` transitions, `32` bundles / `96` rows, `8/8` auxiliary steps accepted, `1,135.455` seconds |
| Shuffled-label auxiliary | `output/mind/mind-v3-public-recurrent-ippo-development-cf-repaired-shuffled-1777057840.json` — `b178fc116de68290cd75152d6ff75992e77d817f3b15d6f687d26670ee7e9689` | `80` worlds, `82,670` transitions, `32` bundles / `96` rows, `8/8` auxiliary steps accepted, `1,168.402` seconds |

The first comparison attempt also failed closed and created no output. Its log,
`output/mind/mind-v3-public-recurrent-ippo-development-cf-repaired-three-arm-comparison-1777057840.failure.log`,
has SHA256
`e6fce915fcfe82bf5fd04149773ab2cf817e45fb689dde67d7eca09574a64869`.
The v1 analyzer conflated the evaluator's broad container key (`broad`) with
the raw-run context (`broad_default`) and therefore falsely rejected the base
argmax paired deltas. Analyzer v2 fixes that namespace normalization, ignores
dormant base-only counterfactual knobs, tolerates only bounded float32 KL
roundoff, and does not equate the evaluator and auxiliary final-state hashes
because they use different encoding contracts. It also records both the
training and current analyzer hashes and accepts post-training source drift only
when the analyzer module is the sole changed runtime file. The final report
records exactly that one-file analyzer repair and zero non-analyzer drift. Its
training/current runtime-manifest digests are
`4061cea433101aa6c16d97e415fc902c2deaa44f95571a786b6361033bcf3a5d` /
`2cfc5f4a852485661187451a62636fcebbf3cbdab591ab20bab1bbd39893fd51`,
and its training/current analyzer-module SHA256 values are
`5c8d2ea9109bffaed7d8d4774992a84e9982c0bf4a9ea452e4f59f0d7f45ffd2` /
`5eea6f5f532ea4218be466f033536856ba717d0e9181ecc4889b419770a0c1b5`.
The current manifest has `380` files; the sole changed path is
`python/evolution_sim/mind/recurrent_counterfactual_comparison.py`.

The sealed v2 comparison is
`output/mind/mind-v3-public-recurrent-ippo-development-cf-repaired-three-arm-comparison-1777057840.json`,
logical exact digest
`24b6883965a9d802b92a9a5368589bc4bd3f35704951179a03c13a8d51293767`,
file SHA256
`4f50f8ce0acda6eff9d43e9c18339814e1c3e1d8e3520a6f6a59ec4c9429626b`,
and size `1,694,289` bytes. Its matched sampled contrasts show:

- on broad worlds, exact minus base was `+1.5833` terminal alive, `+0.625`
  births, and `+6.6213` reward per run cell; exact minus shuffled was `+0.9583`
  terminal alive, `+0.5` births, and `+3.8300` reward;
- on carrion, terminal-alive and terminal-nonextinct contrasts were `0` against
  both controls; exact was slightly worse in births and reward;
- all argmax outcome contrasts were `0`. Every arm had zero terminal alive on
  both broad and carrion argmax runs, with `eat` dominant at share `0.9941` on
  broad and `1.0` on carrion;
- sampled broad terminal-alive means were `9.7917` base, `11.3750` exact, and
  `10.4167` shuffled. Sampled carrion terminal-alive means were `0` for all
  three arms. Heuristic and unsupported requested-action counts remained `0`.

This is a real sampled broad-world signal for exact labels over both controls,
but not the requested carrion-survival breakthrough. The reports contain one
learner initialization per arm; environment seeds and sampling streams are
crossed repeated cells, not independent learner replicates. The branch target
uses one RNG tape, the final evaluator and auxiliary hashes have no same-state
cross-contract bridge, and the source/evidence tree was dirty and unpinned at
collection time. No artifact was retained, no campaign slice was requested or
consumed, and runtime, validation, lockbox, promotion, and gate-relaxation
lifecycle flags stayed closed.

Repository-wide validation after the repaired runs passed `npm run sim:test`
with `1,831` tests in `610.594` seconds against the `900`-second fast-suite
budget. `npm run sim:golden:quick` verified `seed7_ticks20` at SHA256
`ac112d0c8d5da1287852072873237a6c17bb22b8814bb79e65f6a8b2aa032ac9`
and `seed7_ticks100` at SHA256
`057e844f50bd6f36c4c4e0cc1c31561703fc778b3a98592fec2c0012aa82f23c`.
Ruff check, Ruff format check over the `33` new Python source/test files,
`compileall`, and `git diff --check` passed. These checks validate the original
development tree; preservation does not make it candidate evidence.

The compatible reviewed implementation is preserved at source commit
`6ce1f41eab74cc28d2b61811618bc267dfb1f46d`. All `11` reports, analyses, and
retained failure logs named by the current v203/recurrent entry are durable in
`/Users/njm/evolution-sim-p0-backups/20260722T053556Z-v203-recurrent-ippo-development-evidence.tar.zst`
and
`gdrive:evolution-sim-backups/archives/20260722T053556Z-v203-recurrent-ippo-development-evidence.tar.zst`,
archive SHA256
`fd97f087ffd9d0db643da61cb06596745ea7a90dc530959858c40c206a546427`.
A local extraction matched all originals byte-for-byte, and `rclone check`
reported `0` differences and `2` matching files for the archive and SHA256
sidecar. This closes source/evidence durability, not the missing policy-artifact
or scientific-replication gaps.

The next evidence-driven direction is not hardcoded action selection or gate
relaxation. It is to average exact targets over multiple independent branch RNG
tapes, add longer and terminal-aware targets, preregister a stronger but still
bounded utilization of the KL budget (the accepted auxiliary steps used only a
tiny fraction of it), and replicate the matched experiment across multiple
learner seeds on the GPU trainer after the source is reviewed, pinned, and made
durable.

### Preregistered recurrent scale campaign implementation (pending execution)

That next-step implementation is now source-complete but has not yet produced a
campaign result. The executable contract is deliberately stricter than the
single-learner canary:

- eight fresh learner seeds define eight paired blocks; each block crosses base
  PPO, exact aggregate counterfactual, and deterministically shuffled-label
  arms, yielding `24` learner/arm runs rather than `24` independent replicates;
- rollout sampling, branch selection, and continuation tapes are independent
  across learner seeds while all three arms remain common-random-number paired
  within a learner;
- `16` updates of `16` real policy-induced worlds per run, or `6,144` PPO
  training worlds before counterfactual continuations and evaluation;
- eight independent continuation RNG tapes per exact branch, with the same
  restored checkpoint and initial tape state reused across forced actions;
- deterministic branch-tick strata at ticks `16,40,64,72`, relative targets at
  `16` and `48` ticks, and an absolute fixed-tick target at tick `120`;
- for action `a`, horizon `h`, and tape `i`, the paired composite first computes
  `z[h,i,a] = delta_return + delta_focal_alive + 0.05*delta_population_alive +
  0.02*delta_births - 0.02*delta_deaths`, then the horizon score
  `L[h,a] = mean_i(z[h,i,a]) - 0.5*SE_i(z[h,i,a])` over eight tapes. The code
  first forms the normalized relative mixture `0.5*L[16,a] + 0.5*L[48,a]`,
  then blends that mixture at weight `0.5` with `L[120,a]` at weight `0.5`,
  giving effective weights `0.25/0.25/0.50`; uncertainty is applied before the
  horizon blend;
- no critic auxiliary on tick-120 survivors because they are truncated, not
  terminal, and the evidence format does not yet serialize a frozen bootstrap
  value;
- one stronger shared-Adam auxiliary update per PPO update, accepted only when
  forward `KL(pi_pre_aux || pi_post)` over the source public action mask,
  measured at one full-public-history reconstructed branch state per aggregate
  bundle (`2` states per update), has arithmetic mean at most `0.003` and
  per-state maximum at most `0.015`; either excess triggers exact model/Adam
  rollback with no retry;
- fresh scale-train, scale-curriculum, scale-selection, and learner registries
  that expose neither validation nor lockbox seeds;
- per-update optimizer/RNG crash checkpoints, frozen policy artifacts, exact
  CPU full-trajectory behavior digests, full-world replay manifests, and
  arm-independent evaluation sampling streams paired within learner seed;
- one content-addressed CUDA/Python/Torch/cuDNN/dependency/worker runtime
  provenance contract is embedded in every checkpoint, artifact, arm report,
  and the final analysis; resumed work fails closed on runtime drift;
- the final analysis binds the 24 report, artifact, update-journal, checkpoint,
  seed-plan, and evaluation-file hashes before computing any acceptance gate;
- a treatment-delivery gate requiring at least `12/16` accepted auxiliary
  updates in both treatment arms. For each accepted update, movement is the
  global model-state delta
  `sqrt(sum_tensor ||state_post - state_pre||_2^2)`; the gate sums those
  per-update deltas and requires strict `> 0.0`, with no epsilon tolerance. All
  accepted transactions must also remain within the preregistered KL bounds.

The implementation is in
`python/evolution_sim/mind/recurrent_scale_campaign.py`,
`python/evolution_sim/mind/recurrent_scale_execution.py`, and
`python/evolution_sim/cli/mind_v3_public_recurrent_ippo_scale_campaign.py`.
It is informed by exact fixed-context counterfactual credit, common-random-number
terminal credit, pessimistic uncertainty, and trust-region work, but those
papers are hypotheses for this simulator rather than result evidence. The
campaign remains development-only: runtime integration, validation, lockbox,
promotion, and gate relaxation stay closed regardless of the scale outcome.

### The latest slice is not yet the trainable example the project is looking for

The key implementation fact is visible in
[`build_v195_repaired_contract_support_policy_artifact`](../../python/evolution_sim/mind/carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training.py):

- each accepted row must already be terminal-survival support;
- public observation/mask/history are converted into up to eight discrete
  feature keys;
- terminal alive count, births, action validity, and resolution fields are
  combined into a scalar utility;
- utilities are averaged by exact feature key and requested action;
- the result is serialized as an action-utility table.

No neural or recurrent representation is fitted in this slice. More
importantly, the terminal alive/birth summary is attached to every accepted row
from a successful continuation. Without matched failed continuations or
same-state action interventions, this is weak evidence about the marginal
causal value of the individual action. The artifact can know that an action
occurred somewhere in a successful trajectory without knowing whether it
caused, harmed, or was irrelevant to that success.

### The latest slice also has a train/runtime history-construction defect

The v195 artifact builder calls `transition_value_feature_keys` with a fresh
`RolloutContextState()` for every compact row, even though its serialized
feature policy claims to use same-agent finalized public history. The compact
v194 rows contain the current observation and mask but omit the episode/tick,
agent identity, and prior public action/outcome sequence needed to reconstruct
that state. Runtime does the opposite: `v3_policy.py` retains one transition
value context per agent, scores with the live context, and updates it after each
finalized record.

Sequentially reconstructing the six source trajectories shows the scale of the
mismatch:

- `3,458/3,599` rows (`96.0822%`) have a different live exact-context key from
  the reset-context training key;
- all `3,599` reset keys are present in the artifact, but only `141/3,599`
  (`3.9178%`) live exact-context keys are present;
- all `2,039` exact-context keys and `2,598` coarse-context keys serialized in
  the artifact encode empty history.

This is a direct train/runtime feature-parity failure, not merely generic model
under-capacity. It provides a concrete explanation for v196's zero exact hits
and v198's roughly `3.15%` live high-specificity key presence. Before any new
capacity experiment, the harness must construct training and runtime features
through one shared sequential encoder and require `100%` row-level parity on
canonical replay. A larger model trained through the current compact-row path
would not repair this defect.

The runtime scorer then requires supported scores for every currently valid
action and rejects imputed or below-floor entries in
[`_transition_value_override_rejected_reason`](../../python/evolution_sim/mind/v3_policy.py).
That is a defensible fail-closed diagnostic, but exact action completeness grows
combinatorially with public history, mask, and observation detail. It is not a
viable representation strategy for a changing ecology.

### The linear controller has context features, not learned recurrent state

The active controller lineage in
[`evolution.py`](../../python/evolution_sim/mind/evolution.py) is a sequence of
fixed/hand-shaped feature projections with linear action heads. Later variants
append rollout and recovery context summaries, but they do not learn a compact
belief state through recurrence. This distinction matters: manually aggregated
history can expose a few known temporal variables, while an RNN can learn which
parts of prior observations, requested/resolved actions, and feedback predict
future viability.

## Research synthesis

### 1. Offline RL under support mismatch: the local evidence is textbook, but the simulator changes the answer

[CQL](https://arxiv.org/abs/2006.04779) addresses overestimated values on
out-of-distribution actions by learning a conservative Q-function, while
[IQL](https://arxiv.org/abs/2110.06169) avoids directly evaluating unseen
actions during policy improvement. Both are responses to the same problem now
measured locally: the deployment policy visits state-action regions not
supported by the static dataset.

Recent work refines, but does not repeal, that boundary:

- [State-Constrained Offline RL](https://openreview.net/forum?id=KcR8ykFlHA)
  allows unseen actions when they return to familiar state support, showing why
  “every action must have appeared at this exact key” can be over-conservative.
- [ReFORM](https://openreview.net/pdf?id=YvFsyRReeN) (ICLR 2026) uses expressive
  flow policies while enforcing action-distribution support by construction.
- [Cal-QL](https://papers.nips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html)
  and [RLPD](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf) treat
  offline data as initialization for online interaction rather than the final
  authority.
- [Adversarial data splitting for offline-RL generalization](https://proceedings.mlr.press/v235/wang24aj.html)
  explicitly trains across simulated train/validation distribution shifts, but
  still assumes a useful underlying dataset.

For this repository, the practical conclusion is stronger than the generic
offline-RL conclusion: **environment interaction is cheap, deterministic,
branchable, and already canonical. Remaining fully offline has little economic
justification.** Training can remain outside the simulator tick loop and
artifacts can remain frozen during each rollout; “online RL” does not require
non-deterministic in-tick weight mutation.

The exact-table action-completeness requirement is useful as a red alarm for a
table policy, not as a data collection objective. A learned function
approximator should generalize across nearby public histories, while targeted
simulator interventions should cover the states that the learner actually
visits.

The closest foundational match is
[DAgger](https://arxiv.org/abs/1011.0686): ordinary behavior cloning fails
because sequential actions change the future observation distribution; DAgger
iteratively collects states induced by the learner and queries a teacher there.
Here, the teacher need not be the heuristic controller. It can be an exact
branch planner or counterfactual evaluator over canonical simulator snapshots.

**Adopt now:** offline initialization plus on-policy or replay-mixed fine-tuning;
learner-state data aggregation; negative and counterfactual continuations.

**Do not adopt now:** another CQL/IQL coefficient sweep; filling exact feature
tables; a flow/diffusion actor before a recurrent categorical baseline works.
The action vocabulary is small, so policy expressiveness is not yet the main
bottleneck.

### 2. Recurrent and sequence policies: recurrence is the missing minimum, not a speculative luxury

The simulator hides relevant history behind current local perception. Time
since carrion contact, recent energy expenditure, failed movement caused by
occupancy, and whether prior attempts improved or depleted the agent can change
the correct next action without being recoverable from one observation.

Primary results support starting small:

- [Recurrent model-free RL can be a strong baseline for many POMDPs](https://proceedings.mlr.press/v162/ni22a.html)
  found a carefully implemented recurrent baseline competitive with or better
  than specialized POMDP methods on 18 of 21 evaluated environments.
- [R2D2](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf)
  showed that recurrent replay needs careful hidden-state handling, including
  sequence replay and burn-in; naively sampling isolated rows defeats the
  purpose of memory.
- [POPGym](https://arxiv.org/abs/2303.01859) and
  [Memory Gym](https://arxiv.org/abs/2309.17207) provide controlled evidence
  that memory architecture and task horizon materially affect results. Memory
  Gym found GRUs stronger again on its endless tasks even where Transformer-XL
  helped finite tasks.
- [Rethinking Transformers in Solving POMDPs](https://arxiv.org/abs/2405.17358)
  identifies recurrent inductive biases missing from standard Transformers and
  reports strong results from a recurrent linear unit.
- [Decision Transformer](https://papers.nips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html)
  demonstrates return-conditioned offline sequence modeling, but it remains
  downstream of trajectory coverage. It cannot infer a successful return mode
  that is absent or aliased in its data.

For the current scale and low-dimensional public observations, a GRU is the
correct first baseline. An LRU is a useful second comparison. A Transformer is
harder to train, easier to overfit on a small archive, and not justified until
the recurrent baseline and data pipeline are understood.

Required recurrent contract details:

- one hidden state per agent, initialized at birth and removed at death;
- public observation, current public mask, prior requested action, prior
  resolved action, and prior public feedback only;
- no seed, fixture, path, split, or provenance features;
- complete trajectory sequences, not shuffled individual rows;
- truncated backpropagation through time with an explicit burn-in prefix;
- stored action masks reused when computing the training log-probability;
- a frozen policy for each collected rollout batch;
- deterministic CPU inference and a serialized initial hidden-state contract.

**Adopt now:** GRU actor-critic and feed-forward ablation.

**Defer:** Decision/Trajectory Transformer, Mamba/SSM scaling, and open-loop
action generation until a recurrent learner demonstrates signal.

### 3. World models and planning: the exact simulator is already the best first world model

Frontier model-based RL is highly relevant strategically:

- [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) learns a
  recurrent world model and trains actor/critic behavior through imagined
  futures; the 2025 Nature paper reports one configuration across more than 150
  tasks.
- [TD-MPC2](https://openreview.net/pdf?id=Oxh5CstDJU) performs local trajectory
  optimization in a learned latent dynamics model and scales across many
  continuous-control tasks.
- [MuZero](https://www.nature.com/articles/s41586-020-03051-4) learns only the
  reward, value, and policy-relevant dynamics needed for tree search.
- [DIAMOND](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html)
  shows a diffusion world model can improve visual detail and serve as an
  interactive learned game engine.
- [MBPO](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)
  is a useful warning: synthetic rollout quantity must be balanced against
  model bias.

Those systems learn dynamics because the true dynamics are unavailable or
expensive. Evolution Sim has an exact, deterministic transition generator and
checkpoint/branch replay. Training a neural dynamics model before exploiting
that generator would add approximation error to a problem that currently has
none.

The near-term model-based route should be:

1. Snapshot a canonical learner-induced branch state.
2. For each currently valid focal-agent action, hold the comparison protocol
   fixed and branch exact continuations.
3. Reuse the exact checkpoint and other-agent policy state, but distinguish
   deterministic replay from event-aligned common-random-number coupling.
   The current simulator and sampled actor consume sequential RNG streams, so
   branch divergence can reassign later draws to different events.
4. Treat one replay-verified branch as one exact random-tape rollout target,
   not an expected causal effect. Estimate short- and long-horizon outcomes,
   rankings, and uncertainty across multiple independently derived continuation
   stream IDs before making expectation or causal-ranking claims.
5. Store action rankings, margins, and measured uncertainty as training targets, while
   keeping private checkpoint state out of actor inputs.
6. Train a recurrent actor/value model on public histories.
7. Replay-verify every proposed survivor trajectory.

This is also the natural continuation of
[Go-Explore](https://www.nature.com/articles/s41586-020-03157-9): remember
promising states, return to them, explore deliberately, then robustify the
result. The repository has implemented much of the “remember/return/explore”
half. It has not yet robustified through iterative learner-state aggregation.

A learned world model becomes worthwhile only when exact branching is the
measured throughput bottleneck. At that point it should propose or prioritize
branches; canonical replay should still decide whether evidence is real.

### 4. Multi-agent nonstationarity and credit assignment are first-order, not later polish

The action-resolution evidence already proves that another agent's same-tick
movement changes a focal agent's outcome. Population births/deaths and a shared
controller distribution change future dynamics as well. Treating each focal
row as an independent single-agent transition therefore introduces both
nonstationarity and ambiguous credit.

Foundational MARL work maps directly to this problem:

- [MADDPG](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)
  motivates centralized training because independent learners see a
  nonstationary environment and policy-gradient variance grows with agent count.
- [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794) uses a
  centralized critic and a counterfactual baseline that changes one agent's
  action while holding the others fixed, directly matching the available exact
  branch mechanism.
- [QMIX](https://proceedings.mlr.press/v80/rashid18a.html) factorizes a joint
  value into decentralized per-agent utilities under a monotonicity
  restriction. The restriction may be too strong for competitive ecological
  interactions, but the joint-credit framing is relevant.
- [MAPPO/IPPO](https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)
  shows a carefully implemented on-policy PPO baseline is competitive across
  several cooperative MARL testbeds.
- [Multi-level Advantage Credit Assignment](https://proceedings.mlr.press/v258/zhao25c.html)
  (AISTATS 2025) explicitly models rewards produced by different-sized agent
  subsets, relevant when one action affects the focal agent, a mating pair, a
  local conflict, or the whole population.
- [Non-stationary MARL](https://openreview.net/forum?id=LWuYsSD94h) (ICLR 2024)
  reinforces that changing co-players are a distinct learning problem, not
  ordinary single-agent noise.

The safest initial learner is parameter-shared recurrent IPPO: one actor/value
network, separate hidden state per agent, and only public per-agent inputs.
This avoids an immediate policy-contract debate about privileged centralized
critics. A second, explicitly separate experiment can add a centralized
training-only critic if the contract permits it. The serialized actor must be
provably independent of private/global critic features.

Counterfactual branch data should record at least:

- focal requested action and live resolved action;
- other-agent action protocol at the branch;
- individual lifetime/survival/reproduction deltas;
- population alive/birth deltas;
- whether the action changed occupancy or merely encountered expected drift;
- paired difference versus the branch baseline, not only terminal totals.

This directly attacks the credit-smearing problem in v195. It is more valuable
than adding another terminal coefficient.

### 5. Quality diversity and open-ended evolution: keep the archive, change its job

[MAP-Elites](https://arxiv.org/abs/1504.04909) builds a repertoire of elites
over chosen behavioral descriptors rather than returning one global winner.
[QDax](https://www.jmlr.org/papers/v25/23-1027.html) provides a current,
hardware-accelerated implementation family for QD, neuroevolution, and
population algorithms. [Enhanced POET](https://proceedings.mlr.press/v119/wang20l.html)
co-evolves environment challenges and solutions, while
[XLand](https://arxiv.org/abs/2107.12808) demonstrates broad capability from a
large, iteratively refined distribution of procedurally generated multi-agent
tasks.

The repository was right to use QD to preserve behavioral niches and survivor
continuations. It was wrong, or at least insufficient, to expect archive search
over a linear controller and a handful of fixtures to supply missing policy
memory and generalization.

The next role for QD is **curriculum and data coverage**:

- archive task/ecology configurations by difficulty and failure mechanism;
- archive trajectories by energy/hydration recovery, carrion contact phase,
  role, action diversity, births, and survival;
- select branch queries that add public-history coverage or discriminate
  uncertain action values;
- retain matched failures beside survivors;
- sample a balanced mixture for recurrent pretraining and online rollouts;
- keep task/fixture identity out of the runtime policy input.

Open-endedness is not the immediate milestone. XLand used an enormous task
distribution and training budget; it is evidence for curriculum diversity, not
evidence that an unbounded POET loop will repair a broken base learner. First
prove a small recurrent policy can learn and generalize across a bounded,
procedural curriculum.

Two newer artificial-life results sharpen the longer-term measurement route.
[Automated Search for Artificial Life](https://asal.sakana.ai/) uses a
vision-language foundation model to search several ALife substrates for target
phenomena, open-ended novelty, and diverse interesting simulations. That is a
credible researcher-side search and phenotype-scoring tool, but putting its
pretrained representation inside an Evolution Sim agent would violate the
project's no-pretrained-world-knowledge objective. Separately,
[System Neural Diversity](https://www.jmlr.org/papers/v26/24-1477.html)
formalizes behavioral heterogeneity across a multi-agent system. Its lesson for
this repository is to measure diversity over policy behavior and ecological
roles, not to treat a global action-frequency cap as a complete proxy for
open-endedness. Both directions become useful after a base learner can survive
and reproduce reliably; neither repairs the missing trainable controller.

### 6. Population-based training and evolution strategies: useful outer loops after the inner learner works

[Population Based Training](https://arxiv.org/abs/1711.09846) jointly trains a
population and adapts hyperparameter schedules. [OpenAI Evolution Strategies](https://arxiv.org/abs/1703.03864)
optimizes long-horizon return through scalable black-box search and requires
little inter-worker communication. Both fit simulator-heavy workloads.

They do not solve representation or target-identifiability problems:

- PBT will efficiently select whichever metric and seed matrix it is given; if
  those are contaminated, it accelerates overfitting.
- ES can optimize delayed terminal return, but earlier local evolutionary/QD
  search already showed that a linear policy class remains inadequate.
- Neuroevolving a recurrent network is possible, but is likely less
  sample-efficient than actor-critic training until a good reward and task
  distribution are established.

Use PBT only after the recurrent training loop passes basic learnability. Let it
adapt learning rate, entropy, recurrent horizon, curriculum mix, and perhaps
reward-scale schedules on training/selection tasks only. Use ES or QD as a
secondary diversity generator and as an outer loop over curriculum or compact
policy initializations. Freeze candidates before any validation or lockbox run.

The 2025 Nature paper
[Discovering state-of-the-art reinforcement-learning algorithms](https://www.nature.com/articles/s41586-025-09761-x)
shows that population meta-learning across many environments can discover an
update rule competitive with hand-designed algorithms. It is a genuine frontier
direction, but its data/compute scale and meta-objective complexity make it a
long-horizon research program, not the next simulator milestone.

### 7. Action masks, abstention, and uncertainty: distinguish legality, resolution, and fallback policy

[Invalid Action Masking](https://arxiv.org/abs/2006.14171) gives theoretical
and empirical support for masking categorical policy logits rather than merely
penalizing invalid actions. A 2026 preprint on
[valid-action suppression](https://arxiv.org/abs/2603.09090) further argues
that training without masks can suppress actions that are invalid in visited
states but valid elsewhere because network parameters are shared.

The implementation consequences are concrete:

- apply the tick-start public mask before softmax during acting and training;
- compute PPO ratios and entropy on the same stored masked distribution;
- never train the actor to infer unavailable actions from penalties alone;
- treat requested-action legality separately from live resolution outcome;
- do not relabel same-tick occupancy drift as a requested illegal action;
- include prior requested/resolved mismatch in public recurrent history so the
  policy can adapt on the next decision.

The local mask is an engineered affordance surface, not a complete safety or
utility oracle. `eat` and attacks include Foundation utility/condition gates,
whereas movement is closer to physical legality. A masked policy is therefore
constrained, but still needs to learn which valid action is useful.

“Abstention” is also underspecified in an action-taking system. A classifier can
reject a prediction; an embodied policy must still produce an environment
action. [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) and recent
[evaluation-time policy switching](https://arxiv.org/abs/2503.12222) make the
fallback policy explicit. If the fallback is the heuristic, it violates the
Mind v3 objective. If the fallback is the old linear v3 policy, then the new
model is a selective residual and its coverage, final action distribution, and
base-policy dependence must be reported honestly.

For the first full recurrent actor, avoid runtime abstention: mask invalid
logits and choose among at least one valid primitive action. Use epistemic
uncertainty during **training** to request exact branch labels or additional
rollouts. This converts uncertainty into data acquisition rather than hidden
runtime delegation.

Recent discrete flow and chunk methods are interesting but premature:

- [Q-chunking](https://arxiv.org/abs/2507.07969) targets long-horizon,
  sparse-reward offline-to-online learning through temporally coherent action
  chunks.
- [Adaptive Q-chunking](https://arxiv.org/abs/2605.05544) varies chunk horizon
  so reactive states can use short chunks.
- [Discrete flow matching for offline RL](https://arxiv.org/abs/2602.06138)
  and [offline-to-online discrete flow matching](https://arxiv.org/abs/2605.12379)
  extend expressive generative policies to discrete actions.

These 2025-2026 results are frontier signals, several still preprints. The
simulator's dynamic masks and occupancy races make open-loop execution unsafe.
If chunks are tested later, they should be option proposals revalidated and
replanned every tick, with chunk length collapsing near contact and contention.

### 8. Evaluation, generalization, and seed contamination: environment seeds and learner seeds are different axes

[RLiable / the Statistical Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
shows that few-run point estimates and aggregate means can be deeply
misleading. [Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html)
details experimenter bias, hyperparameter selection, multiple comparisons, and
weak statistical evidence. [Deep RL That Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
similarly demonstrates sensitivity to implementation and random seeds.

The deterministic simulator removes one source of transition noise for a fixed
world and artifact. It does not remove randomness from weight initialization,
minibatch order, exploration sampling, GPU kernels, or candidate selection. A
credible experiment needs both:

1. **learner seeds**: independent training runs of the same pre-registered
   configuration;
2. **environment seeds/tasks**: independent worlds or procedural fixtures used
   for train, selection, validation, and lockbox evaluation.

[Procgen](https://proceedings.mlr.press/v119/cobbe20a.html) demonstrates why a
diverse procedural training distribution and a distinct test distribution are
essential for RL generalization. Repeated adaptive use of the same holdout
causes overfitting even when examples are never inserted directly into the
training file; the broader adaptive-data-analysis result is formalized by
[the reusable holdout](https://pubmed.ncbi.nlm.nih.gov/26250683/).

Seeds `5,13,19,29,37,41` have been inspected repeatedly, and recent support
work also used several carrion fixture seeds. The repo already correctly warns
that these are no longer clean promotion heldouts. They should be labeled
`diagnostic_red_team` or `campaign_target`, never silently recycled as
validation.

The next protocol should pre-register a seed/task registry with:

- generator/version digest and split-generation algorithm;
- train, curriculum, selection, diagnostic, validation, and lockbox roles;
- first-seen report and every later use;
- whether outcomes were visible to a human or optimization loop;
- learner-seed count and environment-seed count;
- candidate/config count and selection rule;
- one-time lockbox access policy.

Report paired per-environment deltas because the simulator is deterministic for
a frozen artifact, but also report distributions across learner seeds. Use
stratified bootstrap intervals, interquartile mean, performance profiles, and
probability of improvement in addition to strict failure tables. None of those
aggregate statistics replaces the repo's per-seed promotion floor.

Most importantly, separate gates:

- **learnability:** does optimization change the policy in the intended
  direction on train tasks and generalize to fresh development tasks?
- **research acceptance:** does the recurrent candidate beat masked-random and
  feed-forward controls, avoid collapse, and produce nonzero fixture success?
- **promotion:** does a frozen candidate satisfy the strict broad/fixture,
  no-heuristic, action-diversity, replay, and fresh lockbox boundaries?

A no-op specificity gate can pass non-regression. It cannot pass learnability.

## Ranked research route

### Rank 1 — Build a real recurrent masked actor-critic baseline

Use v203 to scaffold capacity without training. The scaffold should consume the
existing `mind_ecological_policy_input_v1` safe projection (541 values, excluding
`self.mind_inheritance_available`), maintain one hidden state per agent, and
record aligned requested actions, observation-time masks, log-probabilities,
values, rewards, resolution outcomes, and distinct termination/truncation
semantics. Before the first autonomous-learning run, use the same network for a
disposable tiny masked-BC overfit and shuffled-label canary. That proves gradient,
mask, and sequence plumbing; it is not the research candidate.

Make the intended first autonomous trainable artifact explicit:

- shared GRU actor plus value head, initially small (for example 64-128 hidden
  units rather than a large Transformer);
- one hidden state per agent;
- masked categorical action distribution;
- parameter-shared recurrent PPO/IPPO trained outside the tick loop;
- vectorized/process-parallel canonical simulator instances;
- policy frozen for each rollout collection phase;
- artifact serialization, config/source/dataset digests, and deterministic CPU
  inference;
- actor and initial critic restricted to public inputs;
- no runtime heuristic, transition-table override, or abstention fallback.

A Gymnasium/PettingZoo wrapper is optional for this proof. The existing policy
`decide` and transition-feedback callbacks can collect recurrent on-policy
rollouts while weights remain frozen for each batch. Start with independent
one-shot worlds and add persistent process workers after measuring throughput;
the Python simulator, not GPU optimization, is likely to be the first scaling
bottleneck.

Why first: it simultaneously adds temporal capacity, trains on its own state
distribution, avoids static-data support extrapolation, and is a strong simple
baseline in both POMDP and MARL literature.

Disconfirming result: if this learner cannot overfit a tiny controlled
trainability fixture or improve over a masked-random policy, the fault is in the
adapter, reward/termination semantics, sequence batching, mask likelihood, or
optimizer—not in ecological generalization. Stop there and debug the learning
loop.

### Rank 2 — Aggregate exact counterfactual data at learner-induced states

Once the recurrent actor produces rollouts, select states by uncertainty,
failure proximity, novel public history, or QD niche. At a canonical snapshot:

- branch every or a stratified set of currently valid focal-agent actions;
- use paired continuations and record per-stream outcome deltas; aggregate
  multiple continuation streams before calling a ranking stable or causal;
- include failures as well as survivors;
- re-query states the current learner actually reaches;
- train action-ranking/value auxiliaries and optionally behavior-clone the
  exact planner's chosen action;
- iterate in a DAgger-style loop.

Why second: it directly repairs the 2,714 action-complete gaps where they
matter, without trying to enumerate the full exact public-state space. A single
exact branch supplies action-conditional rollout evidence; multiple matched
continuation streams are required before it can support causal-credit or
expected-value language. Even that is much stronger than assigning one terminal
survivor total to every action in a successful trajectory.

Disconfirming result: if identical or sufficiently long public histories have
highly contradictory best actions across matched branches, the public policy
surface is aliased. Measure conditional target entropy/Bayes error. Extend the
public history or observation contract deliberately; do not hide the problem
with a larger table or private inference feature.

### Rank 3 — Add explicit multi-agent credit and test CTDE as an ablation

Compare:

1. parameter-shared recurrent IPPO with public local critic;
2. the same actor with a centralized training-only critic;
3. a counterfactual/difference-return auxiliary from exact one-agent branches.

The actor input and runtime artifact must remain identical across comparisons.
If private/global critic input is not permitted by the project contract, omit
variant 2 and retain counterfactual targets computed offline.

Why third: occupancy races and global terminal labels create ambiguous credit,
but it is better to demonstrate a working recurrent loop before adding a
privileged critic.

Disconfirming result: if centralized or counterfactual credit does not improve
sample efficiency or seed robustness over shared IPPO, keep the simpler public
critic.

### Rank 4 — Use QD and a procedural curriculum around the learner

Generate bounded task variants over resource density, water/carrion geometry,
population density, occupancy contention, hazard pressure, role mix, and
horizon. Maintain an archive of environment-policy pairs near the learner's
competence frontier and a balanced archive of successful/failed behaviors.

Why fourth: it fights overfitting and supplies diverse experience after the
inner optimizer works. It also turns existing QD infrastructure into a data
engine rather than another linear-controller search.

Disconfirming result: if curriculum variants improve train success but reduce
fresh-distribution performance, descriptors or sampling are too narrow and PBT
is selecting task-specific shortcuts.

### Rank 5 — Add PBT/ES only for outer-loop diversity and schedules

Run PBT over training and selection strata for learning rate, entropy,
sequence length, curriculum mix, and reward normalization. Use ES/QD for
initialization diversity or non-differentiable outer objectives.

Why fifth: outer-loop search amplifies a functioning training signal; it cannot
create one.

### Rank 6 — Learn a world model only after exact branching throughput is measured as the bottleneck

Train a compact recurrent dynamics/value model on broad exact rollouts and use
it to prioritize candidate branches or imagine short continuations. Compare its
action ranking and calibration with the exact simulator on fresh branch states.
Every claimed survivor remains exact-replay verified.

Why sixth: Dreamer/MuZero-style modeling is strategically strong, but redundant
while the exact simulator is cheap and the policy learner itself is unproved.

### Rank 7 — Explore adaptive chunks or discrete flow policies after the recurrent baseline

Use them only if the evidence shows primitive-action exploration or multimodal
policy representation remains the bottleneck. Revalidate masks every tick and
keep a one-step fallback internal to the learned policy, not to a heuristic.

## Minimal falsifiable experiment matrix

The first training route should compare architecture and data source, not sweep
dozens of coefficients.

| Candidate | Memory | Data | Purpose |
| --- | --- | --- | --- |
| Masked random | none | online | Lower control and mask-contract check |
| Current linear v3 | hand-shaped context | frozen | Honest repository baseline |
| Feed-forward PPO | none | online | Tests whether recurrence is actually needed |
| GRU-PPO/IPPO | learned recurrent state | online | Primary trainable baseline |
| GRU behavior clone | learned recurrent state | v194 positives | Demonstrates static positive-only covariate shift |
| GRU-PPO + exact branch aggregation | learned recurrent state | online + counterfactual | Primary breakthrough hypothesis |
| Label-shuffled GRU auxiliary | learned recurrent state | corrupted controls | Proves gains depend on target information |

Run at least several independent learner seeds per pre-registered configuration
and enough fresh environment seeds to show a distribution, not six hand-picked
examples. Keep the original strict seeds diagnostic only. Use a newly generated
development split for this campaign and a sealed promotion lockbox.

The learnability gate should require:

- finite losses/gradients and reproducible artifact loading;
- clear improvement over masked random on training tasks;
- improvement over feed-forward PPO on at least one deliberately
  history-dependent fixture if the recurrence hypothesis is true;
- nonzero success on fresh development fixtures;
- no unsupported requested actions;
- no dominant-action collapse above the pre-registered research cap;
- zero heuristic action sources;
- performance degradation under label shuffling or history ablation;
- exact replay equivalence for frozen artifact rollouts.

Only after that should the candidate face broad non-regression and fresh
promotion lockbox gates.

## Stop rules that prevent another hundred-step diagnostic loop

1. **Trainability stop:** if GRU-PPO cannot overfit a tiny controlled task, do
   not generate more ecology data. Debug the adapter, masks, rewards, recurrent
   batching, and optimizer.
2. **Memory stop:** if feed-forward and recurrent agents are indistinguishable
   on a fixture designed to require prior public history, verify that the
   fixture truly aliases current observations before changing architecture.
3. **Coverage stop:** if online recurrent learning works but offline
   initialization harms it, drop the offline artifact; do not force the old
   archive into every future method.
4. **Aliasing stop:** if exact counterfactual labels remain contradictory after
   sufficient public history, revise the observation/history contract rather
   than increasing model size.
5. **Credit stop:** if terminal metrics rise while action entropy collapses or
   individual viability falls, inspect counterfactual/difference rewards before
   tuning terminal coefficients.
6. **Contamination stop:** once validation outcomes influence architecture or
   hyperparameters, relabel those tasks as selection/diagnostic and mint a new
   validation split.
7. **World-model stop:** if a learned model cannot match exact branch action
   ranking and calibration on fresh states, it may not generate acceptance
   evidence.
8. **Complexity stop:** do not introduce Transformer, diffusion, flow, or
   adaptive chunking until the GRU and feed-forward controls identify a
   capacity or exploration limitation those methods specifically address.

## Direction assessment

### What is good

- Deterministic replay, digest pins, explicit lifecycle flags, and per-seed
  failure reporting are stronger than typical research prototypes.
- The no-heuristic Mind v3 boundary is clear and prevents a guarded controller
  from being mislabeled autonomous.
- The simulator's exact branch replay is a major research advantage.
- The project has correctly refused to relax action-collapse and broad
  regression gates after failed candidates.
- v202 finally states a clean public masked capacity boundary.

### What is holding the work back

- Version throughput is dominated by post-hoc diagnostics and contract reports,
  while very few materially different learning systems have been trained.
- “Training slice” has sometimes meant table aggregation, which overstates how
  much neural/sequence learning has actually been tested.
- Successful trajectories were treated as action-value support without enough
  matched failures or focal-action counterfactuals.
- The campaign repeatedly used pointwise or hand-summarized context for a POMDP.
- Static offline support was prioritized even though exact environment
  interaction is cheap.
- Known strict seeds became an adaptive design surface; rigorous documentation
  of that contamination is good, but a fresh split is now mandatory.
- Promotion-style no-regression gates have been used too early as the main
  research signal, so “apply no override” can look cleaner than a learner that
  is making measurable progress.

### Bottom line

The project direction is scientifically sound at the contract level and stuck
at the learning-system level. The breakthrough is unlikely to be a novel scalar
loss, a denser exact support table, or another selective override. It is much
more likely to come from making the simulator serve the role it is uniquely
good at: a deterministic, parallel, intervention-capable experience generator
for a compact recurrent policy.

## Primary source map

### Offline, offline-to-online, and imitation

- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) — NeurIPS 2020
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169) — ICLR 2022
- [DAgger](https://arxiv.org/abs/1011.0686) — AISTATS 2011
- [Cal-QL](https://papers.nips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html) — NeurIPS 2023
- [RLPD](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf) — ICML 2023
- [Adversarial Data Splitting for Offline RL](https://proceedings.mlr.press/v235/wang24aj.html) — ICML 2024
- [State-Constrained Offline RL](https://openreview.net/forum?id=KcR8ykFlHA) — TMLR 2025
- [Evaluation-Time Policy Switching](https://arxiv.org/abs/2503.12222) — 2025 preprint
- [ReFORM](https://openreview.net/pdf?id=YvFsyRReeN) — ICLR 2026

### POMDP and sequence control

- [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://proceedings.mlr.press/v162/ni22a.html) — ICML 2022
- [R2D2](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf) — ICLR 2019
- [Decision Transformer](https://papers.nips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html) — NeurIPS 2021
- [POPGym](https://arxiv.org/abs/2303.01859) — ICLR 2023
- [Memory Gym](https://arxiv.org/abs/2309.17207) — ICLR 2023 / extended 2024
- [Rethinking Transformers in Solving POMDPs](https://arxiv.org/abs/2405.17358) — ICML 2024

### World models, planning, and exploration

- [Go-Explore: First Return, Then Explore](https://www.nature.com/articles/s41586-020-03157-9) — Nature 2021
- [MuZero](https://www.nature.com/articles/s41586-020-03051-4) — Nature 2020
- [MBPO](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html) — NeurIPS 2019
- [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) — Nature 2025
- [TD-MPC2](https://openreview.net/pdf?id=Oxh5CstDJU) — ICLR 2024
- [DIAMOND](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html) — NeurIPS 2024

### Multi-agent learning and credit

- [MADDPG](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html) — NeurIPS 2017
- [COMA](https://ojs.aaai.org/index.php/AAAI/article/view/11794) — AAAI 2018
- [QMIX](https://proceedings.mlr.press/v80/rashid18a.html) — ICML 2018
- [MAPPO/IPPO](https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html) — NeurIPS 2022
- [A Black-box Approach for Non-stationary MARL](https://openreview.net/forum?id=LWuYsSD94h) — ICLR 2024
- [Multi-level Advantage Credit Assignment](https://proceedings.mlr.press/v258/zhao25c.html) — AISTATS 2025

### Quality diversity, open-endedness, and population methods

- [MAP-Elites](https://arxiv.org/abs/1504.04909) — 2015
- [OpenAI Evolution Strategies](https://arxiv.org/abs/1703.03864) — 2017
- [Population Based Training](https://arxiv.org/abs/1711.09846) — 2017
- [Enhanced POET](https://proceedings.mlr.press/v119/wang20l.html) — ICML 2020
- [Open-Ended Learning / XLand](https://arxiv.org/abs/2107.12808) — 2021
- [Automated Search for Artificial Life](https://asal.sakana.ai/) — 2024
- [QDax](https://www.jmlr.org/papers/v25/23-1027.html) — JMLR 2024
- [System Neural Diversity](https://www.jmlr.org/papers/v26/24-1477.html) — JMLR 2025
- [Discovering State-of-the-Art RL Algorithms](https://www.nature.com/articles/s41586-025-09761-x) — Nature 2025

### Masking, safety, and evaluation

- [Invalid Action Masking](https://arxiv.org/abs/2006.14171) — 2020/2022
- [SPIBB](https://proceedings.mlr.press/v97/laroche19a.html) — ICML 2019
- [Overcoming Valid Action Suppression](https://arxiv.org/abs/2603.09090) — 2026 preprint
- [Deep RL That Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694) — AAAI 2018
- [RLiable / Statistical Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) — NeurIPS 2021
- [Procgen](https://proceedings.mlr.press/v119/cobbe20a.html) — ICML 2020
- [Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html) — JMLR 2024
- [Reusable Holdout](https://pubmed.ncbi.nlm.nih.gov/26250683/) — Science 2015

### Frontier methods to monitor, not immediate dependencies

- [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969) — 2025 preprint
- [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) — 2026 preprint
- [Flow Matching for Offline RL with Discrete Actions](https://arxiv.org/abs/2602.06138) — 2026 preprint
- [Discrete Flow Matching for Offline-to-Online RL](https://arxiv.org/abs/2605.12379) — 2026 preprint

The 2025-2026 preprints above should be treated as architecture signals until
their claims are independently reproduced and shown to fit this repository's
discrete masked, multi-agent, deterministic replay boundary.
