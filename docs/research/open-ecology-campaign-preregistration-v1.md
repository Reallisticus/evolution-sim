# Open-Ecology Campaign Preregistration v1

Date: 2026-07-27  
Status: development-only preregistration; not yet authorized to launch  
Research boundary: Mind v3 recurrent, heritable-controller open ecology

## Purpose and claim boundary

This campaign is the first attempt to produce the thing the project actually
needs: one long, replayable ecology in which neural agents live for many
generations, inherited controller variation can change behavior, recurrent
history can matter, and unforced interactions can be watched and measured.
Carrion is not the objective. It remains a mechanics regression fixture so
that choosing to eat a carcass is legal and potentially viable, but neither
training nor model selection receives a carrion-survival target, carrion bonus,
or `carrion_only@120` gate.

The campaign is not preregistering the appearance of a society. It is creating
conditions under which repeated association, signaling, response, reciprocity,
group persistence, and lineage-level differentiation could emerge without
being named in the reward. A successful run may still find none of them. A
long run with honest negative evidence is scientifically useful; token entropy,
population size, or a visually interesting replay alone is not a
breakthrough.

The method follows the corrected review in
`docs/research/open-ecology-corrected-method-review-2026-07-27.md`: a compact
heritable FiLM controller genome, a recurrent individually selfish actor,
an empirical critic/gradient-boundary ablation, fixed-shape synchronous model
inference, opaque physically costly signals with no simulator-assigned meaning,
and persistent multi-generation worlds. The shared neural backbone is trained
only in development worlds and then frozen for the persistent campaign.
Children inherit only their bounded 16-value controller genomes. They never
inherit GRU state, public-history state, optimizer state, or acquired neural
weights.

All evidence in this document is development evidence. The registry explicitly
contains no validation or lockbox seeds, so this campaign cannot authorize a
production promotion, a final generalization claim, or a claim that the result
replicates on sealed worlds. Any such claim requires a later preregistration
and fresh sealed seeds.

## Frozen contracts and seed boundaries

The campaign must be launched from one clean, pushed source commit. Its full
40-character Git SHA, canonical configuration JSON, seed-registry digest,
PyTorch/CUDA/cuDNN contract, artifact schema versions, and this
preregistration's SHA-256 must be written to a machine-readable launch record
before the first training update. A missing value, dirty checkout, or digest
change fails closed. Editing this file after evidence exists is prohibited; a
change requires a new version that says what changed and why.

All campaign seed identities come from
`mind_v3_open_ecology_development_seed_registry_v4`, whose canonical digest is
`ed4ba14609fe2e98795cc4352ec50ab1e172c3c758b1a268f6481ac024e348e0`.
It contains 512 training-world seeds, 128 development-selection seeds, 32
persistent-island seeds, eight learner seeds, and 32 independent high-64-bit
genome-stream roots. It also contains 64 disjoint operational benchmark seeds
that may execute only disposable benchmark updates to measure capacity and
throughput but may not produce a scientific model artifact, select a
configuration, or support a scientific claim, plus 16 disjoint engineering
proof seeds reserved for source-bound noninterference and continuation proofs.
Proof seeds likewise cannot tune configuration or support scientific outcomes.
Validation and lockbox roles are unavailable and must remain inaccessible.

The 64 operational benchmark seeds have disjoint, role-internal subaxes. Phase
A uses environment indices `0` through `15`, model-initialization index `16`,
and genome-stream index `17`. Phase B uses environment indices `18` through
`33`, model-initialization index `34`, and genome-stream index `35`. Selection
resource measurement uses environment indices `36` through `39`,
model-initialization index `40`, and genome-stream index `41`. The Phase-D
throughput H/Z/R triplet shares paired environment index `42`, with
model-initialization index `43` and genome-stream index `44`. Indices `45`
through `63` remain reserved and inaccessible unless a later preregistration
version allocates them. No operational benchmark may consume training,
selection, island, learner, scientific genome-stream, validation, or lockbox
seeds.

The eight learner seeds, in fixed order, are `1570849880`, `1323833341`,
`67094806`, `1871495966`, `411483968`, `1589768480`, `1483917974`, and
`372226408`. The long campaign uses the first four, chosen by registry order
before seeing results, rather than the four best learners. Its island seeds are
the first four island entries: `334007764`, `494074816`, `699844056`, and
`655856875`. Island index also selects the corresponding genome-stream root:
`16663129824008534063`, `16925559243916883243`,
`10876559716473268626`, or `16263353148861378742`. All three arms in a paired
learner-by-island block receive the same environment root, genome-stream root,
model initialization, and domain-separated policy-sampling root. Pairing means
common starting roots and contracts; after treatments change births, deaths,
or decision counts, their mutable streams may legitimately diverge. Reports
must not misdescribe post-divergence draws as identical counterfactual noise.

Policy-sampling seeds are derived with the repository's versioned
domain-separated derivation from the exact phase, learner index, environment
index, arm, update, and tape identity. They are never reused as world,
learner, or genome seeds. Every report includes both the identities and their
derived values.

Within phase A, cell is deliberately excluded from both the policy-sampling
identity and the genome-population world identity. For a fixed learner,
update, and world, A0 through A3 therefore receive the same policy RNG seed,
environment, genome-stream root, and founder-genome binding while retaining
cell-distinct task and evidence identities. This pairing is prospective and
does not claim mutable streams remain identical after cell behavior diverges.

## Fixed neural, environmental, and learning configuration

The neural policy uses the tokenized public observation contract with four
opaque communication-token channels, the stable 20-action ordering, encoder
width 256, GRU hidden width 256, one recurrent layer, and
`actor_film_v1` genome conditioning. The genome has 16 values in `[-1, 1]`;
founders use the existing absolute limit `0.35`, and inheritance uses the
existing deterministic recombination plus mutation contract with mutation
rate `0.20` and maximum per-mutation step `0.08`. Actor FiLM scale and bias
remain bounded by the existing development map. No controller width,
mutation parameter, or signal setting changes after phase A begins.

Communication is enabled in every experimental cell and every long-run arm:
four opaque tokens, two physical profiles per token, base radius 3, base
duration 6 ticks, decay `0.58`, base intensity `0.10`, trait intensity bonus
`0.20`, and base emission energy cost `0.005`. The current shared signal-cost
contract also applies that base cost to reproductive signaling; this is held
identically across all cells and arms and must be recorded rather than
mistaken for an isolated communication treatment. Signals have no assigned
semantics, no communication reward, no cooperation bonus, and no privileged
sender identity in the policy input. Sender/contributor identity may be
recorded outside the observation for later causal evaluation. The primary
receiver test must exclude self-echo; implementing and testing that exclusion
is a launch dependency.

Training uses recurrent PPO with learning rate `0.0002`, Adam epsilon `1e-8`,
discount `0.997`, GAE lambda `0.97`, policy clip `0.15`, value clip `0.20`,
value-loss coefficient `0.50`, entropy coefficient `0.02`, four update epochs,
sequence minibatch size 16, TBPTT length 128, burn-in 16, maximum gradient norm
`0.50`, normalized advantages, world-balanced loss, and post-step target KL
`0.02`. Policy and value parameters are initialized from the learner seed.
There is no auxiliary imitation loss, hand-coded action target, carrion target,
reward shaping, archive fitness bonus, or hidden heuristic action selector.
A KL rejection or rollback is evidence and remains visible; it is not silently
retried with a stronger update.
No phase-A update may enter the durable evidence prefix unless at least one PPO
minibatch was accepted, the reported parameter delta is positive, and the full
model-state SHA differs from the preceding committed state. Mixed updates that
accept one or more minibatches before a later post-step KL rejection remain
valid and retain the rejection diagnostics. The terminal authority binds the
initial and terminal model SHAs, requires them to differ, and reports cumulative
accepted-minibatch and KL-rejected-step counts across the complete prefix.
For every update, `post_step_kl_audit_count` must equal accepted minibatches
plus post-step KL-rejected steps; an attempted optimizer step cannot disappear
from the audit.

Development worlds use the existing 48-by-32 terrain, maximum population 320,
and otherwise the exact bound `WorldConfig` committed at launch. Phase A uses
64 initial agents to compare learning architecture without changing density.
Phase B uses a fixed per-world density cycle of `32, 64, 128, 64`, yielding
128, 256, and 128 training worlds per learner at those respective densities.
Phase C then treats population density as a real environmental axis rather
than assuming that a low-density success generalizes. The selected long-run
density is frozen before any persistent island starts.

## Campaign matrix

The campaign has four stages. The table fixes each stage's primary matrix;
selection replays are itemized immediately below it rather than hidden inside
the training-world count. All counts refer to distinct world executions, not
agent decisions, trajectory rows, or PPO minibatches.

| Stage | Fixed matrix | Scientific use |
| --- | --- | --- |
| A: critic/gradient ablation | 4 cells × 4 learner seeds × 8 updates × 16 broad worlds = 2,048 training worlds, 128 ticks each | Choose one value-path contract without changing the actor/genome contract |
| B: selected training | chosen cell reinitialized × 8 learner seeds × 32 updates × 16 broad worlds = 4,096 training worlds, 256 ticks each | Produce eight independent frozen development artifacts |
| C: density treatment | densities 64, 128, 256 × first 4 artifacts × 16 fresh selection worlds = 192 non-training worlds, 2,000 ticks each | Choose a viable operating density, not a social outcome |
| D: persistent ecology | first 4 artifacts × first 4 island seeds × 3 arms = 48 primary islands, 50,000 ticks each | Observe multi-generation ecology and the effect of inheritance and memory |

Thus the learning work contains 6,144 preregistered training worlds, of which
4,096 belong to the selected configuration trained again from scratch. No
phase-A weights are carried into phase B. Each phase-A artifact runs 160 trained
policy executions (32 environments × four stochastic tapes plus argmax) and 128
initialized-baseline executions on the exact same four stochastic tapes, with
no duplicate baseline argmax or causal battery. Across 16 artifacts this is
4,608 primary executions, followed by an equal 4,608 full authority
reexecution, for 9,216 physical world runs. Phase-B terminal selection similarly
contains 2,304 primary executions (1,280 trained plus 1,024 same-tape initialized
baseline) and an equal 2,304 authority reexecution, for 4,608 physical world
runs. Resource accounting records trained, initialized-baseline, primary-pass,
authority-reexecution, and total physical world-run counts separately. Phase C
has exactly the 192 executions shown in the table and no additional argmax
copy. All physical executions are part of the resource projection even though
they never update a model.

### Phase A: critic access crossed with value-gradient boundary

The actor always sees its controller genome through the same FiLM development
map. The 2-by-2 ablation changes only whether the critic sees that genome and
whether value loss may update the actor's shared public-history trunk:

| Cell | Critic genotype access | Value-loss gradient into shared trunk |
| --- | --- | --- |
| A0 | `none` | `shared` |
| A1 | `none` | `stop_gradient_v1` |
| A2 | `film_v1` | `shared` |
| A3 | `film_v1` | `stop_gradient_v1` |

“Stop gradient” refers to the value-loss path into shared recurrent features.
It does not make inherited genome values trainable, and it does not freeze the
value head. For a given learner seed, all four cells start with the same
actor/backbone tensors and consume the same ordered training-world list. The
additional critic-FiLM tensors are deterministically initialized from a
separate substream.

Phase A uses training-seed indices 0 through 127. Evaluation uses selection
indices 0 through 31, never used for PPO, with four fixed stochastic
policy-sampling tapes plus an argmax diagnostic, each for 512 ticks.
Selection-world outcomes are first aggregated over tapes within environment
seed, then over environment seeds within learner. The independent selection
unit is the learner seed, not ticks or decisions.

A cell is eligible only if it has finite outputs, exact action-mask legality,
exact same-contract replay, no hidden heuristic action source, no persistent
global action collapse, sufficient collapse evidence, causal use of its genome,
and strictly positive paired median normalized-return improvement over the
learner-seed initialized model on the same four selection tapes. Pairing is
hierarchical: take the median of the four trained-minus-initialized tape deltas
inside each environment, then take the equal-weight median of the resulting 32
environment values. Flattening all 128 tape rows is descriptive only and cannot
decide eligibility. The initialized baseline model SHA and report digest are
bound to the evidence, and the baseline does not receive an extra argmax or
causal execution. The causal genome test
samples 64 policy states from each of the 32 selection worlds, or 2,048 states
per artifact. It freezes observation, previous feedback, recurrent state,
action mask, backbone, and sampling RNG, then compares the original genome
with zero, donor, and single-locus perturbations. At least 10% of states must
have original-versus-donor Jensen-Shannon divergence at least `0.01`, mean
divergence must be at least `0.002`, and the 99th percentile total-variation
distance for a bounded `±0.05` single-locus perturbation must not exceed
`0.25`. An eligible cell must pass for at least three of four learners.
Global action collapse is detected over every possible rolling span of 3,000
consecutive decisions, not only windows aligned to decision zero. A span
qualifies when one requested action has share at least `0.80` and that same
action is dominant in at least 90% of represented lineages. Fewer than 3,000
decisions is insufficient evidence and cannot be eligible. Collapse is a
failure, not an invitation to tune a prior.

Held-out value and advantage targets use the training truncation convention.
Each world executes exactly `T` scored ticks. On a disposable clone of the
authoritative post-run world, the evaluator advances only the real tick-`T`
policy-visible ecology prefix: tick-start signal decay, climate and emissions,
resource regrowth, observations, action masks, and deterministic turn order.
It then evaluates each living agent exactly once using its previous feedback,
recurrent state, and inherited genome. It samples, commits, and resolves no
action, and the original world, environment RNG, runtime counters, policy
state, sampling RNG, model, and genome population must remain canonically
unchanged. Every living terminal agent appears in bootstrap evidence, but only
an agent with at least one scored decision may enter a target array. A child
born on the final scored tick is therefore retained and counted as
zero-decision evidence, but excluded from targets. Dead agents bootstrap zero.
A passive terminal death reward remains a separate raw outcome; in the value
target it enters exactly at the `gamma` boundary after the last scored action,
matching PPO training. Bootstrap rows, target eligibility, state-invariance
digests, and the target-contract version are part of full-behavior evidence and
must reproduce exactly.

The causal-state adapter is read-only but reaches protected recurrent state.
Before its output can be authoritative, the final exact source must run the
source-bound capture-noninterference proof over all four cell configurations,
the Cartesian product of all four cells and densities 32/64/128, the first
twelve seeds of the dedicated `open_ecology_proof` engineering role, and
aggregate worlds with observed births and deaths. It must not open an
`open_ecology_selection` world. For every case,
the capture adapter and the ordinary public recurrent policy must produce
identical summaries, trajectories, decision diagnostics, action/RNG evidence,
genome provenance, and unchanged model-state digests. A self-digested report
is only structural evidence: the launch dependency must reexecute the proof on
the exact source and match the full report. A short capture-versus-capture unit
test is not enough.

Selection is exactly two-pass. Per artifact the producer executes 160 trained
worlds plus 128 initialized-baseline worlds, or 288 physical worlds, and
emits only a provisional report. The authority pass first reconstructs the
canonical run contract from this preregistration, validates the complete
eight-update checkpoint/commit chain and terminal model, resolves the artifact
and run contract only through the terminal bundle's relative file references,
then independently reruns the same 288 worlds. Only an exact report match may
emit learner evidence. That receipt binds the terminal digest and final prefix
commit digest; caller-supplied artifact paths, source hashes, or self-signed
run contracts cannot choose those pins.

The clean-checkout producer is
`PYTHONPATH=python .venv/bin/python scripts/prove_open_ecology_capture_noninterference.py
--repository-root . --output <proof.json>`. Its twelve full cases are campaign
engineering evidence only. The dedicated proof seeds cannot tune configuration
or thresholds, and the proof does not consume scientific selection seeds.

Among eligible cells, selection is lexicographic. First maximize the median,
across learner seeds, of each learner's median per-decision normalized
individual return. Candidates within 1% of the leading absolute return are
tied. Ties minimize median held-out value RMSE, then median advantage variance.
If still tied within 1% on the relevant quantity, prefer
`stop_gradient_v1`, then critic `none`, because that is the smaller
interference surface. Actor/value gradient cosine, clip rates, KL rejections,
genome diversity, extinction, births, and action distributions are reported
for interpretation but cannot reorder this rule. If no cell is eligible,
phase B does not run and thresholds are not relaxed.

### Phase B: several-thousand-world selected training

The chosen cell is reconstructed from initialization for all eight learner
seeds. Each learner sees all 512 training seeds exactly once in registry order:
32 updates of 16 worlds, 256 ticks per world. The fixed density cycle is
determined by global world index, not by outcome. Training stops at update 32;
there is no best-checkpoint selection.

The terminal artifact from every learner is evaluated on selection indices 32
through 63 using the same four stochastic tapes plus argmax convention and the
same causal-genome battery, for 2,000 ticks per execution. At least six of eight
terminal artifacts must pass
numeric, mask, replay, no-heuristic, no-collapse, bounded-perturbation, and
causal-genome gates. A failing artifact is not replaced by its best earlier
checkpoint or by a ninth learner. If fewer than six pass, the persistent
campaign stops. The first four registry-ordered artifacts—not the four
highest-return artifacts—are the fixed phase-C and phase-D artifacts once this
six-of-eight gate passes.

### Phase C: density as an operating condition

The first four artifacts run without gradient updates on selection-seed
indices 64 through 79 at initial populations 64, 128, and 256, for 2,000 ticks.
World size and maximum population remain unchanged. Evaluation uses one fixed
stochastic tape per sampled world; there is no extra argmax execution in this
stage.

A density is viable only if each of the four learners remains nonextinct in at
least 12 of its 16 sampled worlds, emits no numeric or action-contract error,
does not satisfy the global action-collapse definition, and does not spend
more than 20% of decisions in unexpected resolution-invalid outcomes or
sustained movement congestion. Same-tick occupancy races explicitly covered by
the repaired resolution-evidence contract are reported separately and are not
silently counted as successful moves. Among viable densities, choose the
highest density. This deliberately creates more opportunities for interaction
without selecting on signals, cooperation, group score, or visual
interestingness. If no density is viable, phase D does not start; density
thresholds are not tuned after seeing the result.

### Phase D: heritable, zero-genome, and history-reset worlds

Every learner-by-island block has three arms:

* **H — heritable/history:** bounded founder genomes, deterministic
  recombination and mutation at birth, and lifelong recurrent/public-feedback
  history that resets only at birth, death, or world initialization.
* **Z — zero/history:** every organism's controller genome is the exact
  all-zero vector while recurrent/public-feedback history remains enabled.
  Architecture, parameter count, environment, and sampling machinery stay the
  same.
* **R — heritable/history-reset:** the same heritable-genome contract as H,
  but GRU state and previous public feedback are reset before every policy
  decision. No other observation or policy behavior changes.

The trained neural backbone is frozen from tick zero through tick 50,000.
There is no PPO, archive-directed replacement, fitness reshaping, or periodic
world reset in phase D. Natural births and deaths are the selection mechanism.
An island that becomes extinct records a terminal scientific outcome and is
not reseeded. Its paired siblings continue. The bounded archive, if present,
is observational in this first campaign: it may retain descriptors and genome
bundles but may not migrate, recover, or overwrite a primary island.

The full matrix contains 16 paired blocks and 48 worlds, totaling 2.4 million
scheduled island ticks before natural terminal extinctions. Scheduling order is
learner index, then island index, then H/Z/R arm. Worker or host assignment may
change for throughput, but task identities, row order inside a world, seed
roots, numerical runtime contract, and artifact names may not.

## The first real observation at tick 10,000

After the operational launch gate passes, the first scheduled block—learner
seed `1570849880` crossed with island seed `334007764`—is prioritized as one
H/Z/R triplet. Each member is a primary phase-D world, not a disposable pilot.
At tick 10,000, the runner atomically emits a restartable checkpoint, bounded
replay shard, ecological/social summary, causal-intervention sample, and a
frozen milestone report for all three arms. The report is generated from
predeclared metrics and is not used to select a learner, density, checkpoint,
reward, or configuration.

The same in-memory worlds then continue unchanged toward tick 50,000. They are
not restarted from the milestone checkpoint, and no parameter or seed is
changed after viewing the report. If an arm has naturally become extinct, that
terminal state remains the arm's outcome; it is not resurrected merely to
produce a later video. Only a contract, corruption, or resource safety failure
may interrupt the continuation. The remaining 45 worlds are the same
preregistered matrix whether they execute concurrently or through a fixed
queue. The 10,000-tick report gives the user an early, real interaction window;
it is explicitly not an interim scientific acceptance test.

## Measurement and causal interpretation

Every 100 ticks, the runner emits bounded ecological summaries: current
population, births and deaths from explicit events, population AUC, vitality
and resource distributions, completed generations and lineage depth, founder
and descendant persistence, Hill effective lineage diversity, controller
genome diversity, species/ecotype and trophic occupancy, requested and
resolved action distributions, action validity, movement congestion,
reproduction, attacks, feeding, drinking, and signal emissions. MODES-style
change, novelty, complexity, and ecological-potential series are descriptive
only. They are never rewards, archive admission scores for the primary worlds,
or reasons to stop an island.

Social measurement is likewise observational. It records repeated spatial
association, dyadic persistence, conditional response after another agent's
action or signal, reciprocity, stable group membership, partner turnover,
signal entropy, and token/profile use. These are computed from public events
and separately recorded signal-contributor provenance. A group is not called
a society merely because members remain close, and token use is not called
language merely because it has entropy or correlates with an action.

The causal-listening evaluator samples preregistered receiver decisions from
the 10,000, 25,000, and terminal windows. It freezes receiver observation
except the incoming signal channels, previous feedback, recurrent state,
action mask, and RNG, then substitutes null and every token/profile allowed by
the same physical observation contract. It reports action-distribution
divergence and short paired continuations. Longer paired-tape controls compare
intact input with receiver-blind, token-permuted, and time-shifted input.
Source-shuffled controls are allowed only when contributor provenance is
complete. Evidence of communication requires both a receiver-policy effect and
a downstream receiver, sender, or lineage consequence. Emission frequency,
entropy, or mutual information alone cannot pass.

The independent unit for phase D is the learner-by-island block (`n=16`), with
H/Z/R treated as paired arms. Tick and decision rows are repeated measurements,
not independent samples. Reports show all block-level values, paired effect
sizes, bootstrap intervals over blocks, and exact paired sign/randomization
tests where applicable. No p-value converts exploratory development evidence
into validation evidence.

Three outcome labels are kept separate:

1. **Operational completion** means the exact source/configuration/seed
   contract ran, checkpointed, replayed, and archived without integrity errors.
   It says nothing about ecology quality.
2. **Open-ecology feasibility** means at least 12 of 16 H worlds reach tick
   50,000 nonextinct, at least 12 have lineage depth of ten or more, genome use
   remains causal without global action collapse, and replayable inter-agent
   events exist. This is a development milestone, not evidence of society.
3. **Development breakthrough candidate** additionally requires a
   preregistered causal receiver effect with a downstream consequence in at
   least 12 H blocks, persistence across at least ten generations in at least
   eight H blocks, and a directionally consistent H advantage over both Z and
   R in at least 12 of 16 paired blocks on the frozen primary interaction
   persistence measure. Both primary H-versus-control contrasts must survive a
   two-sided exact paired randomization test with Holm correction at `0.05`.
   The interaction-persistence measure is fixed before launch as the geometric
   mean of normalized repeated-association duration, conditional-response
   effect, and group-membership survival; a zero causal-response component
   makes the score zero.

Failure to reach the third label does not invalidate the run. A claim stronger
than “development breakthrough candidate”—especially society, language,
culture, or open-ended evolution—requires fresh sealed validation, independent
replay, and additional treatments. Cultural inheritance is out of scope.

## Throughput and resource launch gate

The completed RTX 4070 SUPER ladder is encouraging but deliberately narrow. On
clean exact source SHA `32a51890b4c349470fd70f5e74ccb9cbea3b014c`, with
Python 3.14.3, Torch 2.11.0+cu130, CUDA 13.0, cuDNN 91900, float32 tokenized
input, and `actor_film_v1`, width 256 reached 1,207,800 model rows/s at batch
1,024 with median latency `0.848` ms and approximately 41.9 MiB peak allocated
VRAM. The width-256 actor-FiLM path was roughly 7–12% slower than the
disabled-genome control.

The report digests are
`f3a73e466113346f02e720e18cb6948df367ff2dc25bd234f5b2094347d5a80b`
for the initial actor ladder,
`04d8eca6f9e76e6f862e77ca087ccc71700ccf1b18e27ac5b4857e85f271ebe4`
for the width-256 disabled control.

The benchmark measures only one-step
`PublicRecurrentActorCritic.forward_sequence`. It excludes world construction,
observation construction, action sampling and resolution, recurrent-state
commit, training, telemetry, checkpoint I/O, and remote archival. It therefore
supports the preregistered width 256 and shows that the GPU can absorb large
forward batches; it does not estimate campaign duration.

Before phase A, the exact source SHA must run the end-to-end recurrent pipeline
benchmark on the intended training host with worker counts `1, 2, 4, 8, 16`,
one full-shaped 16-world update, 128 rollout ticks, fixed density 64, and both
`heritable` and `zero_all` conditioning. It uses benchmark-seed indices 0
through 15 in order for environments, reserved benchmark-seed index 16 for
model initialization, and reserved benchmark-seed index 17 for the controller
genome stream. It may not access training, selection, validation, or lockbox
environment seeds. It runs three fresh, identically seeded repetitions per mode
and worker count through the same fixed-capacity 320 collector used by the
Phase-A cells. The benchmark includes world construction, observation encoding,
policy sampling, ordered worker merge, GAE, and PPO. A worker
topology is eligible only when every repetition's final model-state and
semantic-evidence digests exactly match the one-worker case. The topology
minimizing the slower H/Z median wall time is frozen, with smaller worker count
as the exact tie-break. Phase-A cost is 128 full updates multiplied by that
slower median and a fixed `1.20` safety factor.

The Phase-A training wall envelope is frozen prospectively at exactly `604800`
seconds: seven consecutive 24-hour days. It is not a CLI-selected budget and
cannot be enlarged after observing benchmark or training results. The
machine-readable resource envelope's `evidence_sha256` is the SHA-256 of the
exact committed bytes of this preregistration document. Resource-envelope
construction must fail closed if the live document hash differs from the
source-sealed document hash, and preregistration accepts only the canonical
envelope for its exact clean source commit.

The same exact source must separately benchmark the phase-B shape before phase
B can be budgeted: one 16-world update, 256 rollout ticks, density cycle
`32,64,128,64` repeated four times, benchmark-seed environment indices `18`
through `33`, model-initialization index `34`, genome-stream index `35`, both H
and Z conditioning, and three fresh repetitions at worker counts one and the
already-selected topology. Exact model/evidence equality is again mandatory.
Phase-B cost is 256 full updates multiplied by the slower selected-topology
H/Z median and the same `1.20` safety factor. The phase-A tick rate may be
retained as a planning proxy, but it cannot authorize phase B.

Selection has an independent CPU process topology and must not inherit the
training worker count. On the exact source, a separate long-horizon resource
benchmark uses benchmark-seed environment indices `36` through `39`, a
canonical frozen width-256 actor-FiLM artifact initialized from benchmark seed
index `40`, benchmark genome-stream index `41`, H and Z genome populations,
horizons 512 and 2,000, and initial densities 64 and the population-cap stress
case 320. It runs the real observation, policy, world, bounded-evidence,
primary, and independent-replay paths. Evaluation-worker counts are
`1,2,4,8,16`, capped by task count, with three fresh repetitions per case. A
topology is eligible only when semantic evidence exactly matches the one-worker
case; the topology minimizing the slowest case median is selected, with smaller
count as tie-break. Primary and replay wall time are recorded separately. The
slower per-horizon median, multiplied by `1.20`, projects all 9,216 phase-A and
4,608 phase-B physical selection world runs. The stress density cannot
contribute scientific outcomes or model selection.

Phase A remains blocked until the authoritative phase-A training projection
fits its resource envelope, and phase-A selection cannot start until its
long-horizon projection does too. Phase B remains separately blocked until its
mixed-density training and 2,000-tick selection projections pass. Any
128-tick-rate extrapolation may be shown only as a planning proxy. A
dirty-source smoke result may guide engineering but cannot satisfy a benchmark
gate.

Before any of the 48 primary islands starts, the final exact source SHA must run
an end-to-end resource benchmark on every intended host class. It uses one
throughput-only H/Z/R triplet at the selected density for 2,000 ticks, paired
benchmark environment-seed index `42`, benchmark model-initialization index
`43`, and benchmark genome-stream index `44`, twice. It enables the production
observation builder, batching, sampling, action resolution, recurrent/genome
state commit, 100-tick summaries, replay bounds, and forced checkpoints at
ticks 1,000 and 2,000 so checkpoint overhead is conservatively represented.
It records total and component wall time, world ticks/s, agent decisions/s,
batch occupancy, CPU and GPU utilization, peak GPU memory, peak process and
host RAM, swap delta, temperature, Xid/OOM events, bytes per tick, checkpoint
size and latency, writer size and latency, disk free space, and measured
Google-Drive upload/verification bandwidth. Its worlds are operational data
and cannot enter selection or scientific reports.

The measured schedule must project the full 2.4-million-tick matrix to at most
72 hours at median observed throughput and at most 96 hours using the slower of
the two repetitions plus a 20% safety margin. On each host, GPU memory and host
RAM must remain below 80% of capacity, swap must have no positive delta, GPU
temperature must remain below the device's throttling threshold, and there
must be no Xid, OOM, non-finite value, or evidence-write error. Projected
active storage must be at most 200 GiB and must leave the greater of 100 GiB or
20% of the target filesystem free. The verified Google Drive destination must
report at least 300 GiB free before launch.

These are operational gates, not outcome gates. If ETA is excessive but the
exact batching/equivalence contract passes, add workers or GPU hosts and rerun
the same benchmark. The 48 scientific tasks, seeds, arms, ticks, ordering
rules, and numerical runtime contract stay unchanged; only their static host
assignment and queue width may change. The matrix is never shortened to make
an ETA pass. If multiple hosts cannot reproduce the same bound runtime
contract, use homogeneous host classes or treat each device contract as a
separately versioned campaign rather than pooling trajectories as exact
replicates.

## Bounded evidence and Google Drive storage

The Mac must not become the primary evidence store. Active training and island
files remain on the compute hosts. The Mac receives only the preregistration,
compact aggregate reports, policy/configuration artifacts, manifests, hashes,
and selected viewable replay bundles, with a hard aggregate download budget of
5 GiB unless the user explicitly requests a restore.

Each island writes a summary every 100 ticks and rotates a completed evidence
shard every 5,000 ticks. Full public decision records are retained only in
32-tick windows at opening, immediately before ticks 10,000, 25,000, and
50,000. Each shard allows at most 8,192 replay rows and 64 MiB of canonical
replay bytes; omissions are counted explicitly. A restartable checkpoint is
written atomically every 5,000 ticks, with a 256 MiB file ceiling. The runner
retains only the latest two locally after older immutable checkpoints are
uploaded and independently verified. Tick 10,000 and terminal evidence are
never omitted from cold storage.

Cold storage uses immutable campaign-scoped names beneath
`gdrive:evolution-sim-backups/archives/open-ecology/<campaign-id>/`. Every
closed shard or checkpoint bundle is represented by a deterministic
`tar.zst`, stable manifest, and archive-SHA sidecar. The upload must stream each
remote object back through SHA-256 and obtain an independent `rclone check`
with zero differences before local pruning is allowed. Active writer
directories are never archived or pruned. A changed file, missing sidecar,
quota shortfall, source drift, failed stream check, or remote hash mismatch
stops pruning and the run keeps its local source bytes. The previously verified
10.21-GB migration proves that this path and workflow can work; it does not
pre-verify any new campaign object.

The 200-GiB active-run budget is global across hosts. When 70% is reached, the
orchestrator must finish and archive already closed shards sooner; it may not
drop required evidence or prune unverified bytes. At 90%, new tasks pause after
an atomic checkpoint while existing active writes are brought to a safe
boundary. Reaching the hard budget without verified remote capacity is a
resource failure, not permission to delete evidence.

## Stop conditions and no-tuning rule

The campaign stops or pauses only for a scientific-contract or resource-safety
failure: source/configuration/seed digest drift, missing or mismatched genome,
non-finite model or world state, illegal requested action, replay mismatch,
checkpoint continuation mismatch, writer corruption, lost evidence prefix,
CUDA/Xid/OOM, swap growth, thermal throttling, disk safety breach, or failed
verified archival that threatens the local bound. An individual natural
extinction is recorded and does not restart that world or stop its siblings.
An uninteresting tick-10,000 milestone does not stop the campaign.

There is one pass through the matrices. Failed acceptance gates route to a new
architecture or implementation version, not another numbered diagnostic loop,
stronger hidden shaping, threshold relaxation, best-seed substitution, or
post-hoc action rule. No outcome observed in phase A, B, C, the 10,000-tick
milestone, or phase D may alter this version's seeds, rewards, widths, genome,
density rule, run length, arms, metrics, or causal interventions.

## Implementation readiness dependencies

This document fixes the experiment but does not assert that the current
checkout can execute it. Launch remains blocked until all of the following are
implemented and behaviorally proved:

1. The tokenized observation, 20-action signal surface, model artifact, replay,
   checkpoint, and viewer must bind the same field order and token/profile
   count. The primary policy observation must exclude communication self-echo.
2. Founder and child runtime hooks must attach the correct controller genomes
   from actual parents under `heritable` and the exact zero vector under
   `zero_all`. Birth resets, genome digests, recombination, mutation, and replay
   must have exact tests. The runtime may never fall back to a missing genome
   or heuristic action.
3. The four critic/gradient cells and the fixed world-density schedule must be
   expressible in the training runner and canonical configuration. Gradient
   probes must prove that `stop_gradient_v1` blocks only the intended value
   path.
4. Any synchronous fixed-shape rollout path must be opt-in until the exact
   Phase-A shape proves same-contract repeatability, scalar behavior agreement
   for that declared shape, ordered semantic-evidence equivalence, and a
   measured speed benefit on the intended host. PyTorch does not promise
   universal bit identity between batched and elementwise kernels, so
   higher-density numerical equality cannot be assumed. A slower batch path
   cannot satisfy the throughput gate merely because it uses larger tensors.
5. A persistent-island runner must advance the same world to 50,000 ticks
   without episode reset, keep the backbone frozen, schedule the 48 fixed
   tasks, emit the 10,000-tick report without replacing the running worlds, and
   monitor the declared resource envelopes.
6. Concrete adapters must fill all seven checkpoint envelopes: world,
   environment RNG, frozen policy, public feedback/history, policy-sampling
   RNG, genome population, and evidence-writer continuation. An uninterrupted
   run and a save/load continuation must produce identical subsequent world
   state, actions, draws, hidden state, genomes, and evidence digest. A
   container with `restartable=true` is not sufficient by itself.
7. The bounded writer must rotate completed shards, preserve a resumable
   evidence prefix, and record explicit birth, death, lineage, dyadic,
   contributor, congestion, and intervention events needed by the declared
   metrics. Periodic public summaries alone cannot infer all of them.
8. Genome-swap/perturbation and signal-intervention evaluators must operate on
   frozen real policy states with action-mask and RNG controls. Tests must
   reject entropy-only, correlation-only, self-echo, missing contributor
   provenance, and repeated-row pseudo-replication claims.
9. A machine-readable preregistration builder and validator must reproduce
   every matrix row, seed assignment, configuration and evidence digest in this
   document, then fail closed on unknown, dirty, or stale inputs.
10. The exact-SHA end-to-end training benchmark, separate 512/2,000-tick
    primary-plus-replay selection benchmark, multi-host equivalence checks,
    output lock, Google Drive quota check, immutable uploader,
    verification-before-prune path, and terminal aggregate validator must pass
    from a fresh clean checkout. Torch tests must be non-optional in CI for
    this lane.

Only after those dependencies and the resource gate are closed does this
document authorize phase A.
The honest near-term milestone is therefore not “run more diagnostics.” It is
to finish these specific execution seams, seal one exact source/configuration
contract, run the fixed 2-by-2 ablation and 4,096-world selected training once,
then start the primary long worlds and show the first unchanged H/Z/R triplet
at tick 10,000 on its way to tick 50,000.
