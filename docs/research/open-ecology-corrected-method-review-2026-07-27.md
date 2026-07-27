# Corrected Open-Ecology Method Review

Date: 2026-07-27
Repository boundary: current `master` at
`f2ab4bf5effc1ada1265f68d2fa2cb16d8f69cd4`; recurrent candidate source at
draft PR 28 commit `db81df9e8f8da7c02b6724d47f36d4077f5bb346`.

## Decision

The research supports building one genuine open-ecology lane now, but not by
copying any paper or the prior synthesis literally. The defensible first system
is a recurrent, individually selfish actor with a small heritable controller
genome; a value pathway with an explicit, evidence-selected gradient boundary;
persistent multi-generation
islands; a bounded diversity archive used for migration and recovery; optional
opaque communication; and synchronous fixed-shape inference batches. None of
these ingredients guarantees society or open-ended evolution. Together they
make those outcomes possible and measurable without naming cooperation,
language, roles, or carrion survival as optimization targets.

The candidate recurrent implementation is a useful substrate, not the finished
architecture. It has one shared encoder and GRU feeding both actor and value
heads ([source](https://github.com/Reallisticus/evolution-sim/blob/db81df9e8f8da7c02b6724d47f36d4077f5bb346/python/evolution_sim/mind/recurrent_actor_critic.py#L534-L681)),
and its rollout adapter evaluates one organism at a time in a `[1, 1, feature]`
tensor ([source](https://github.com/Reallisticus/evolution-sim/blob/db81df9e8f8da7c02b6724d47f36d4077f5bb346/python/evolution_sim/mind/recurrent_rollout.py#L204-L296)).
The world already has founder and child metadata hooks, but the child hook
passes parent identities rather than the parents' metadata and otherwise
returns an empty record
([source](https://github.com/Reallisticus/evolution-sim/blob/db81df9e8f8da7c02b6724d47f36d4077f5bb346/python/evolution_sim/env/world.py#L888-L930)).
Those are the seams to deepen.

## Heritable controller: compact genome, stable development map

A compact conditioning vector is the right first controller genome. FiLM shows
that a conditioning input can modulate a network through inexpensive
feature-wise affine transformations
([Perez et al., 2018](https://aaai.org/papers/11671-film-visual-reasoning-with-a-general-conditioning-layer/));
HyperNetworks demonstrate the more general alternative of generating recurrent
network weights from a smaller network
([Ha, Dai, and Le, 2016](https://arxiv.org/abs/1609.09106)). Neither paper
studies biological inheritance or this ecology. They justify the mechanism,
not the scientific outcome. A full hypernetwork would enlarge the mutation
surface, artifact contract, and inference cost before compact inheritance has
shown any benefit, so the first version should use bounded FiLM-style scale and
bias modulation rather than generated per-organism weight matrices.

The genome should be a small fixed-length float vector with explicit numeric
bounds and a versioned deterministic development map. Founders receive seeded
vectors; children receive deterministic recombination and mutation from actual
parent vectors; recurrent state always resets at birth. The shared backbone
must remain frozen during each demographic evidence window. Otherwise the same
inherited vector changes behavioral meaning while natural selection is
measuring it. For the first interpretable long run, train the backbone in
development worlds, then freeze it for the entire persistent-world campaign;
later backbone changes start a new versioned evidence window rather than
silently continuing the same evolutionary history. Migrating a bare genome
between islands is valid only when both islands share the exact backbone and
development-map digest. If backbones diverge, migration must move a complete
compatible bundle or wait for a declared synchronization boundary.

Within-life gradient learning and between-generation inheritance should remain
separate in the first experiment: do not inherit GRU hidden state, optimizer
state, or an individual's acquired weight changes. Earlier artificial-life
experiments found that the effect of inheriting acquired neural changes depends
strongly on environmental dynamics and did not establish Lamarckian inheritance
as generally superior
([Sasaki and Tokoro, 1999](https://direct.mit.edu/artl/article/5/3/203/2321/Evolving-Learnable-Neural-Networks-Under-Changing)).
The causal gate is therefore not “genomes differ.” At fixed observation,
history, action mask, and backbone, swapping genomes must sometimes change the
action distribution; small genome perturbations must have bounded effects; and
offspring genome digests, mutation draws, birth resets, and resulting actions
must reproduce exactly. The campaign should also report whether genotype
distance predicts behavioral and descendant differences. A latent that mutates
but is ignored is not evolution of controllers.

## Actor and value learning: test interference, do not blind by decree

The critic does not need to be genotype-blind. In the policy-gradient theorem,
an action-independent baseline may depend on the state available at that
decision without changing the expected policy gradient
([Sutton et al., 1999](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)).
A genotype-aware critic may reduce variance because genotype affects future
behavior and survival. The problem in the candidate is representational:
value-loss gradients pass through the same encoder and GRU that produce the
genotype-conditioned actor.

Primary empirical work supports treating this as an ablation rather than a
theorem. PPG explicitly identifies interference between shared policy and value
objectives and separates their optimization while retaining controlled feature
transfer ([Cobbe et al., 2021](https://proceedings.mlr.press/v139/cobbe21a));
IDAAC reports benefits from separate policy and value networks, while also
showing that naïve separation can remove useful dense value features
([Raileanu and Fergus, 2021](https://proceedings.mlr.press/v139/raileanu21a.html)).
The smallest decisive comparison is critic input
`public-history only` versus `public-history plus genotype`, crossed with
`shared trunk` versus `separate or value-stop-gradient trunk`. Select on fresh
development worlds using return, value error, advantage variance, actor/value
gradient cosine, diversity, and the causal genome-use gate. Do not choose the
winner merely because it produces more genotype variance.

## Communication: selfish rewards permit it, causal listening proves it

Communication is not structurally impossible under individual rewards.
Self-interested agents have learned manipulative communication that improved
their own outcomes
([Blumenkamp and Prorok, 2021](https://proceedings.mlr.press/v155/blumenkamp21a.html)),
and selfish agents in repeated partner-selection games learned retaliation and
cooperation without a prosocial reward
([Anastassacos, Hailes, and Musolesi, 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6190)).
These domains are much simpler than this simulator, so they establish
possibility, not expected success.

The first signaling treatment should expose distinct opaque tokens, assign no
token meaning and no communication bonus, use a preregistered low physical
cost, and remove self-echo from the primary inter-agent test. Token frequency,
message entropy, and correlation with the sender's next action are insufficient:
agents can exhibit positive signaling without any receiver listening. The
relevant primary work defines causal influence by intervening on the incoming
message while holding the receiver's other inputs fixed
([Lowe et al., 2019](https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf)).

Accordingly, the evaluator should reconstruct the same receiver state and RNG,
replace only the incoming token with every token and a null token, and measure
the change in its legal action distribution and short continuation. Longer
paired-tape controls should compare intact communication with receiver-blind,
token-permuted, and time-shifted inputs. A source-shuffled control is valid only
after contributor provenance is recorded outside the policy observation,
because the present spatial token field aggregates token intensities without
exposing sender identity in the policy view
([signal source](https://github.com/Reallisticus/evolution-sim/blob/db81df9e8f8da7c02b6724d47f36d4077f5bb346/python/evolution_sim/env/runtime/signals.py#L701-L730)).
Evidence requires both positive listening and a downstream sender or lineage
consequence; mere token emission is not communication. Resource caches,
transfer actions, and partner choice are later experimental treatments because
they alter the fitness landscape; they are not prerequisites for this first
feasibility test.

Population scale and density must be treated as explicit environmental axes,
not incidental constants. JaxLife reported its strongest large-scale
qualitative phenomena with 256 agents and weaker or absent effects at smaller
population sizes. A newer independent-MARL preprint likewise reports
coordinated, fragile, and jammed regimes separated by scale and density, with
temporal synchronization distinguishing sustained from collapsing coordination
([Yamaguchi, 2025](https://arxiv.org/abs/2511.23315)). That paper studies a much
simpler navigation task and is not evidence that this simulator will reproduce
its phase diagram. It is enough to reject a single low-density world as a fair
test of emergent interaction. The development campaign should therefore vary a
small preregistered set of initial densities while keeping reward and action
semantics fixed, then select a viable operating range using extinction,
congestion, diversity, and throughput rather than a cooperation score.

The tokenized observation surface must also become a configuration-bound policy
contract. PR28 validates both base and tokenized observation schemas, while the
policy projection still accepts only the static base vector size
([observation source](https://github.com/Reallisticus/evolution-sim/blob/db81df9e8f8da7c02b6724d47f36d4077f5bb346/python/evolution_sim/env/runtime/observations.py#L1022-L1062),
[policy source](https://github.com/Reallisticus/evolution-sim/blob/db81df9e8f8da7c02b6724d47f36d4077f5bb346/python/evolution_sim/mind/policy_inputs.py#L17-L83)).
Communication training must not begin until policy, artifact, replay, and viewer
all bind the same token count and field order.

## Persistent islands and archive: complementary timescales

Persistent worlds are necessary for the requested phenomenon. Resetting every
short episode erases resource history, repeated relationships, spatial
inheritance, constructed traces, and within-world frequency-dependent
selection. An archive cannot reconstruct those causal histories. It should
preserve viable alternatives and supply bounded migrants or extinction
recovery, while the islands preserve ecology.

MAP-Elites demonstrates why retaining diverse high-quality solutions can avoid
the single-winner bottleneck
([Mouret and Clune, 2015](https://arxiv.org/abs/1504.04909)); Enhanced POET
demonstrates populations of environment-solution pairs and cross-environment
transfer ([Wang et al., 2020](https://proceedings.mlr.press/v119/wang20l.html)).
Neither demonstrates persistent societies. Malthusian RL supplies a closer
two-timescale precedent—behavior inside episodes and population change across
islands—and shows specialization in its tested games
([Leibo et al., 2019](https://ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf)).
For this repository, actual births, deaths, descendants, and resource-limited
population sizes should remain the evolutionary mechanism; an external league
must not overwrite biological fitness with a cooperation score.

Archive admission should use viability plus a preregistered, behavior-derived
descriptor that does not name desired lifestyles. The archive is a bounded
reservoir, not the main world and not proof of open-endedness. MODES separates
change, novelty, complexity, and ecological potential and warns against
equating long-running turnover with open-ended evolution
([Dolson et al., 2019](https://direct.mit.edu/artl/article/25/1/50/2915/The-MODES-Toolbox-Measurements-of-Open-Ended)).
Those quantities should be streamed as descriptive time series alongside
lineage persistence, effective diversity, trophic occupancy, repeated
association, reciprocity, and group stability. They should not be optimized
directly in the first campaign.

## Deterministic batched recurrent inference

The model already accepts time-major `[time, batch, feature]` tensors, so the
performance repair belongs at the rollout boundary. At every tick, each world
must publish its existing immutable tick-start observations and action masks.
Only the pure Torch forward pass may be staged, in the simulator's exact
hash-permuted turn order rather than agent-ID order, and padded to a
configuration-bound shape. Sampling, recurrent-state commitment, and action
resolution remain sequential: an earlier attack can remove a later agent, and
that passive victim must consume neither a sampling draw nor a recurrent-state
update. Staged rows for such agents are discarded. This preserves the current
mutable per-world sampling stream instead of changing the scientific contract
to a new counter-derived RNG merely for speed. Recurrent state and previous
public feedback remain keyed by full world and agent identity. Cross-world
batching is a later step and must order rows by `(task_id, tick, turn_rank)`;
it is admissible only after exact requested-action, RNG-consumption, hidden
state, and passive-death equivalence tests pass.

GPU batching is a proven throughput pattern—Sample Factory combines batched GPU
sampling with parallel environment workers
([Petrenko et al., 2020](https://proceedings.mlr.press/v119/petrenko20a.html))—but
its asynchronous, off-policy architecture is not directly compatible with this
simulator's frozen-policy and exact-replay boundary. R2D2 also shows that
distributed recurrent learning must address stale hidden states and parameter
lag, using stored state and burn-in
([Kapturowski et al., 2019](https://openreview.net/pdf?id=r1lyTjAqYX)).
Here, synchronous collection with a frozen policy is preferable until measured
simulation latency justifies asynchrony.

JaxLife is the closest recent systems comparison to the requested direction: it
uses recurrent neural agents, inherited mutated controllers, communication,
terrain modification, and programmable tools, and reports qualitative
communication, agriculture, and tool-use phenomena
([Lu et al., 2024](https://arxiv.org/abs/2409.00853)). Its public implementation
JIT-compiles the world step and vectorizes agent inference rather than driving
one Python model call per organism
([source](https://github.com/luchris429/JaxLife)). Its scaling result is also a
warning against tiny demonstrations: the reported large-scale patterns appeared
at 256 agents and were absent or weaker in smaller populations. This is useful
engineering evidence for fixed-capacity vectorization and population-scale
experiments, not evidence that copying its environment or evolving every neural
weight would produce the same result here.

“Deterministic batching” must mean repeatability for one bound runtime contract,
not equality with batch-one inference. PyTorch explicitly states that a batched
calculation is not guaranteed to be bitwise identical to the corresponding
slice calculation, even for mathematically identical inputs
([PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)).
Artifacts therefore bind batch size, padding, row order, device class, dtype,
PyTorch/CUDA/cuDNN versions, TF32 settings, deterministic-algorithm mode, and
cuBLAS workspace configuration. Exact environment replay should consume the
persisted requested actions. Policy reproduction is a separate same-contract
check; cross-device CPU reevaluation is a fresh evaluation, not a claim of
bitwise trajectory identity.

## Frontier follow-up: measure organization without prescribing society

A June 2026 preprint proposes Multi-Scale Path Divergence (MSPD), a
population-level finite-resolution measure of how heterogeneous local
transition laws organize across time scales, and evaluates it both as a search
objective and as a post-hoc trajectory lens
([Akhtyrchenko et al., 2026](https://arxiv.org/abs/2606.17091)). The post-hoc
use is relevant here; the fitness use is not. Optimizing MSPD in the first
campaign would replace open ecological selection with another researcher
chosen complexity objective. If the simulator can expose a defensible local
transition-law representation, MSPD may be added later as an exploratory
streaming statistic, frozen before reading island outcomes, and reported
beside rather than substituted for the preregistered MODES, lineage, ecology,
and social time series. Until that estimator is independently reproduced, it
must not become an acceptance gate or reward.

The social-learning evidence points in the same conservative direction.
Independent model-free agents often fail to exploit social cues; a model-based
auxiliary can induce generalized social learning
([Ndousse et al., 2021](https://proceedings.mlr.press/v139/ndousse21a.html)).
Causal-influence rewards can also promote coordination and meaningful
communication
([Jaques et al., 2019](https://proceedings.mlr.press/v97/jaques19a.html)).
Both methods are informative future ablations, but adding either now would
bias the phenomenon the user explicitly wants to leave emergent. The first
campaign therefore keeps individual Foundation rewards unchanged and uses
counterfactual signal ablation only as an evaluator: it asks whether another
agent listened, without paying either agent to communicate. If the unforced
treatment yields no causal signal use, that is a valid result and a later
preregistered auxiliary-learning arm—not a reason to rewrite the completed
run.

## Implementation and campaign consequence

The implementation order follows the dependencies: first make Torch CI
non-optional and correct the observation/KL/provenance contracts; then add
bounded streaming telemetry and storage guards; then implement the controller
genome and the preregistered actor/value gradient-boundary comparison; then add
synchronous fixed-shape inference; finally enable the controlled communication
treatment and persistent islands. A fixed-shape saturation ladder must
establish memory and throughput before the long run.

The first real campaign should compare the recurrent heritable system against
its existing history-reset control and a genotype-zeroed control, with the
actor/value choice frozen from the small preregistered ablation. It should run
multiple persistent islands for many generations, retain periodic checkpoints
and short replay windows rather than every full trajectory, and stream compact
ecological and social event summaries. Communication is evaluated through the
predeclared causal interventions above. Carrion remains a mechanics regression
fixture only. Success at this stage is not “society emerged”; it is a
replayable long-lived ecology in which inherited controller variation
causally changes behavior, selection acts on that variation, population and
behavior diversity do not immediately collapse, and genuine inter-agent
effects can be inspected.

That is deliberately narrower than cumulative culture. Recent open-ended
evolution work argues that culture adds a second inheritance system capable of
preserving learned innovations across generations
([Froese et al., 2024](https://direct.mit.edu/artl/article/30/3/417/116175/Evolved-Open-Endedness-in-Cultural-Evolution-A-New)).
This repository should not label recurrent memory or opaque tokens as culture.
After the first long ecology is stable, a later treatment can expose
agent-created persistent environmental state or copied conventions and test
whether acquired information survives replacement of the original individuals.
That is a future major-transition experiment, not a hidden requirement or
reward in the first campaign.
