# Open-Ended Eco-Evolution After the Recurrent Scale Campaign

Date: 2026-07-24. Scope: research direction and one implementable training
architecture. Source boundary:
`cada67040ff3d27848d37cfde0645f37ec41174e`.
Research policy: primary papers, official proceedings, and repository source only

## Decision

The next project should not be another carrion-optimization campaign and should
not become v204 through v250 of increasingly specific fixture diagnostics.
Carrion is one ecological niche. The right requirement is only that the physics,
observations, masks, and rewards do not make scavenging a deterministic trap: a
legal controller must be able to exploit it under some conditions, and a lineage
that samples it must still be able to discover its way back out. Evolution does
not have to select the niche, and carrion survival should not be a training
reward, curriculum score, or headline acceptance gate.

The main objective is broader: maintain a changing population of genuinely
different inherited controllers in changing ecologies, let survival and
reproduction select among them, and measure whether the repertoire grows to
include specialization, repeated association, signalling, exchange, reciprocal
help or retaliation, and stable groups. None of those social behaviours should
receive a direct cooperation bonus. The system should supply memory, heritable
behavioural variation, repeated encounters, optional communication and neutral
exchange affordances, then let individually useful behaviour survive.

My recommendation is one coherent system, called the **Heritable Ecology
League**. It combines a structured recurrent actor, a small inherited neural
adapter per lineage, within-world demographic selection, an archive of diverse
viable lineages, a population of changing worlds, and a high-throughput batched
rollout engine. This is an informed experimental hypothesis, not a claim that
the literature already knows how to produce artificial societies. The
Artificial Life field still treats sustained open-endedness and major social
transitions as open problems, not solved recipes
([Dorin and Stepney, 2024](https://direct.mit.edu/artl/article/30/1/1/120293/What-Is-Artificial-Life-Today-and-Where-Should-It)).

## Why the current direction is too narrow and too slow

The current scale policy is a useful proof that real PPO training, masked
actions, recurrent state, deterministic replay, sealed evidence and independent
evaluation can coexist. It is not yet an architecture for open-ended evolution.
Its raw encoded observation has 542 values; the safe policy projection removes
one controller-private diagnostic, then concatenates 541 ecological values, the
20-action mask and 43 previous-public-feedback values into the 604-value learned
input. The scale-v2 preregistration specifically projected that input through a
192-unit `tanh` encoder and one 192-unit GRU before its 20-action and scalar-value
heads; these are campaign dimensions, not architectural constants. Every
organism therefore uses the same learned behavioural genome; lineages differ
biologically, but the learned controller itself does not inherit, mutate and
speciate with them. That departs from the repository's original Mind v3 design,
in which controller state belongs to the agent, founders vary, and offspring
inherit and mutate it.

The implementation also explains the low GPU utilization. Rollout collection
copies the policy to CPU workers. Each simulated organism calls the network
separately for one decision, while the Python simulator advances worlds and the
exact treatment repeatedly forks each selected state across valid actions and
future random tapes. The GPU receives comparatively small PPO batches only
after the expensive CPU experience has been generated. A second or faster GPU
cannot materially accelerate that control flow. More CPU cores can shorten it;
more GPUs become valuable only after inference is batched or independent league
islands are distributed across devices.

The exact counterfactual machinery remains scientifically valuable, but it is
too expensive and too specialized to sit inside every update of the main
open-ended learner. It should become a sparse regression and causal-audit tool:
sample a small number of surprising states after training, verify replay and
action feasibility, and investigate a concrete failure. It should no longer
multiply every training campaign into base, exact and shuffled arms.

## The Heritable Ecology League

The actor remains decentralized and sees only the established public surface,
but it stops treating the 5-by-5 neighbourhood as an arbitrary flat vector. A
small shared encoder represents the focal organism, each local cell, navigation
targets, recent public action feedback, and each opaque signal channel as
separate tokens. Two lightweight attention or spatial-mixing blocks pool those
tokens into a recurrent core. A 256- to 384-unit GRU is sufficient for the first
campaign; the total model should be roughly two to five million parameters, not
hundreds of millions. The key change is relational structure and heritable
variation, not making the existing flat GRU merely wider. PPO and IPPO are
credible foundations rather than obsolete baselines: carefully implemented
PPO variants were competitive across four cooperative multi-agent benchmarks
([Yu et al., 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)),
and independent PPO matched or exceeded more elaborate joint learners in the
tested StarCraft scenarios
([de Witt et al., 2020](https://arxiv.org/abs/2011.09533)). Those results do not
establish performance in a mixed-motive evolving ecology, so the project should
retain recurrent IPPO and add a training-only, permutation-invariant value
baseline only if an ablation shows that population context improves return
prediction. The runtime actor must remain local and replayable.

The controller is `pi_theta(action | observation, memory, genotype)`. `theta` is
a shared sensorimotor backbone learned by recurrent PPO. The genotype is a
small, bounded controller latent stored through the existing
`mind_inheritance_metadata` path. A shared deterministic development map turns
that latent into FiLM-style modulation and small initialization offsets; the
organism does not carry, inherit, or merge a full per-agent adapter. Founder
genotypes are sampled from the current archive; children recombine and mutate
them through the existing reproductive lifecycle. The genotype is not smuggled
in as a private world observation: it is part of the organism's own controller,
in the same way that inherited biological traits already shape its body.
Recurrent state is within-life memory and is reset at birth; genotype is
between-generation memory. PPO improves common perception and behavioural
primitives, while ordinary in-world survival and reproduction select the latent
variants that use those primitives well. This shared-backbone/small-genome
split is the practical way to let thousands of organisms differ without
storing and training a full neural network per individual.

This inheritance path needs a stricter replay contract than the current
metadata placeholder. A versioned controller-genotype record must bind the
latent dimension and numeric bounds, founder-sampling identity, deterministic
development-map version and parameters, mutation and recombination operators,
parent identities, birth tick, and every RNG state used to produce the child.
The trajectory must record the resulting genotype digest and the artifact must
bind the shared backbone plus development map. Loading, reproduction and replay
must reconstruct that digest exactly; recurrent state must be zero at every
birth and must never cross an individual or generation boundary. Until those
properties have focused tests and exact CPU replay, controller inheritance is
an engineering proposal rather than an authorized training surface.

Eight persistent league islands are a sensible first scale. Each island owns a
backbone checkpoint, optimizer, bounded founder-genotype reservoir and several
parallel worlds. Worlds vary only through replayable configuration parameters
already meaningful to the ecology: season severity, spatial resource
patchiness, hydrology, hazard pressure, founder mixture and population density.
Every league season freezes the actor during collection, trains it from
policy-induced trajectories, evaluates the resulting island bundle on fresh
development worlds, and exchanges migrants. Population-Based Training provides
the proven systems pattern for asynchronously copying stronger models and
mutating their training hyperparameters
([Jaderberg et al., 2017](https://arxiv.org/abs/1711.09846)); AlphaStar shows
that a changing league can preserve strategies and counter-strategies in a
complex competitive domain
([Vinyals et al., 2019](https://www.nature.com/articles/s41586-019-1724-z)).
Those results support population training; they do not prove that league
training yields biological speciation or society.

Adaptive worlds are development data only. Before the campaign, seal a
promotion matrix of ordinary broad ecologies, edge-condition mechanics
fixtures, and long multi-generation worlds, with independent environment,
policy-sampling and founder seeds. League replay, PBT selection, archive
admission and hyperparameter mutation must never read it. Candidate snapshots
are evaluated against that immutable matrix only at declared boundaries, and
promotion remains fail-closed on replay integrity, legality, per-seed viability
regression, diversity collapse and source/provenance mismatch. Social metrics
remain descriptive: a candidate is neither promoted nor rejected merely for
cooperating more or less.

The league should not keep only one highest-return winner. A compact
quality-diversity archive stores complete compatible bundles: the shared
backbone, founder genotype pool, exact source contract and a trajectory-derived
behaviour embedding. Quality is fresh-world viability and descendant
persistence, not cooperation. Diversity comes from an unsupervised embedding of
ordinary trajectory summaries, so the archive does not encode a human-written
preference for attack, signalling, sharing or any other specific lifestyle.
MAP-Elites established the value of retaining high-performing solutions across
behaviour regions
([Mouret and Clune, 2015](https://arxiv.org/abs/1504.04909)), while AURORA showed
that dimensionality reduction can learn the behaviour descriptor rather than
requiring hand-coded niches
([Cully, 2019](https://arxiv.org/abs/1905.11874)). These methods produced
diverse repertoires in their tested domains, not open-ended societies. Here the
archive is a diversity-preserving migration reservoir and an empirical record
of innovation; it never changes an organism's immediate reward.

World selection should initially use prioritized replay rather than a full
learned adversarial world generator. Randomly mutate bounded ecology
configurations, admit only worlds in which at least one current lineage is
viable but the world is not already saturated, and replay worlds with high
learning progress or cross-lineage disagreement. Enhanced POET demonstrated the
usefulness of populations of environment-solution pairs, minimal criteria and
cross-environment transfer
([Wang et al., 2020](https://proceedings.mlr.press/v119/wang20l.html)).
PAIRED and replay-guided environment design demonstrated adaptive curricula and
improved zero-shot transfer in their benchmark domains
([Dennis et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/985e9a46e10005356bbaf194249f6856-Abstract.html);
[Jiang et al., 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html)).
For this simulator, a simple replay store is preferable to a second adversarial
neural network until bounded world mutation has proved useful.

Within each world, keep the existing selfish homeostatic and reproductive
reward. Do not add team return, kin bonus, communication reward or a
``cooperate`` action. Population size should continue to emerge from births and
deaths rather than from a league scheduler inventing biological fitness.
Malthusian reinforcement learning found that return-linked subpopulation
dynamics can drive specialization and ongoing strategic change in social
dilemmas
([Leibo et al., 2019](https://ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf));
that is encouraging, but its task-level result is not evidence that the same
transition will occur here.

The environment does need usable social affordances. The repository already has
opaque communication actions, but emission is disabled by default and the
observation collapses all token identities into one scalar field. Enabling those
actions without exposing separate token channels would create several buttons
whose messages the receiver cannot distinguish. The open-ended ecology config
should enable the existing token actions, expose one local decaying field per
opaque token, charge a small physical energy cost, and assign no simulator
meaning or reward to any token. The next social-contract slice should expose
bounded, stable public cues that make repeated individual encounters
recognizable, make mate choice an explicit policy decision instead of an
automatic local best-partner lookup, and add a conservative resource-flow
affordance such as a lossy public cache. The cues must not reveal private state
or label friends, kin, roles, or groups; a cache must not assign ownership or
call the act sharing. These are affordances, not forced cooperation. If later
agents leave persistent signals, cache layouts, trails, or constructed objects
that outlive their creators, those artifacts can support a cultural timescale
distinct from both within-life recurrent memory and inherited controller
genotype. Recent artificial-cultural evolution work demonstrates that evolved
transmission mechanisms can sustain open-ended cultural lineage diversification
in a deliberately constructed model, which makes a persistent public medium
worth testing here without turning it into a claim that culture or society will
necessarily emerge
([Raviv et al., 2024](https://direct.mit.edu/artl/article/30/3/417/116175/Evolved-Open-Endedness-in-Cultural-Evolution-A-New)).
Resource-exchange work has shown that selfishly rewarded agents can invent an
exchange protocol when the environment supplies both transfer and repeated
congregation
([Garbus and Pollack, 2024](https://direct.mit.edu/artl/article/30/1/28/119154/Emergent-Resource-Exchange-and-Tolerated-Theft)).
Partner-selection experiments likewise produced cooperation from selfish
objectives when agents could choose whom to revisit
([Anastassacos et al., 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6190)).
Conversely, vanilla model-free agents often failed to use social information
without appropriate environmental constraints
([Ndousse et al., 2021](https://proceedings.mlr.press/v139/ndousse21a.html)).
The defensible claim is therefore that distinguishable signals, repeated
encounters, memory and optional exchange make social emergence possible; they
do not guarantee it.

Every social affordance must first pass the same deterministic Foundation
boundary as food or movement. Its full mutable state, decay and conflict order
must be explicit; observations may contain only bounded public facts; action
masks and costs must be reproducible; trajectory, summary and viewer schemas
must version the surface; malformed or mixed-version artifacts must fail
closed; and default-disabled runs must remain byte-identical. Focused tests must
cover competing same-tick uses, death and birth boundaries, capacity limits,
replay after serialization, and the absence of hidden kin/team labels or
affordance-specific rewards. Those are mechanics acceptance criteria, not a
requirement that trained agents use the affordance.

## Making the campaign materially faster

The first speed project is not installing another RL library or adding a second
GPU. It is changing the rollout boundary. At each simulator tick, CPU world
workers should publish all living organisms' encoded observations, masks,
genotypes and recurrent states to one inference queue. A GPU service performs
one large actor batch across many worlds, returns actions in deterministic
`(world_id, tick, agent_id)` order, and stores next recurrent states by the same
key. Workers stay alive across updates, policy snapshots change only at declared
rollout boundaries, and collection and PPO learning are double-buffered. This
preserves frozen-policy replay while replacing thousands of tiny CPU forwards
with dense GPU work. Sample Factory validates the general architecture of
asynchronous actors plus batched GPU sampling
([Petrenko et al., 2020](https://proceedings.mlr.press/v119/petrenko20a.html));
its published frame rate is not a forecast for this much richer simulator.

The second speed project is profiling and moving the hottest environment kernels
out of Python after batching exposes them. Occupancy resolution, observation
assembly, diffusion and resource updates are likely candidates for a
preallocated array kernel in Rust/C++ or a functional JAX training twin. The
canonical Python simulator remains the authority: a candidate fast kernel must
match exact small-world trajectories and state digests before it can produce
training data. EnvPool identifies environment execution as a common RL
bottleneck and demonstrates the value of compiled vectorized environments
([Weng et al., 2022](https://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html)).
Neural MMO 2.0, the closest published large-population ecology platform, reports
a threefold speed improvement from a simulator rewrite
([Suarez et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html)).
JaxMARL demonstrates much larger gains when both environments and algorithms are
accelerator-vectorized
([Rutherford et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5aee125f052c90e326dcf6f380df94f6-Abstract-Datasets_and_Benchmarks_Track.html)),
but that would require a real functional simulator port; installing JAX around
the current object-heavy Python world would not provide those gains.

Once batching exists, the current RTX 4070 should be enough for one two-to-five
million-parameter island learner and inference service. A machine with more CPU
cores will improve the current code immediately. Additional GPUs are worthwhile
for independent islands, learner seeds and archive evaluation; they are not
especially useful for splitting one small GRU. The campaign should record
environment steps per second, agent decisions per second, inference batch-size
distribution, simulator time, inference time, learner time, queue wait, CPU
utilization and GPU utilization before any hardware purchase. No honest speedup
factor can be promised until that profile exists.

One production-shaped measurement is now available for the existing exact-fork
path. On `gpu4070` at exact source
`4032f7b6a4b4241009967f7d41c1bca48a758ade`, two scale-v2-shaped bundles with
eight tapes, relative horizons `16,48`, branch strata `16,40,64,72`, terminal
tick `120`, and the scale-v2 `192/192` model took `69.433783` seconds with one
worker and `17.484045` seconds with eight workers: `3.971265x` faster. Both runs
produced scientific bundle digest
`6ece204747b7040a20a540292a1560423a468f2dd3198130c120082ac74f9763`.
The parallel run truthfully reports `4,280` actual simulator ticks versus
`2,978` canonical ticks (`1.437206x` work) because each process reconstructs
the source prefix and legacy rows; it still wins substantially in wall time.
The serialized result is
`docs/research/artifacts/recurrent-counterfactual-tape-benchmark-4032f7b6a4b4.json`
with SHA-256
`7ee0f67a031ce07ade209edd1840fa1eddf96dbc97b006ea172b44f463d3c88f`.
This result justifies CPU tape parallelism for retained exact audits. It does
not measure the proposed batched rollout engine, does not make the GPU useful
for the current Python simulator, and is not a scientific learning result.

## One campaign, not another diagnostic staircase

This should be managed as one versioned program with three engineering
milestones, not dozens of scientific claims. First, land the batched rollout
contract and prove exact parity on small frozen worlds. Second, land the
structured actor, inherited adapters, distinguishable opaque signal channels,
league islands and archive. Third, run one predeclared campaign long enough for
many biological generations and millions of agent decisions, with periodic
checkpoint snapshots but no mid-run retargeting toward whichever fixture looks
weak.

The campaign succeeds operationally when it has no heuristic action source,
selects only legal actions, replays frozen checkpoints exactly, preserves
lineage/controller inheritance, saturates the GPU inference batches, and
produces complete CPU-verifiable artifacts. Its scientific evidence is not one
score. It should report fresh-world population persistence, lineage and species
turnover, phylogenetic depth, genotype and behaviour-archive coverage,
cross-world transfer, and whether behavioural novelty continues rather than
plateaus. Social outcomes should be measurements: conditional response to
different signal tokens, repeated non-random association, partner choice,
resource-flow asymmetry, reciprocity after prior interaction, division of
foraging roles, group persistence and conflict between groups. These measures
must not enter the reward or archive quality calculation.

Carrion is one small invariant alongside water, plants, predation and mating:
the action is legal under the right conditions, grants the declared resource,
and at least one replay-valid policy can survive after using it. A trained
population is allowed to ignore it, specialize in it, or abandon it. The
headline question becomes whether inherited neural behaviour and ecological
pressure keep generating viable, measurably different ways of living—and
whether any of those ways begin to rely on other organisms.

## Evidence boundary

Primary research supports recurrent/on-policy multi-agent learning, population
training, adaptive environment curricula, diversity archives, and batched or
compiled execution as useful components in tested domains. It also supplies
examples of cooperation emerging from selfish rewards when partner choice,
exchange and repeated encounter are possible. No cited paper proves that this
particular combination will create societies, that two-to-five million
parameters is optimal, or that a given number of simulated generations is
enough. Those are the central hypotheses of the proposed campaign. The most
important falsification is not failure on carrion; it is a long run in which
viability improves while heritable behavioural diversity and innovation
collapse to one policy family.
