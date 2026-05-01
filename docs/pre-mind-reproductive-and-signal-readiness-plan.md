# Pre-Mind Reproductive And Signal Readiness Plan

Date: 2026-04-30
Status: architecture plan with initial Foundation scaffold slices implemented

This document defines the Foundation cleanup needed before Mind v1 around
reproduction, signals, communication, and future learned-controller contracts.
It is intentionally pre-Mind work. The simulator should not start learned
controller training while these world semantics are still fluid.

## Executive Decision

Mind v1 remains blocked until Foundation can support the intended biological
world without later rewriting the observation/action/reward/replay contracts.

The next Foundation direction is:

- keep the validated deterministic Foundation as the baseline while scaffolding
  new contracts;
- add a general signal substrate with intensity, range, duration, decay, cost,
  and policy-facing opaque emission slots;
- add reproductive genome structure and live reproductive groups;
- evolve from current single-parent asexual reproduction toward facultative
  same-group sexed recombination;
- later add X/Y/Z-like reproductive expression, multi-offspring strategies, and
  rare gated hybridization;
- keep all policy-facing contracts numeric and opaque, with human labels only in
  replay/debug metadata;
- keep the default heuristic from using communication semantics or hardcoded
  signal chasing.

The first implementation slice after this document should be scaffolding with
unchanged behavior and unchanged replay goldens where possible.

Implemented slices as of 2026-04-30:

- behavior-preserving action/signal/reproductive-genome scaffolding, with future
  `mate` and opaque signal slots present but masked;
- Stage 0 live reproductive-group registry, recorded in summary and replay
  catalog surfaces while behavior remains single-parent asexual;
- grouped genome recombination contract and deterministic helper for future
  two-parent births, kept separate from current mutation logic.
- Stage 1 same-group facultative sexed reproduction, trait-gated and local, with
  grouped recombination, both-parent cost/cooldown, dual-parent replay metadata,
  and asexual fallback when no valid partner exists.
- Signal contract v2 profile/provenance metadata for the reproductive readiness
  field, including debug-only source agent IDs, profile IDs, source positions,
  radius, duration, decay, and energy-cost data in full replay frames.

Not implemented yet: default/emergent communication use, X/Y/Z-like expression,
hybridization, multi-offspring strategies, or learned state inheritance.

## Non-Goals

These are explicitly out of scope for this Foundation slice:

- pretrained text LLMs;
- human language tokens;
- hand-authored meanings such as "food", "danger", or "mate" in policy-visible
  communication;
- heuristic mate chasing based on pheromone labels;
- hidden population rescue rules that preserve species after biology would
  otherwise let them collapse;
- direct policy access to human-readable action names, agent IDs, species names,
  lineage labels, or debug semantics.

## Mind Direction

The intended Mind is from-scratch, simulation-native, and LLM-like in method,
not in training data.

Recommended long-term direction:

- sequence modeling over trajectories;
- attention and memory over numeric observation/action/reward/event histories;
- world-model prediction of future observations, rewards, and survival pressure;
- communication tokens learned inside the simulator;
- inherited learned state later, after Foundation has stable genetics and
  reproduction contracts.

This maps to the method family behind transformer/world-model approaches, but
the data source is this simulator only. No pretrained human text model should be
used as an agent brain.

## Policy Contract

The policy should see an opaque numeric control surface:

- action IDs, masks, and numeric arguments;
- encoded observations with bounded numeric fields;
- rewards and outcomes with versioned numeric contracts.

The policy must not see:

- action names such as `eat`, `drink`, `mate`, or `signal`;
- species/debug labels;
- lineage names;
- reproductive group names;
- communication token meanings;
- pheromone names or mate-target hints.

Human-readable labels can still exist in replay, viewer, logs, and diagnostics.
Those labels are for auditability, not for agent cognition.

## Action Contract

Foundation should add an explicit `ActionContract` layer:

- policy-facing action IDs are opaque and stable;
- world-facing resolvers map action IDs to physical effects;
- masks expose availability without explaining semantics;
- invalid reasons are recorded for humans and rewards, not exposed as language to
  the policy;
- future slots can exist from tick zero as masked or no-op capabilities.

The current base action set remains behaviorally valid for now, but conceptually
it should be treated as opaque slots over physical affordances, not as agent
knowledge of named actions.

Near-term reserved slots:

- mate attempt;
- communication emission profiles.

Do not reserve every imaginable future social/building/tool action now. Reserve
only the slots needed to avoid contract churn for the already planned
reproductive and signal work.

## Mate Action Semantics

The `mate` action is a future attempt slot, not a reproduction command.

When it becomes visible to an eligible learned policy:

- biology always validates the attempt;
- both parents must be alive, local enough, ready enough, compatible enough, and
  able to pay costs;
- space/population constraints still apply;
- failures emit structured invalid/block reasons for replay and reward;
- no policy can force offspring creation by choosing the action.

The first sexed-reproduction implementation does not need to expose `mate` to
the default heuristic. Stage 1 can begin through lifecycle proximity and
biological readiness while the action slot remains reserved for the learned
controller era.

## Signal Substrate

Foundation should implement one general signal substrate that can support both
biological pheromones and future intentional communication.

Each signal emission should have:

- signal token/type ID;
- source position;
- source agent ID in replay/debug metadata only;
- intensity;
- range or diffusion radius;
- duration;
- decay curve;
- energy/metabolic cost;
- optional noise/clipping;
- policy-visible numeric local sensing fields.

Internally, the substrate can be continuous. Policy-facing emission control
should begin as discrete profiles so the action contract stays manageable.

Current implementation status: the live reproductive readiness signal uses
`foundation_signal_contract_v2`. Full replay frames carry per-tick debug
emission metadata with source agent ID, profile ID, source position, intensity,
radius, duration, decay, and energy cost. Policy observations still expose only
opaque numeric local signal values.

Signal physics must be deterministic for a fixed seed.

## Pheromones

Reproductive pheromones are biological emissions, not agent-authored language.

Initial design:

- automatically emitted by eligible organisms as a function of reproductive
  readiness and reproductive traits;
- intensity, range, duration, and decay are configured and later heritable;
- emissions cost energy or metabolism;
- weak baseline sensing exists so the substrate is observable;
- sensitivity can evolve;
- no hardcoded heuristic movement toward pheromones;
- no semantic label is exposed to the Mind.

Current implementation status: the reproductive-readiness field is live on the
default Foundation path. It is emitted automatically by biologically ready
agents, diffuses deterministically across land tiles, decays by config, can
charge a configured energy cost, and is surfaced as opaque numeric observation
data plus debug-only replay provenance. It does not drive scripted movement or
unmask any policy action.

Pheromones should primarily signal readiness. A small self-stabilizing component
can be allowed for local scarcity, reproductive-group scarcity, or role
imbalance, but it must not become a scripted "find mate" behavior.

Predators or competitors can later learn to exploit signal fields if their
observations expose compatible numeric traces. Foundation should not script that
interpretation.

## Communication Signals

Communication uses the same substrate as pheromones but has stricter semantics:
the simulator must not assign meanings.

Initial reserved profile:

- four opaque tokens;
- two emission profiles per token;
- eight reserved emission slots total;
- profile controls intensity/range/duration/cost;
- token meanings are learned, if they ever become useful.

Default behavior:

- communication substrate enabled;
- individual emission trait-gated and initially rare;
- weak sensing available through numeric observation fields;
- default heuristic never emits communication tokens;
- no gate should require a specific token meaning.

This lets communication become a learned/evolved phenomenon later without
requiring a replay/action schema rewrite.

Current implementation status: communication profiles are reserved in the signal
contract as opaque token/profile slots and have a separate decay-rate config
field. The observation and trajectory contracts are generated from the active
`SignalConfig`, and reserved action keys/masks now use the same configured
token/profile counts. Replay and streaming trajectory metadata therefore cannot
declare a different communication capacity than the policy-facing action
surface. An opt-in `communication_signal_emission_enabled` path can unmask
trait-gated opaque signal actions and emit numeric communication fields with
debug-only provenance; the default Foundation config keeps communication
emission disabled, the token/profile counts are explicitly bounded, and the
default heuristic never emits communication tokens.

## Genome Structure

The implementation should move toward structured genome groups now, even while
Stage 0 behavior remains clone-and-mutate.

Recommended groups:

- core physiology;
- metabolism;
- combat, injury, and healing;
- trophic strategy;
- habitat affinity;
- reproduction;
- signal emission/sensitivity;
- future Mind inheritance placeholder.

Stage 0 can keep current effective behavior: one parent mutates a genome and
creates a child. The representation should still be compatible with later
chromosome-like grouped recombination so Stage 1 does not require a rewrite.

Current implementation note: `python/evolution_sim/genome/recombination.py`
groups every active scalar genome field exactly once and includes the inert
reproductive module. Signal-specific and Mind-inheritance groups should be added
only when those heritable fields exist, rather than represented as empty groups.

## Reproductive Genome Module

Reproduction should have its own heritable module with distinct mutation bounds.

Initial and planned traits:

- sexual reproduction drive;
- recombination affinity;
- role differentiation drive;
- sex expression bias;
- sex plasticity;
- hybridization tolerance;
- fecundity or offspring-count potential;
- recombination/crossover controls;
- signal emission intensity/range/duration tendencies;
- signal sensitivity tendencies.

Mutation and recombination are separate concepts:

- `mutation_scale` controls how traits change during inheritance;
- recombination controls how parental genome groups are selected or crossed.

Do not overload mutation scale to mean sexual recombination rate.

## Reproductive Groups

Reproductive group is a live biology layer, separate from replay species.

Definitions:

- lineage: genealogy/provenance;
- species/replay taxonomy: durable analysis and viewer classification;
- reproductive group: live compatibility and reproductive-stage system.

Group rules:

- inherited by default;
- periodically reevaluated through deterministic similarity/stage criteria;
- splits can happen when subgroups become persistently incompatible;
- merges are rare and explicit;
- hybridization creates hybrid provenance before it creates a new durable group;
- stage promotion uses rolling multi-generation evidence;
- stage demotion is possible but hysteresis-protected.

The reproductive group layer is allowed to affect live behavior. Replay taxonomy
must not silently drive mating compatibility.

## Reproductive Stages

Stage labels are human-readable summaries over capability flags. The simulator
should store explicit capability/stage state, not infer everything from labels.

### Stage 0: Asexual Clone And Mutation

This is the current behavior:

- one parent;
- child genome is parent genome plus mutation;
- child inherits parent lineage;
- parent pays energy and cooldown;
- no mate selection;
- no recombination.

This remains the starting condition.

### Stage 1: Facultative Same-Group Recombination

Sexed reproduction first appears as a mutation-enabled capability:

- rare by default;
- gated by reproductive genome traits;
- same reproductive group or same species equivalent only;
- small radius-based mating;
- lifecycle proximity is enough for first implementation;
- both parents pay energy/cooldown costs;
- compatibility is continuous, not binary;
- asexual fallback remains available early;
- sex's benefit is recombination, not free offspring vitality.

Compatibility should have a sweet spot:

- too genetically/recent-ancestry close: inbreeding penalty;
- moderately compatible: healthiest recombination;
- too distant: outbreeding/incompatibility penalty.

Inbreeding penalty should use both genetic similarity and shallow recent ancestry
with a fixed depth. The penalty should affect offspring stability, fertility, or
genome health more than immediate attempt success.

### Stage 2: Proto-Role Differentiation

Groups can evolve role differentiation before fixed X/Y/Z-like expression:

- role tendencies are continuous;
- compatibility uses complementary reproductive expression;
- no hard male/female labels are required;
- asexual fallback can weaken as group-level sexed reproduction becomes stable.

### Stage 3: X/Y/Z-Like Expression

Later-stage reproductive groups can express roles:

- X and Y are fixed for life;
- Z is heritable reproductive plasticity;
- Z expresses X-like or Y-like behavior at runtime based on local/group imbalance;
- Z has strong benefits in skewed or sparse populations;
- Z has meaningful costs so it does not dominate every group.

Z costs should include some combination of:

- higher energy upkeep;
- longer cooldown;
- reduced specialization;
- lower offspring stability;
- frequency-dependent penalty when too common.

Supported pairings:

- X/Y;
- X/Z-as-Y;
- Y/Z-as-X;
- Z/Z only with extra penalty.

X/Y/Z should not directly hardcode behavioral roles like predator, caregiver, or
forager. Any behavioral difference should come from evolved physiology,
compatibility, cost, and learned policy later.

### Stage 4: Hybridization And Near-Obligate Sex

Hybridization is allowed, but late and highly gated:

- reproductive group stage must allow it;
- both individuals need sufficient hybridization tolerance;
- distance penalties must be strong;
- child provenance records both groups and both parent lineages;
- viability/fertility penalties scale with distance;
- repeated stable hybrid-to-hybrid reproduction can promote a durable hybrid
  group/species;
- backcrossing can support introgression.

Near-obligate sex can also appear late:

- asexual fallback weakens as the group demonstrates stable sexed reproduction;
- eventually solo reproduction can become impossible for that group;
- extinction is allowed if the group evolves itself into a dead end.

There should be no hidden asexual rescue rule for mature sexual groups.

## Multi-Offspring Strategy

More than one child per reproductive event should be supported, but only as a
rare, high-criteria strategy at first.

Rules:

- enable for sexed reproduction before asexual reproduction;
- require heritable fecundity potential and high parent readiness;
- increase total parent cost;
- lower per-child starting quality or stability;
- expose offspring-count metrics and costs in replay/evaluation.

The goal is a tradeoff, not free population growth.

## Lineage And Parentage

Sexed reproduction requires dual-parent provenance.

Rules:

- every sexual child records both parent IDs;
- every sexual child records both parent lineage IDs;
- same-lineage sex can keep the lineage ID while recording dual parentage;
- cross-lineage sex creates a new lineage ID with parent-lineage provenance;
- hybridization records reproductive group provenance separately;
- global birth counts increment once per child;
- both parents receive reproductive success credit.

Full ancestry proportions are deferred until hybridization and introgression
need them. Stage 1 only needs shallow recent ancestry and parent-lineage
provenance.

## Events, Replay, And Metrics

Future reproductive events should be versioned before behavior changes land.

Required fields for sexed or hybrid births:

- mode: `asexual`, `sexual`, or `hybrid`;
- child ID;
- parent IDs;
- parent lineage IDs;
- parent reproductive group IDs;
- child lineage ID;
- child reproductive group ID;
- compatibility score;
- inbreeding score;
- outbreeding distance score;
- parent costs;
- offspring count for the event;
- blocked/invalid reason when no birth occurs;
- inert Mind inheritance metadata placeholder.

Metrics should distinguish:

- birth count;
- parent credit count;
- reproduction attempts;
- blocked attempts by reason;
- local mate opportunities;
- compatibility distributions;
- inbreeding/outbreeding penalty distributions;
- per-group sexed/asexual/hybrid birth shares;
- X/Y/Z role ratios once Stage 3 exists;
- extinction/collapse after loss of asexual fallback.

## Learned State Inheritance Placeholder

Learned state inheritance means a future child can inherit some compressed
structure learned by parents during life, not only fixed genome traits.

For now, Foundation should reserve inert metadata only:

- no learned controller;
- no inherited model weights;
- no behavior change;
- no policy-visible parent memories.

The placeholder exists so reproductive event schemas, lineage provenance, and
future Mind data formats have a stable place to attach this later.

When implemented in Mind work, the inherited state must be bounded, compressed,
versioned, and auditable. It should not become an unlimited memory copy from
parent to child.

## Default Heuristic Boundary

The current observation-only heuristic is a compatibility baseline, not the
future Mind.

It may:

- use numeric observation fields;
- remain deterministic;
- keep the simulator alive enough for Foundation gates.

It must not:

- emit communication tokens;
- chase pheromones by hardcoded meaning;
- choose mates by reading debug labels;
- use species names or reproductive group labels as semantic input;
- learn or update state across runs.

If a behavior would only make sense because a human named a signal, the heuristic
should not do it.

## Validation Requirements

Each behavior slice must come with tests, metrics, and gate coverage.

Scaffolding slice:

- action contract has reserved slots and stable metadata;
- slots are masked/no-op until biologically enabled;
- signal config validates bounds;
- genome/reproduction module serializes deterministically;
- zero/no-op observation fields do not change behavior;
- goldens remain unchanged if possible.

Stage 1 sex slice:

- same seed is deterministic;
- asexual behavior still works at Stage 0;
- sexual birth requires both parents and compatibility;
- both parents pay configured costs;
- blocked reasons cover no partner, incompatible partner, partner not ready,
  crowding, and population cap;
- inbreeding penalty affects offspring metrics;
- sexed births are rare but observable in default Foundation gates;
- no interspecies/cross-group breeding occurs before hybridization gates.

Pheromone slice:

- emissions have deterministic intensity/range/duration/decay;
- energy cost is charged;
- local sensing fields are numeric and bounded;
- heuristic does not hardcode movement toward pheromones;
- replay/viewer can inspect signal fields for humans.

Communication slice:

- opaque tokens and profiles exist;
- emission is trait-gated;
- heuristic emits none;
- observation fields expose numeric traces only;
- no gate asserts human meaning.

X/Y/Z and hybridization slices:

- stage thresholds use rolling evidence and hysteresis;
- Z does not dominate under neutral conditions;
- hybridization is absent before late-stage gates;
- hybrid children carry penalties/provenance;
- mature groups can go extinct after losing asexual fallback.

## Implementation Order

1. Documentation and audit alignment. Implemented as this plan plus the
   readiness-audit/readme updates.
2. Behavior-preserving scaffolding. Implemented:
   - `ActionContract`;
   - reserved `mate` and communication slots;
   - signal config and no-op substrate types;
   - reproductive genome module with inert defaults;
   - Mind inheritance placeholder metadata;
   - zero/no-op observation fields.
3. Reproductive groups and grouped genome recombination helpers. Implemented as
   Stage 0 registry plus two-parent grouped recombination helper.
4. Stage 1 same-group sexed reproduction, default Foundation path. Implemented
   as a rare facultative path: individuals can unlock it through reproductive
   gene mutation, both local same-group parents must be biologically ready, both
   pay cost/cooldown, and current asexual reproduction remains the fallback.
5. Reproductive runtime cleanup before signal substrate. Implemented: the
   reproductive runtime owns eligibility, child construction, mate selection
   orchestration, accounting, readiness reporting, and the tick-level
   reproductive signaling/birth phase behind a narrow boundary, without
   changing replay/golden semantics.
6. Pheromone substrate with readiness emissions and numeric sensing.
   Implemented for the reproductive-readiness path: biologically ready agents
   emit deterministic numeric fields with configured intensity, radius, duration,
   decay, and energy-cost hooks. Observations, replay frames, summary metrics,
   analytics, and the viewer expose the field. Communication tokens remain
   reserved and opaque.
7. Communication substrate with trait-gated opaque emission. Implemented as an
   opt-in config path: communication actions remain disabled by default, but can
   be unmasked by `SignalConfig` plus agent signal-emission traits and emit
   opaque numeric fields without simulator-assigned token meanings.
8. X/Y/Z stage system.
9. Rare multi-offspring sexed strategy.
10. Late hybridization/introgression.
11. Mind v1 sequence/world-model work only after Foundation gates stabilize.

The user direction is to put this on the default Foundation path, not behind a
separate experimental profile. That means each behavior slice must update gates
and goldens intentionally when behavior changes.

## Research Anchors

These are anchors for design reasoning, not requirements to copy biology
literally:

- [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
  supports treating trajectories as sequence-modeling data.
- [World Models](https://arxiv.org/abs/1803.10122) supports learning compact
  predictive environment representations before or alongside control.
- [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676)
  supports emergent communication protocols without hardcoded human meanings.
- [Experimental tests of the adaptive significance of sexual recombination](https://www.nature.com/articles/nrg760)
  summarizes long-term adaptive advantages of recombination.
- [Search, encounter rates, and the evolution of anisogamy](https://pmc.ncbi.nlm.nih.gov/articles/PMC391862/)
  supports modeling sex-role differentiation as an evolutionary transition
  rather than a starting assumption.
- [Mating-type switching in yeast](https://pmc.ncbi.nlm.nih.gov/articles/PMC3338269/)
  is a useful biological anchor for role plasticity without treating fixed
  binary sex as universal from the beginning.
- [Adaptive introgression: a plant perspective](https://pmc.ncbi.nlm.nih.gov/articles/PMC5897607/)
  supports late, gated hybridization/introgression as a source of variation.
- [The genomic consequences of hybridization](https://pmc.ncbi.nlm.nih.gov/articles/PMC8337078/)
  supports treating hybridization as high-variance and penalty-bearing rather
  than automatically beneficial.

## Readiness To Move On

The project is ready to move from release-gate hardening into this pre-Mind
readiness cleanup only if the latest validated claims still hold:

- full tests pass;
- full goldens pass;
- release gate passes without blockers or warnings;
- benchmark output is coherent;
- viewer smoke passes;
- compact replay serialization and streaming trajectory behavior remain within
  expected cost ranges.

It is not ready for Mind v1 until the staged reproductive/signal Foundation work
above is implemented, measured, represented in replay/trajectory contracts, and
covered by gates.
