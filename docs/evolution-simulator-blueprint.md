# Evolution Simulator Blueprint

## 1. Goal

Build an open-ended artificial life simulator where agents:

- start without pretrained world knowledge
- survive, reproduce, mutate, and diversify under environmental pressure
- improve both across generations and, later, within a lifetime
- can eventually develop communication, memory, tools, and world models
- produce emergent behavior that can be inspected both visually and through structured data
- may eventually interact with systems outside the simulator through explicit capability gates

The target is not a chat demo. The target is a persistent ecology that can support emergent behavior, speciation, and increasingly rich cognition.

The primary success criterion is not raw benchmark score. It is the appearance of increasingly rich, inspectable behavior in runs, replays, lineages, and metrics. The long-term ambition includes externalized agency, but only through controlled, sandboxed, auditable interfaces rather than unrestricted host access.

## 2. Non-Goals For The First Versions

The first versions should explicitly avoid:

- pretrained text LLMs
- English-first language behavior
- per-agent LoRA inheritance
- browser-authoritative simulation
- 3D physics
- unbounded online weight updates during every tick
- unrestricted host or desktop access

These all add complexity faster than they add useful emergence.

## 3. Core Design Principles

1. The world must create tradeoffs.
No real emergence happens in a world with one resource, one objective, and no scarcity.

2. The simulator is the source of truth.
Visualization observes the world; it does not own world state.

3. Keep inheritance channels separate.
Genetics, lifetime learning, and social transmission are different mechanisms and should remain distinct in the architecture.

4. Start with cheap, frequent control and sparse, expensive cognition.
Dense tick-by-tick control should use small recurrent policies. Richer planning should be optional and event-driven.

5. Design for replay and inspectability before scale.
If lineages, rewards, and world events are not inspectable, later scaling will hide failure modes instead of solving them.

6. External-world interaction must be mediated.
If agents later interact with the host machine or external tools, every capability must be permissioned, sandboxed, rate-limited, and logged.

## 4. Execution Model

The project should be executed in four layers:

1. `Foundation`
World plus observability. This layer creates the pressures, logs, replay tools, and analytics that make later intelligence meaningful and inspectable.

2. `Mind`
Learned controllers, lifetime adaptation, and later planning. This layer should only begin after the foundation is strong enough to measure whether intelligence actually improves behavior.

3. `Culture`
Memory, signaling, traces, public artifacts, imitation, and eventually richer symbolic behavior.

4. `Capability`
Controlled interaction outside the simulator through explicit, sandboxed, auditable interfaces.

This is the only active execution order for the project. Do not treat world work, observability work, and intelligence work as separate competing roadmaps.

### 4.1 Foundation

Foundation includes two inseparable tracks:

- `world`
- `observability`

Every world addition must ship with matching observability. If a new mechanic cannot be inspected, graphed, replayed, or attributed to species and lineages, it is not complete enough to keep.

Foundation observability is not only numerical. The project should support strong visual understanding of the world as it evolves. Important environmental state and ecological behavior should be visible with the eye, not only recoverable from logs or tables.

### 4.1.1 Hydrology And Refuge Semantics

Foundation observability should keep the following meanings stable across replay data, viewer panels, tests, and docs:

- `primary hydrology reason` = the tile's main hard-water source used for drinking semantics
- `hydrology support` = additional water-related support conditions present on the tile, such as shoreline adjacency, wetland substrate, or flooded support
- `global refuge score` = refuge score averaged across forest tiles
- `species refuge score` = refuge score averaged across tiles occupied by that species at the current frame

These meanings must not be merged into a single unlabeled count or chart.

### 4.1.2 Biotic Pressure Semantics

Foundation observability should keep the following meanings stable across replay data, viewer panels, tests, and docs:

- `hazard type` = the dominant current land-tile damage pressure, not a generic danger bucket
- `hazard level` = the normalized current severity of that pressure on the tile
- `trophic role` = a derived phenotype from inherited combat and diet-bias traits, not a hard-coded caste flag
- `carcass stats` = current carrion state in the world
- `combat stats` = attack, kill, and damage activity over a frame or over the full run

These meanings must stay distinct. Hazard pressure, trophic phenotype, carrion availability, and combat outcomes are related but not interchangeable.

### 4.2 Why Foundation Comes First

If the world is too thin:

- agents learn shallow hacks
- speciation is fragile or misleading
- memory and planning have little reason to emerge

If observability is too weak:

- we cannot distinguish luck from adaptation
- we cannot tell whether intelligence improved anything
- we cannot debug extinction, collapse, or convergence

### 4.3 Gate To Start Mind

`Mind v1` should start only when Foundation is complete enough that the following are true:

- the world contains multiple stable, competing ecological pressures rather than one dominant survival strategy
- species and lineages can be tracked through time, not just at a single frame
- resource pressure, terrain occupancy, reproduction success, and collapse events can be explained from logs and viewer tools
- replay and analytics make it possible to compare runs and understand why one population outperformed another
- the team can state what "smarter" would mean in the current world without relying on vague intuition

## 5. System Model

The project should be built as six major systems:

### 5.1 World Substrate

Owns:

- terrain and climate
- resource spawning and depletion
- hazards and combat
- time, weather, seasonality
- public artifacts and signals
- deterministic simulation ticks

### 5.2 Embodiment And Genome

Owns:

- body plan traits
- sensors and action capabilities
- metabolism and energy economics
- reproduction thresholds and mating rules
- mutation and crossover
- developmental mapping from genome to phenotype

### 5.3 Controller Or Mind

Owns:

- observation encoding
- recurrent hidden state
- action selection
- optional communication head
- later: external memory and sparse planner

### 5.4 Memory And Culture

Owns:

- episodic memory buffers
- persistent local artifacts
- public notes, traces, signals, maps
- imitation and teaching channels

### 5.5 Training And Evaluation

Owns:

- trajectory logging
- replay generation
- species and lineage metrics
- asynchronous policy updates
- checkpoint promotion and rollback

### 5.6 Capability Layer

Owns:

- tool and host capability registry
- permission and approval rules
- sandboxing and resource limits
- audit logs for external actions
- later: controlled interfaces to the desktop, filesystem, network, or attached services

## 6. Inheritance Channels

This distinction is foundational.

### 6.1 Genetic Inheritance

Inherited at birth:

- body parameters
- sensor parameters
- metabolism parameters
- controller initialization codes
- mutation rates or mutation style

### 6.2 Developmental Inheritance

Maps genotype to phenotype:

- genome -> body traits
- genome -> controller initialization offsets
- genome -> plasticity parameters

This is where compact genomes can influence large controllers without requiring full weight matrices to be stored directly in the genome.

### 6.3 Lifetime Adaptation

Not inherited directly:

- recurrent hidden state
- episodic memory
- plastic updates inside an episode
- learned working memory contents

### 6.4 Cultural Transmission

Indirectly inherited through the world:

- symbols
- trails
- notes
- public artifacts
- local conventions

This is the right place for "texting" or symbolic messaging later.

## 7. Recommended Technical Stack

Use different tools for different responsibilities.

### 7.1 Near-Term Stack

- simulation and training: Python
- ML: PyTorch
- multi-agent environment interface: PettingZoo-compatible wrapper
- dashboard and replay viewer: current Next.js app
- 2D rendering in browser: PixiJS
- analytics and run inspection: DuckDB + Parquet

### 7.2 Long-Term Stack

- headless simulation core: Rust
- trainer remains: Python + PyTorch
- browser UI remains: Next.js + PixiJS
- IPC between sim and UI: file-based replay first, service boundary later

### 7.3 Why This Split

- Python is still the fastest path for experimentation and ML iteration.
- Rust is the better long-term place for a deterministic, high-throughput simulator.
- The browser is ideal for visualization, controls, replay, lineage graphs, and run inspection.

### 7.4 Visualization Strategy

The rendering model does not have to match the simulation model.

- keep the simulation deterministic and authoritative
- allow rendering to be smoother and more natural-looking than the underlying grid
- prioritize a strong scientific inspection view first
- allow a later immersive renderer without changing simulator truth

Near-term visualization approach:

- keep `PixiJS` as the primary browser renderer for Foundation work
- use it for replay, overlays, diagnostics, heatmaps, species views, and dense agent rendering
- progressively make the world feel less visibly grid-based through shading, interpolation, blended terrain, and effects

Longer-term visualization approach:

- support a second optional `immersive view` later if the project benefits from richer presentation
- likely candidates are `Three.js` or `Babylon.js`
- this renderer must consume the same replay or simulator contracts rather than becoming the source of truth

The project should maintain two visualization modes over time:

1. `scientific view`
Exact, data-rich, explainable, optimized for debugging and ecological understanding.

2. `immersive view`
More cinematic and natural-looking, optimized for intuition and presentation.

## 8. Repo Shape In This Workspace

This repository is already a Next.js application. The simplest path is to keep the web app as the dashboard and add the simulator as a sibling subsystem.

Recommended structure:

```text
docs/
  evolution-simulator-blueprint.md

python/
  evolution_sim/
    __init__.py
    config/
    env/
      world.py
      terrain.py
      resources.py
      agents.py
      reproduction.py
      events.py
    genome/
      schema.py
      mutation.py
      crossover.py
      development.py
      species.py
    brains/
      encoder.py
      gru_policy.py
      action_heads.py
      memory.py
    training/
      rollout.py
      replay_buffer.py
      trainer.py
      evaluation.py
      checkpoints.py
    io/
      logging.py
      parquet_writer.py
      replay_writer.py
    cli/
      run_headless.py
      inspect_run.py

src/
  app/
    sim/
      page.tsx
      runs/
        [runId]/
          page.tsx
  components/
    sim/
      WorldCanvas.tsx
      Timeline.tsx
      AgentInspector.tsx
      SpeciesPanel.tsx
      RunControls.tsx
      MetricsCharts.tsx
  server/
    api/
      routers/
        sim.ts
```

Later, if the Python simulator becomes the bottleneck, replace `python/evolution_sim/env` with a Rust engine while keeping the UI contract stable.

## 9. First Concrete World Design

Choose a world that is simple enough to build but rich enough to support niches.

### 9.1 World Type

For `v1`, use a deterministic 2D grid world with fixed ticks.

Reason:

- easier reproducibility
- easier debugging
- simpler replay
- easier action masking
- sufficient for early emergence

The world can become continuous later if that becomes necessary.

### 9.2 Spatial Features

Each tile should carry channels such as:

- terrain type
- traversability cost
- food amount
- water amount
- hazard level
- shelter value
- scent or trace intensity

### 9.3 Environmental Pressure

The world should include:

- multiple resource types
- local depletion and regrowth
- terrain-biased movement cost
- at least one hazard or predator-prey asymmetry
- seasonality or weather shifts

This is the minimum needed to make niche differentiation likely.

## 10. First Agent Design

The first agent should be intentionally simple but evolvable.

### 10.1 Body State

Each agent tracks:

- position
- orientation
- energy
- hydration
- health
- age
- reproductive readiness
- inventory or held item slot later, not in `v1`

### 10.2 Sensors

Start with:

- egocentric local patch view
- scalar self-state
- nearby agent presence
- nearby resource intensities
- last received symbol token, if any

Avoid global map access.

### 10.3 Action Space

Use multi-head discrete actions:

- movement: stay, north, south, east, west
- interaction: none, eat, drink, attack, mate
- signaling: emit symbol or no-op

Action masking should prevent impossible actions from polluting early learning.

## 11. Genome Schema

Do not begin with "the genome is the full neural network." That is too large and brittle.

Use a compact genome that controls phenotype and controller initialization.

### 11.1 Genome Groups

- metabolism genes
- morphology genes
- sensor genes
- communication genes
- controller initialization genes
- plasticity genes
- mutation genes

### 11.2 Example Traits

- max energy
- move cost multiplier
- attack cost multiplier
- food preference weights
- vision radius
- signal vocabulary size
- reproduction threshold
- gestation delay
- controller latent code
- learning rate multiplier for later plasticity

### 11.3 Development Mapping

At birth:

- genome traits instantiate body and sensors
- controller latent code generates low-rank offsets or initialization values for the policy network
- recurrent hidden state starts empty
- episodic memory starts empty

This is cleaner than inheriting and merging full per-agent LoRA adapters.

## 12. Brain Design

### 12.1 Phase 1 Brain

Use:

- observation encoder
- small GRU core
- action heads
- optional value head for later training

This should be tiny and fast.

### 12.2 Why Not A Text LLM

Early agents need:

- sensorimotor competence
- memory over short horizons
- robust cheap control

They do not need internet text priors or a chat interface.

### 12.3 Upgrade Path

Later upgrades can be:

- GRU -> LSTM
- GRU -> tiny causal transformer over event tokens
- add external memory
- add event-driven planner

The upgrade should happen only after the ecology is already producing interesting pressure.

## 13. Learning Strategy

Use staged learning, not one giant online RLHF loop.

### 13.1 Phase 1: Evolution First

The first version should rely primarily on:

- mutation
- crossover
- selection by survival and reproduction

No online gradient learning is required yet.

This phase is still part of `Foundation`, because the purpose is to establish ecology, inheritance, and observability before adding learned minds.

### 13.2 Phase 2: Mind v1

After Foundation is complete enough:

- replace the heuristic controller with a tiny learned recurrent controller
- keep world rules and observability stable while introducing the first learned mind
- define success in ecological terms, not benchmark terms

This phase begins only after the `Gate To Start Mind` above is satisfied.

### 13.3 Phase 3: Species-Level Training

After the ecology is stable:

- collect trajectories by species or lineage
- train shared species checkpoints asynchronously
- use checkpoints only for new births or scheduled refreshes

This is far easier to manage than per-agent online weight updates.

### 13.4 Phase 4: Lifetime Adaptation

Add one of:

- plastic memory updates
- local reinforcement learning within limited windows
- curiosity or novelty objectives

This should remain bounded and observable.

### 13.5 Phase 5: Preference Shaping

Only after the above works:

- preference ranking over behaviors
- DPO-like shaping of a shared checkpoint
- optional GRPO-like online objective on selected species or roles

This should be used to steer behavior, not replace ecological pressure.

## 14. Communication And Culture

Communication should emerge because it is useful, not because we hard-code dialogue.

### 14.1 Early Communication

Start with:

- a tiny symbol vocabulary
- local broadcast radius
- no built-in semantics

### 14.2 Mid-Stage Communication

Add:

- persistent traces on tiles
- local markers
- public boards or note objects
- imitation triggers

### 14.3 Late Communication

Only later consider:

- event-token sequence models
- compositional symbol systems
- translation layers from internal symbols to human-readable summaries

If you go straight to English, you lose the "learn only inside the world" constraint.

## 15. Logging, Replay, And Analytics

This is a first-class subsystem.

### 15.1 Per-Tick Logging

Store:

- seed
- tick
- world config
- births and deaths
- actions
- rewards or fitness signals
- resource changes
- attacks, mating events, signals
- species assignments

### 15.2 File Strategy

Use:

- Parquet for structured logs
- compressed JSON or binary blobs for replay snapshots
- DuckDB for interactive analysis

### 15.3 Required Tools

The first UI should support:

- run playback
- tick scrubbing
- agent inspector
- lineage tree
- visual overlays for world state such as resources, fields, and species occupancy
- enough visual fidelity that important ecological changes can be understood by eye

## 16. Immediate Foundation Queue

The next active work should stay inside `Foundation`.

### 16.1 World Expansion

Add environmental state that creates richer but still interpretable pressure:

- regional fertility, moisture, and heat fields
- resource growth driven by those fields, not only terrain labels
- more local variation inside the same terrain class
- state-driven disturbances only after those fields exist
- health, injury, hazards, carcasses, and trophic asymmetry once the abiotic world is stable enough to explain

### 16.2 Observability Expansion

Add tools that explain the world rather than only replay it:

- time-series charts for births, deaths, alive population, and species counts
- terrain occupancy by species and lineage
- hydration stress, starvation pressure, and reproduction success by species
- hazard exposure, injury pressure, carcass use, and attack outcomes by species
- extinction and collapse events with visible timing
- trait drift over time
- richer visual overlays for environmental state and ecological pressure

### 16.3 Definition Of Done For This Queue

This queue is complete when:

- world mechanics create visibly different ecological regions or conditions inside a run
- those conditions are measurable and visible in the viewer or logs
- we can explain why a species is growing or collapsing using recorded evidence rather than guesswork
- the next intelligence milestone would enter a world that rewards memory, adaptation, and niche-aware behavior

## 17. Metrics

Do not rely on "it looks interesting" as the main success criterion.

Track:

- survival time
- reproduction count
- lineage persistence
- species count
- genetic diversity
- niche occupancy
- resource efficiency
- communication usage rate
- cultural artifact usage rate
- run-to-run stability under fixed seeds

The system should also support qualitative review of:

- novel survival strategies
- niche differentiation
- emergent communication patterns
- tool-use attempts and failures
- transitions from purely in-world behavior to mediated external action

## 18. Stage Plan

### Stage A: Foundation Baseline

Purpose:

- establish deterministic ecology
- establish replay and inspection tools
- establish initial species and lineage visibility

Status:

- completed enough for the current baseline

Delivered so far:

- deterministic world loop
- food, water, terrain, metabolism, reproduction, death
- genome mutation
- replay output and browser viewer
- species clustering and per-agent inspection

### Stage B: Foundation Expansion

Purpose:

- make the world richer without making it opaque
- make ecological outcomes explainable through viewer and analytics

Status:

- active

Deliverables:

- environmental field layer such as fertility, moisture, and heat
- resource growth driven by those fields
- time-series analytics in the viewer
- species occupancy and stress diagnostics
- collapse and extinction visibility
- improved scientific visualization of world state, including field overlays and less visibly grid-bound rendering
- biotic-pressure layer with health, injury, hazards, carcasses, trophic-role derivation, diet-gated predation, and matching viewer surfaces

Exit condition:

- we can explain population growth, collapse, and niche dominance from recorded evidence

### Stage C: Mind v1

Purpose:

- replace the heuristic controller with the first learned recurrent controller

Prerequisite:

- `Stage B: Foundation Expansion` must be complete enough to satisfy the `Gate To Start Mind`

Exit condition:

- the learned controller improves ecologically meaningful behavior in the existing world

### Stage D: Culture

Purpose:

- add signaling, traces, public artifacts, and memory-bearing world objects

Exit condition:

- information passed through the world becomes instrumentally useful in repeated scenarios

### Stage E: Capability

Purpose:

- add explicit external-world interfaces under strict mediation

Exit condition:

- narrowly scoped external action works through sandboxed, auditable capabilities only

## 19. Decision Record

The following are the recommended decisions unless evidence later forces a change:

- `world authority`: headless simulator, not browser
- `first world`: 2D grid, deterministic ticks
- `first learning`: evolution before gradient training
- `first brain`: GRU, not text transformer
- `first UI`: replay and inspection, not live multiplayer
- `visualization strategy`: PixiJS scientific view first, optional immersive renderer later on the same replay contract
- `data backbone`: Parquet + DuckDB
- `future scale path`: Python first, Rust core later
- `host interaction model`: mediated capability layer, never unrestricted desktop escape

## 20. Immediate Next Tasks

The next implementation slice should stay entirely inside `Stage B: Foundation Expansion`.

1. Add cross-run evaluation.
The simulator should compare seeds directly so species dominance, hazard pressure, trophic behavior, and collapse can be explained across runs rather than only within a single replay.

2. Run the Foundation gate pass.
Do long seeded sweeps, tighten replay-size and analytics discipline, and decide explicitly whether the current world is rich and legible enough to justify `Mind v1`.

Do not start the learned controller until these tasks are in place.

## 21. Open Questions To Resolve Early

- Should `v1` require both food and water, or only food?
- Should reproduction be sexual from the start, or begin asexual and add mating later?
- Should species be defined by genetic clustering, reproductive compatibility, or both?
- How much of the controller should be genetically specified versus trained?
- What is the first allowed external capability: file read, note writing, browser-like interaction, or none?
