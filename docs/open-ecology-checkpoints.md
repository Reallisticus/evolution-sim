# Persistent-island checkpoint container

`open_ecology_checkpoint.py` defines the minimal non-executable container for a
future persistent-island resume path. It does not serialize
`SimulationWorld`, a recurrent policy, an RNG object, or an evidence writer
itself. Those adapters remain separate work. Instead, the container forces
each adapter to supply versioned canonical JSON state and binds every state,
the exact source commit, configuration contract, seed contract, tick, and
generation identity with SHA-256.

A complete container has seven independently versioned state components:
world state, environment RNG state, frozen recurrent policy state, public
feedback/history, policy-sampling RNG state, the genome-population snapshot,
and evidence-writer continuation state. Callers must provide every argument
explicitly. A missing component is represented by a strict absent envelope,
never by an empty object. The computed `restartable` gate is true only when all
seven envelopes are present, non-empty, type-valid, and digest-valid. A partial
observational snapshot remains loadable for inspection, but
`require_restartable=True` rejects it.

```python
from evolution_sim.io.open_ecology_checkpoint import (
    VersionedCheckpointState,
    load_open_ecology_checkpoint,
    write_open_ecology_checkpoint,
)

state = VersionedCheckpointState(
    schema_version="simulation_world_state_adapter_v1",
    payload=complete_world_state,
)

checkpoint = write_open_ecology_checkpoint(
    "output/open-ecology/island-03-tick-1250.json",
    source_git_sha=exact_clean_source_sha,
    config_contract=versioned_config_contract,
    seed_contract=versioned_seed_contract,
    run_generation_id="generation-0009",
    island_id="island-03",
    generation_index=9,
    tick=1250,
    world_state=state,
    environment_rng_state=versioned_environment_rng_state,
    recurrent_policy_state=versioned_frozen_policy_state,
    public_feedback_history=versioned_public_history,
    sampling_rng_state=versioned_sampling_rng_state,
    genome_population_snapshot=versioned_genome_population,
    evidence_writer_continuation_state=versioned_writer_state,
)

validated = load_open_ecology_checkpoint(
    "output/open-ecology/island-03-tick-1250.json",
    expected_source_git_sha=exact_clean_source_sha,
    expected_generation_identity_sha256=checkpoint[
        "generation_identity"
    ]["identity_sha256"],
    require_restartable=True,
)
```

The file format is canonical UTF-8 JSON with one trailing newline. It rejects
duplicate keys, non-finite values, unknown container/envelope keys, stale
contracts, digest mismatches, non-regular files, and symbolic-link loads. It
never invokes pickle, object hooks that construct application classes, imports,
or executable payloads. Writes use a same-directory temporary file, file
`fsync`, and atomic replacement, so validation, size-ceiling, or replacement
failure preserves the previous complete destination. The default ceiling is
256 MiB and is configurable at both write and load.

The generation identity binds the generation ID, island, generation index,
source commit, configuration-contract digest, and seed-contract digest. Its
digest is validated independently of the full checkpoint digest, and a loader
may pin the expected identity. This prevents a checkpoint from silently moving
between source/configuration/seed generations even if somebody recomputes only
the outer digest.

The name `restartable` currently means **container-complete**, not
simulation-proven. No production run should resume from this format until
adapters exist for every state component and a continuation-equivalence test
shows that an uninterrupted run and a save/load continuation produce the same
subsequent world state, actions, RNG draws, policy hidden state, genomes, and
evidence-stream digest. Until that proof exists, the format is a guarded
substrate and its truth gate must not be reported as end-to-end resume proof.
