# Persistent-island aggregate generation commits

`open_ecology_aggregate_commit.py` closes the storage-level atomicity gap
between the seven-envelope checkpoint container and the rotating evidence
writer. It is a substrate for a future persistent-island runner, not proof that
`SimulationWorld`, recurrent-policy, RNG, genome, or evidence adapters can yet
resume an uninterrupted world exactly.

The store has one cross-process writer lock, an immutable `generations/`
directory, and one canonical `CURRENT` pointer. Each generation contains
exactly:

- `checkpoint.json`, a complete and digest-valid seven-envelope checkpoint;
- `evidence-manifest.json`, the canonical snapshot of the externally verified
  rotating-writer manifest named by the checkpoint continuation state;
- `commit.json`, which binds both files to source, configuration, seed, island,
  simulation generation, tick, aggregate-generation, predecessor, evidence
  status, counters, chain head, and SHA-256 identities.

Publication writes and fsyncs all three files in a same-filesystem staging
directory, atomically renames that complete directory into `generations/`, then
atomically replaces `CURRENT`. A crash before the directory rename leaves the
previous generation authoritative. A crash after the directory rename but
before `CURRENT` leaves one complete successor; the next locked load validates
its predecessor and contents before advancing `CURRENT`. Existing generation
directories are never replaced or pruned.

The evidence writer must be checkpointed before publication. The aggregate
publisher takes both the aggregate lock and the evidence writer's existing
process lock, verifies the live manifest and every declared shard, requires an
`open` continuation head, and snapshots that manifest. Historical generation
loads validate their pinned snapshot and the referenced immutable shard prefix
without requiring the mutable live manifest to remain at the old head. Later
shards are therefore not mistaken for part of an older generation, and the
protocol never deletes them to simulate a rollback.

The rotating writer's source contract must name the exact `source_git_sha`,
`config_contract_sha256`, `seed_contract_sha256`, `run_generation_id`, and
`island_id` used by the seven-envelope checkpoint. This prevents a valid
checkpoint from being combined with a valid but unrelated evidence stream.

Resume is deliberately noisy and explicit. Callers must provide
`OpenEcologyAggregateResumePins` containing the exact commit, checkpoint,
checkpoint-generation identity, evidence-manifest digest/status, source,
configuration, seed, run-generation, island, simulation-generation, aggregate
generation, and tick. Missing or mismatched pins fail closed. Readers also
reject duplicate keys, non-finite values, noncanonical JSON, symbolic links,
hard-linked control or generation files, path traversal, generation gaps,
surplus files, stale schemas, digest drift, and metadata over configured byte
or count ceilings.

This does not yet materialize a writable historical evidence prefix, implement
the seven runtime state adapters, enforce a campaign-wide disk quota, or prove
save/load continuation equivalence. The persistent runner must do those pieces
and demonstrate equality of subsequent world state, actions, RNG draws, hidden
state, genomes, and evidence digests before calling any generation
simulation-restartable.
