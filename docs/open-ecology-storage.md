# Bounded open-ecology trajectory storage

Long persistent worlds must not write one full trajectory record for every
agent decision. That representation is useful for short exact experiments but
grows with both population and time, and it is the wrong default evidence
surface for an open-ended ecology run.

`BoundedOpenEcologyTrajectoryWriter` is a `TrajectorySink` implementation for
that workload. It consumes the existing public trajectory contract one record
at a time, retains no full records in memory, writes compact interval summaries,
and writes full records only for replay windows explicitly named before the run.
Replay retention has independent hard row and canonical-byte ceilings. Once
either ceiling is reached, later selected rows are omitted and the footer
reports the exact omitted count; the limit is never exceeded silently.

```python
from evolution_sim.io.open_ecology_writer import (
    BoundedOpenEcologyTrajectoryWriter,
    ReplayWindow,
)

sink = BoundedOpenEcologyTrajectoryWriter(
    "output/open-ecology/run-17.jsonl.gz",
    summary_interval_ticks=100,
    replay_windows=(
        ReplayWindow("opening", 0, 31),
        ReplayWindow("late-observation", 9_900, 9_999),
    ),
    max_replay_rows=4_096,
    max_replay_bytes=64 * 1024 * 1024,
)
world.run(mode=RunMode.SUMMARY_ONLY, trajectory_sink=sink)
```

The writer derives only facts evidenced by public trajectory fields. Each
period and the final lifetime summary include requested and resolved action
counts, validity and reward totals, observed post-decision vitality ratios,
resource/feeding/drinking records, reproduction and death records, attack
attempt/success/kill/damage records, mate requests/resolutions, emitted
communication tokens, and decision-record distributions by runtime species and
ecotype. These are record counts, not inferred births, population censuses,
alliances, meanings, or causal claims. For example,
`reproduction_event_record_count` counts parent decision records marked
`reproduced`; it must not be relabeled as a birth count without a separate
birth-event contract.

The format is canonical JSON Lines, optionally in deterministic gzip when the
path ends in `.gz`. It begins in a same-directory temporary file and becomes
visible at the destination only after the footer is written, the file is
flushed, and an atomic replace succeeds. Abort and input-contract failures
remove the temporary file while leaving any previous complete destination
untouched. Records must be canonical-JSON compatible, tick ordered, within the
configured input-size ceiling, action-mask consistent, and compatible with the
source action contract. Non-finite numbers and malformed nested outcome fields
fail closed.

The footer binds the canonical configuration, source trajectory contract,
writer configuration, supplied run-summary digest, the digest of every
canonical input record, and the digest of all emitted bytes before the footer.
The pre-footer digest deliberately excludes the footer so it has a
non-self-referential verification scope. The footer and the live `diagnostics`
property also expose replay rows/bytes written, rows omitted by bounds, peak
bounded-counter occupancy, logical bytes written, and an explicit
`in_memory_full_record_count` of zero.

This sink is an observational evidence stream, not an exact world checkpoint.
Periodic summaries cannot reconstruct hidden recurrent state, RNG state,
population state, or optimizer state, and a replay window is exact only for the
public records it contains. Persistent training therefore still needs separate
atomic model/world checkpoints and their own restore validation. The output
also grows slowly with the number of summary periods; very long deployments
should rotate completed runs or shards and archive verified artifacts to remote
storage rather than treating one file as an unlimited log.

## Rotating persistent-island evidence

`RotatingOpenEcologyEvidenceWriter` is the opt-in storage primitive for the
explicit event side of a persistent island. It does not accept trajectory
dictionaries and does not infer social or evolutionary facts. A caller must
construct one of the typed birth, death, lineage, dyadic-interaction,
signal-contributor, congestion, or intervention evidence values from an
already-observed runtime event. The serialized vocabulary therefore records
only what its producer states; it does not turn reproduction records into
unstated birth counts, signal proximity into communication meaning, or repeated
dyads into societies.

```python
from evolution_sim.io.open_ecology_rotating_writer import (
    BirthEvidence,
    RotatingEvidenceConfig,
    RotatingOpenEcologyEvidenceWriter,
)

writer = RotatingOpenEcologyEvidenceWriter(
    "output/open-ecology/island-0-evidence",
    run_id="phase-a-island-0",
    source_contract={
        "source_git_sha": source_git_sha,
        "campaign_contract_sha256": campaign_contract_sha256,
        "event_producer": "explicit-public-runtime-events-v1",
    },
    config=RotatingEvidenceConfig(
        max_ticks_per_shard=512,
        max_rows_per_shard=100_000,
    ),
)
writer.append(
    BirthEvidence(
        tick=17,
        event_index=0,
        event_id="birth:0",
        child_agent_id=91,
        parent_agent_ids=(12, 44),
        lineage_id="lineage-91",
        genome_sha256=genome_sha256,
    )
)
writer.finish()
```

Every shard is canonical JSONL inside deterministic gzip. Tick span, event-row
count, canonical event-row size, total uncompressed bytes, and compressed bytes
have independent hard ceilings. One writer stream also has a hard compressed
byte ceiling (`8 GiB` by default), plus shard-count and manifest-byte ceilings.
That is not a campaign-wide disk quota: 24 streams at their defaults could
still retain roughly 192 GiB before accounting for checkpoints, reports, and
temporary files. A real campaign must therefore add a root-level quota and a
verified remote archive/retention policy before it is allowed to prune local
artifacts.

A completed shard is flushed and fsynced in a same-directory temporary file.
The writer then records a digest-bound transaction journal, atomically publishes
the shard, atomically advances the manifest, and clears the journal. Handled
publication failures roll back to the last durable manifest, while a resume
with an externally retained continuation state resolves an interrupted
transaction to the side authorized by that state. A persistent, exact-content
control file carries an exclusive process lock, preventing two resumers from
writing the same stream. The canonical manifest binds the source contract and
writer configuration, records both byte sizes and the event-stream digest, and
chains each shard to the compressed SHA-256 of its predecessor. Manifest and
shard readers reject extra keys, unknown event types, duplicate or noncanonical
JSON, non-finite values, discontinuous event indexes, backward ticks,
undeclared directory entries (including stale temporary shards),
digest/chain mismatches, malformed locks, symlinks, and files over their
declared ceilings.

The gzip byte stream is deterministic for the same Python and zlib runtime,
compression level, input, and shard boundaries. Cross-runtime byte identity is
not asserted; campaigns that compare compressed artifact digests must pin that
runtime provenance. The SHA-256 chain detects mutation relative to a trusted
head but is not itself an authenticity or completeness authority. A final
evaluator must retain the expected terminal manifest digest and status outside
the stream and pass them to `load_open_ecology_evidence_manifest`; otherwise an
attacker able to rewrite the directory could replace it with a self-consistent
shorter prefix.

`checkpoint()` seals the current shard and returns a small canonical-JSON
continuation payload suitable for the checkpoint component named
`evidence_writer_continuation_state`. The old writer becomes read-only and a
new instance must resume with exactly the same run id, source contract, writer
configuration, manifest digest, chain head, event index, and tick. A resume at
an already-natural shard boundary is byte-for-byte equivalent to uninterrupted
ingestion. A forced checkpoint deliberately creates a shard boundary, so it
preserves the event semantics and chain but is not byte-identical to a run that
did not make that checkpoint.

The writer retains no event rows in memory; active memory is fixed-size gzip and
hash state plus a manifest entry per completed shard, capped by `max_shards` and
`max_manifest_bytes`. It is still only a storage primitive. In particular,
natural rotation advances the mutable evidence manifest independently of an
enclosing world/model/RNG checkpoint. The transaction journal repairs a crash
during one shard publication, but it does not provide an aggregate two-phase
commit after that journal has been cleared. A persistent runner must add
versioned manifest generations or another atomic aggregate checkpoint protocol
before it can safely rewind evidence to an older world checkpoint. No runner
currently emits these typed events, enforces a campaign-wide disk budget,
archives and verifies shards remotely, or proves aggregate continuation, so
this component must not be treated as end-to-end 50,000-tick readiness.
