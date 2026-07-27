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
