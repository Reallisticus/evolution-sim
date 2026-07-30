# Persistent open-ecology island runner

`open_ecology_persistent_island.py` is the first real execution kernel for the
preregistered Phase-D observation. It constructs the fixed four-learner by
four-island by three-arm matrix in learner, island, H/Z/R order, with learner
0 and island 0 H/Z/R as the first triplet. Each task binds the frozen artifact,
learner seed, environment seed, independent genome-stream seed, policy-sampling
identity, world configuration digest, and task identity before the world is
created. Only tokenized, one-layer, width-256 `actor_film_v1` frozen artifacts
with the canonical seed-registry provenance are accepted.

The kernel does not call the episode-oriented `SimulationWorld.run()` loop. It
advances one `SimulationWorld` instance tick by tick, retains that same object
across calls, performs no optimizer update, verifies the model-state digest and
`requires_grad=False` after each chunk, and treats natural extinction as a
terminal result without reseeding. H is heritable controller genome plus
recurrent history and Z is the all-zero controller genome plus recurrent
history. R uses the same heritable-genome contract as H while resetting both
GRU state and previous public feedback before every decision. The runner
verifies after each finalized chunk that R retained neither history channel nor
a pending decision.

Observed simulator facts are projected into the typed rotating evidence stream:
founder and novel-child lineage transitions, births, deaths, attacks, mating,
physical signal contributors, same-target movement congestion, sampled
adjacency, and externally supplied frozen-branch intervention receipts.
Compact ecological summaries are produced every 100 observed ticks. The first
10,000-tick milestone records world continuity, frozen-model identity, event
accounting, extinction state, and the explicit absence of interim tuning or
acceptance. It is an observation record, not a promotion gate.

Every mutating call first acquires a persistent per-task exclusive lock, so two
processes cannot advance the same island from competing in-memory frontiers.
World ticks, evidence mutation, and task-local checkpoint publication then hold
the campaign epoch lock in shared mode: independent H/Z/R/island tasks can
execute and publish simultaneously, while a campaign storage scan cannot
observe a moving tree. At each quiescent 5,000-tick boundary, and again before
an advancing call returns if it stopped between scheduled boundaries, the task
seals the active evidence shard and, while retaining its per-task exclusive
lock, captures all seven real runtime components: world state, environment RNG,
frozen recurrent state, prior public feedback/history, sampling RNG, genome
population, and the exact evidence-writer continuation. Each task has its own
checkpoint destination, aggregate writer lock, immutable generation sequence,
and atomic `CURRENT`, so publication does not need a campaign-wide exclusive
critical section. The shared campaign lease excludes a storage scan until each
task's `CURRENT` matches its returned in-memory frontier, while the per-task
lock continues to exclude competing publications for the same island.
The same atomic JSON container also binds the immutable task, source, world and
writer configuration, all seed identities, artifact/model digests, and the
runner's interval counters, summaries, milestone records, event cursor, and
extinction state. Checkpoints are canonical non-executable JSON, fail above
256 MiB, and are read back immediately. Dynamic runner bookkeeping lives in a
reserved, schema-validated world-state field, so the configuration and seed
contract digests remain immutable for the whole run. The sealed checkpoint and
matching open evidence manifest are then copied into one immutable aggregate
generation before `CURRENT` is atomically advanced. Capturing the call-return
boundary as well as the scheduled boundary prevents the durable evidence
prefix from advancing beyond the authoritative restartable generation. An
intervention receipt publishes a new same-tick aggregate generation for the
same reason. Loose checkpoints are deliberately not pruned: the intended
latest-two policy may be applied only after an older bundle has a separately
verified cold-storage archive receipt, and that receipt-gated deletion path
does not yet exist.

Each newly published checkpoint's metadata exposes canonical byte counts for
the seven component envelopes and monotonic nanosecond timings for capture,
container write, readback validation, aggregate publication, and the complete
checkpoint pipeline. These measurements are deliberately outside the
deterministic checkpoint payload; the exact-source throughput gate can
attribute serialization cost without making wall-clock noise part of replay
identity. The same profile records the concurrency contract: per-task
exclusive, campaign-epoch shared, and storage-barrier exclusive. Its frontier
latency model is the slowest task publication rather than the sum of all task
publications. This removes the known global serialization blocker but does not
prejudge filesystem contention or satisfy the still-required exact-source
throughput gate.

`PersistentIslandRunner.restore_from_current` requires externally retained
aggregate resume pins, validates `CURRENT`, its immutable generation,
checkpoint, evidence snapshot, source/config/seed identity, and continuation
chain, then constructs a new frozen world/policy pair and restores all seven
components without re-emitting founders or reseeding. A lower-level loose
checkpoint restore remains available for inspection, but only the pinned
aggregate path carries continuation authority. The next call resumes the
hash-chained evidence writer from the captured prefix and extends the aggregate
chain. Because the campaign epoch lock is an external sibling of the campaign
root, restoration also initializes or validates that lock under its exact
campaign/source identity; copying a campaign root cannot silently omit the lock
needed by the resumed worker. The seven-adapter and aggregate-publication
blockers are therefore removed; a repeatable interval storage check and the
exact-source Phase-D throughput gate remain launch blockers.

The remaining storage gate is intentionally coordinator-owned.
`campaign_barrier_frontier()` returns only while the task lock is held, the
writer is inactive, and a freshly reread canonical `CURRENT` pointer for an
authorized restartable aggregate exactly matches the live tick. Its canonical
receipt pins task/campaign/source identity,
config/seed/world digests, terminal state, aggregate generation and commit,
checkpoint identity, and evidence-manifest continuation. Once all worker
futures have joined, the coordinator must reread and validate the full expected
48-task matrix. It may then enter
`PersistentIslandRunner.campaign_storage_barrier(...)`, which validates the
receipt structures and digests, takes the campaign lock exclusively, and also
takes every listed task lock before yielding. A worker already mutating, a
checkpoint publisher, a duplicate task receipt, identity drift, or a new
advance during the scan therefore fails closed. The context deliberately does
not claim to validate cross-task seed roles or completeness; those are
coordinator gates.

A scan is authorized only after all workers returned from `advance_to`, their
live frontier receipts were reread, and the exclusive campaign barrier
successfully acquired all listed task locks. The barrier uses nonblocking lock
acquisition. Workers acquire task-exclusive then campaign-shared; the barrier
acquires campaign-exclusive then task-exclusive. This cannot deadlock: either
an active shared publisher rejects the barrier, or a barrier that wins the
campaign lock makes a worker's nonblocking shared acquisition fail and unwind
its task lock; if the barrier reaches that task first, its own nonblocking
attempt fails and releases the campaign lock rather than waiting in a cycle.
Per-runner scanning would hash the same growing tree dozens of times, while
reusing the first island's scan would incorrectly bless files written by later
islands. The runner advertises
both `OPEN_ECOLOGY_PARALLEL_CAMPAIGN_ADVANCE_AUTHORIZED=True` and
`OPEN_ECOLOGY_PARALLEL_CHECKPOINT_PUBLICATION_AUTHORIZED=True` because hostile
tests prove independent shared advances and two simultaneously held,
distinct-task publication hooks. It keeps
`campaign_storage_interval_check_not_integrated` truthful until the coordinator
persists the global scan receipt and schedules the next frontier only after that
receipt.

The focused test uses a real serialized width-256 frozen artifact and real
64-agent worlds. In addition to task order, treatment state, frozen weights,
lock exclusion, evidence accounting, and terminal extinction, it forces two
distinct tasks to block inside `_run_tick` simultaneously while an exclusive
scan is rejected, rejects a duplicate same-task advance, and holds two
checkpoint publications at deterministic barriers to prove both overlap and
scan exclusion without comparing wall-clock timings. The same hook proves a
second publication for one task remains excluded. A publication-failure case
also proves that the prior atomic `CURRENT` bytes and exact resume pins remain
loadable. The aggregate layer separately injects crashes during generation and
pointer publication. The runner test also forks each of H, Z, and R from a real
disk checkpoint into a freshly constructed runner.
Uninterrupted and restored branches must then match in requested/resolved
behavior, world and RNG state, recurrent/public-feedback state, genome
population, summaries, event counts, and final evidence manifests. The test
also proves pinned `CURRENT` restoration and that a third checkpoint candidate
is not deleted without a verified archive receipt. It does not claim that a
10,000-tick triplet has run.
