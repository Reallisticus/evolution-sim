# Open-ecology runtime checkpoint adapter

`open_ecology_runtime_checkpoint.py` is the real-runtime adapter for the
seven-component persistent-island container. It captures and restores a
`SimulationWorld` paired with a frozen
`DeterministicPublicRecurrentPolicy`; it does not claim that arbitrary Python
objects are restartable.

The adapter records the complete mutable world state, the environment's Python
RNG, frozen-policy recurrent state, finalized public feedback and optional
bounded public history, the Torch action-sampling RNG, the strict recurrent
genome-population snapshot, and a validated rotating evidence-writer
continuation. Every component carries the same exact git/config/seed,
generation/island, generation-index, and completed-tick binding. That binding
also pins the source-manifest SHA256, so a checkpoint cannot be resumed by a
different source tree that happens to report the same git commit. Runtime
dataclasses use a fixed whitelist and tagged canonical JSON. Tensors are raw
CPU float32 bytes with shape and SHA256 rather than pickle or another
executable format. Captured public history, genome provenance, continuation
state, and persistent-runner metadata are deep-owned JSON values: mutating the
live objects after capture, or a caller-owned payload after restore, cannot
silently rewrite the checkpoint or a later restore.

Capture is allowed only at a finalized tick boundary: there may be no pending
policy decisions or open trajectory sink. Persistent mode must also disable
retention of events, viewer frames, trajectory history, and policy diagnostic
history. The current tick's finalized trajectory rows remain in the world
snapshot, but the next tick replaces that scratch buffer. Explicit byte and
public-history ceilings make oversized state fail closed.

Restore requires a freshly constructed world and policy with the exact same
world config, frozen model digest, artifact, sampling mode/seed, genome
conditioning, history mode, and recurrent-reset mode. It rejects component
schema or binding drift, unknown/missing fields and dataclasses, non-finite
values, malformed tensor/RNG state, genome/live-agent disagreement, and stale
writer continuation. The fresh policy's genome stream is itself part of the
contract: mode, seed, world identity, pre-founder binding, mutation settings,
and empty/pre-founder state must agree before the serialized manager is
accepted. Every restored live agent is then checked against the manager and
its mind-inheritance metadata. Restore is transactional: all seven components,
their cross-component relationships, and the writer continuation are decoded
and validated into temporary values before any live world or policy field is
changed. A late validation failure therefore leaves the fresh runtime
untouched rather than producing a partially restored process.

Persistent-island checkpoints also carry one reserved, strict-JSON runner-state
attribute. Its exact schema records the observed/completed tick boundary,
interval counters, summaries and milestones, reset/replacement state, model and
task digests, and evidence progress. Its own canonical digest and the writer's
continuation digest, next-event index, and last-evidence tick are checked
together. Evidence may legitimately lag a completed world tick when no event
was emitted at that tick, but only this reserved runner contract authorizes
that lag; generic adapter captures still require an exact evidence tick.

Aggregate commit does not reach into checkpoint internals. The adapter exposes
a strict public extraction envelope containing the runtime binding and the
validated rotating-writer continuation. Aggregate validation accepts that
envelope only at the current adapter schema and rebinds its source manifest,
config, seed, generation, island, and tick to the checkpoint manifest before
committing evidence. The older direct rotating-writer continuation path remains
separate and unchanged.

The integration test runs real H (heritable), Z (zero-genome), and R (full
recurrent-input reset) worlds before and after a save/load boundary. It verifies
exact subsequent trajectory rows (including requested and resolved actions),
world summary and dynamic state, environment and policy RNG streams, recurrent
hidden/public-feedback/history state, genomes, and the hash-chained evidence
manifest.

`reset_recurrent_state_each_decision=True` is a full R-arm reset contract: both
the GRU hidden state and `PreviousPublicFeedbackInput` are zero before every
decision. Finalized feedback is not retained in that mode, and the checkpoint
adapter rejects an R payload that contains it.
