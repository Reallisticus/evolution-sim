# Persistent open-ecology campaign coordinator

`open_ecology_campaign_coordinator.py` is the campaign-wide authority above the
individual persistent island runners. It accepts only the frozen 4 learner × 4
island × H/Z/R matrix in exact learner, island, arm order. The first learner and
first island H/Z/R triplet alone advances through 5,000 and then 10,000 ticks
before the other 45 tasks start; it therefore produces the first real
observation without waiting for 480,000 unrelated world ticks. It remains
primary scientific data and is not a disposable pilot or an interim tuning
gate. The following two catch-up barriers advance the pending matrix to 5,000
and 10,000 while the first triplet remains unchanged at 10,000. Only the next
15,000-tick barrier is again a common 48-task frontier.

Qualification and catch-up receipts are explicitly partial-scope receipts.
They list the complete frozen matrix, but unstarted tasks have state
`pending_unstarted` and no fabricated aggregate pin. `all_tasks_at_common_frontier`
is false until the second catch-up receipt puts all 48 tasks at 10,000. That
receipt and subsequent full-matrix receipts carry real pins for all 48 tasks.
Static worker queues are round-robin assignments over the unchanged matrix
order. More than one worker is refused unless the runner exports its tested
shared-advance/exclusive-barrier locking capability and a separately supplied
persistent process/host worker launcher owns the queues. The coordinator does
not treat an in-process `ThreadPoolExecutor` as production parallelism because
Python world-loop GIL cost and per-runner model memory have not been benchmarked.
The bundled process launcher uses real spawned operating-system processes,
keeps their PIDs and task queues stable across barriers, and returns immutable
digest-bound runner views; live worlds and Torch modules never cross IPC.

The only mutating unit is one 5,000-tick scope barrier. Before any selected
worker can mutate a task, the coordinator creates an immutable barrier intent
outside the active root. The intent binds the exact sequence, scope, target,
selected task order, every predecessor state and aggregate pin, and a
deterministic per-task attempt authority derived from task identity,
`run_generation_id`, predecessor commit (or explicit genesis), and requested
target tick. A completed campaign receipt binds the intent's whole-file
SHA-256. If the interval fails, an immutable failure observation preserves any
parent-visible task results and runner pins, but explicitly says that it may be
incomplete and cannot authorize a new attempt. An unresolved intent therefore
cannot silently become a fabricated frontier. On restart, the coordinator
reopens that exact intent instead of creating another one, validates its
predecessor and per-task attempt authorities, preserves or creates the
explicitly incomplete failure journal, and asks the persistent workers to
reconcile durable task state. A worker may accept only the exact predecessor,
an exact target successor of that predecessor, or preserved pre-`CURRENT`
evidence belonging to the same original attempt. Mixed target/predecessor
publication is allowed; unknown, forged, intermediate, or multiply plausible
lineage fails closed. The final receipt binds the original intent file, failure
journal file, per-task reconciliation dispositions, and joined process batch.

Selected workers advance
to the requested frontier and return with their evidence writer inactive and
aggregate `CURRENT` restartable. A naturally extinct world keeps its earlier
terminal tick and pinned terminal aggregate; every selected nonterminal world
must be at the exact requested tick. After all selected workers join, the
coordinator rereads every started task frontier, validates task, campaign,
source, configuration, seed, checkpoint, evidence, and aggregate commit pins,
then enters the runner's campaign-wide exclusive storage barrier. It scans the
active campaign root under that lock. Carried unmaterialized qualification rows
are also reloaded from their live aggregate `CURRENT` and external evidence
prefix under the same final exclusive barrier; copying their old receipt row is
not sufficient. Only after a source recheck, quota/free-space check, and real
resource-health hook succeed does it atomically create the immutable scope
receipt and release the next step.

Every authoritative barrier performs a complete descriptor-safe SHA-256 audit
of the active tree. There is no exported incremental hash-reuse path: inode,
size, and restored modification time are not accepted as evidence that bytes
are unchanged. The terminal 50,000-tick barrier identifies its terminal audit
mode, while every earlier receipt identifies the same complete-byte validation
contract.

This conservative contract is cumulative: scan work is proportional to the
current tree bytes at every barrier. The first exact-SHA qualification run must
record scan wall time, bytes, and entry count separately from simulation time.
An authenticated-delta optimization is not authorized yet because an ordinary
writable filesystem cannot prove that an old prefix remained byte-identical
from inode, size, timestamps, or a prior receipt alone. Any future optimization
must narrow the interim claim or add genuinely immutable/content-addressed
prefix storage, retain the prior receipt hash chain, hash all new and mutable
leaves, and periodically plus terminally repeat the complete audit.

Frontier receipts live in a canonical directory outside and not above the
active campaign root, so writing a receipt cannot invalidate the scan it binds.
Each receipt contains the exact source commit and source-manifest digest, task
matrix and static assignment digests, every started task's externally usable
aggregate resume-pin envelope, explicit pending rows for the rest,
configuration and seed contract digests, task/world digests, requested and
observed ticks, storage bytes/free space and an entry-manifest digest, pre/post
resource snapshots, the preceding receipt hash, qualification and matrix
frontiers, and an explicit statement that no pruning or deletion occurred.
Process-mode receipts additionally bind the joined batch digest, every selected
task-result digest, worker PIDs, host/device/thread declarations, and each
worker's before/after source snapshot. These operational fields do not replace
the simulator checkpoint or scientific evidence. Recovery receipts additionally
bind the preserved failure observation and original attempt. Health snapshots
bind a canonical executable identity, descriptor metadata, executable SHA-256,
and the externally retained health-baseline file SHA-256. A native probe is
invoked directly. An executable script is copied byte-for-byte into a private
snapshot and run only through its separately pinned absolute native shebang
interpreter. Every shebang script must carry the interpreter's exact SHA-256 in
the generated second-line marker; a script without that external binding is
rejected. Arbitrary interpreter arguments, `python -m`, and replaceable script
arguments are rejected. The externally supplied launch-spec SHA-256 is also
retained as launch authority.
Existing receipts are never overwritten or silently adopted. If a process
stops after publishing one, an operator must retain its whole-file SHA-256 and
resume from that exact receipt rather than rerunning the same step.

Resume is fail closed. A nonzero started-task frontier cannot be constructed
without the externally retained receipt and its independently retained
whole-file SHA-256; a digest stored inside the same JSON file is not resume
authority. `restore_campaign_runners_from_receipt` passes every existing pin
into `PersistentIslandRunner.restore_from_current`; pending rows remain unopened
until catch-up selects them. The current task tick and live `CURRENT` must equal
those retained pins. Missing selected tasks, source drift, a stale live
frontier, an unrestartable aggregate, a missing or stale storage receipt,
output-lock contention, and storage-barrier contention all stop before the next
step is authorized. The read-only `status()` path never advances a world or
writes a file.

On a process-level restart after triplet qualification, those three runners
need not be materialized merely to advance the other 45 catch-up tasks. Their
externally hashed 10,000-tick rows are carried forward explicitly as
`retained_unmaterialized_worker`, with unchanged real pins, while an exclusive
pre-advance storage rescan proves that the active root has not drifted. They are
not sent a 5,000- or 10,000-tick catch-up command. At the common 15,000
frontier, fresh workers restore them from those exact pins and all 48 tasks
advance one interval.

The coordinator requires a resource hook before any advance; it has no
placeholder that can count as production authority. The CLI launch spec names
a non-shell absolute command that receives the phase and target in
`EVOSIM_OPEN_ECOLOGY_HEALTH_PHASE` and
`EVOSIM_OPEN_ECOLOGY_FRONTIER_TICK` and must return a strict JSON object. The
object must echo that exact phase and integer frontier and set `healthy` to
JSON `true`. The server hook is expected to report CUDA utilization, memory,
temperature, Xid, OOM, RAM, swap, disk, and active worker health. A nonzero
exit, hook exception, stale phase/frontier, unhealthy result, or non-canonical
value stops the barrier. Probe stdout and stderr share a streaming byte ceiling
and deadline; a breach terminates the isolated probe process group. The storage
scan retains the non-weakenable 200-GiB
active budget and free-space floor from the storage gate.

Source checks do not discover `git` through inherited `PATH`. The launch
authority names one canonical absolute Git executable and its whole-file
SHA-256. Both the coordinator and every process worker descriptor-revalidate
that executable before and after each bounded command and invoke it with a
fixed environment. Replacement, stderr, timeout, oversized output, a dirty
checkout, or a source/manifest mismatch fails closed.

This slice does not archive, upload, delete, or prune anything. A frontier
receipt is operational restart/storage authority, not a Google Drive archive
verification receipt. Older checkpoints remain local until a separate closed
bundle is uploaded, streamed back, independently checked, and represented by
the existing immutable Drive receipt contract. There is deliberately no
coordinator API that turns a frontier receipt into deletion eligibility.

The production CLI consumes one strict launch specification:

```json
{
  "schema_version": "mind_v3_open_ecology_campaign_launch_spec_v1",
  "campaign_id": "sealed-campaign-id",
  "campaign_root": "/absolute/compute-host/active-campaign",
  "receipt_directory": "/absolute/compute-host/frontier-receipts",
  "repository_root": "/absolute/exact-source-checkout",
  "source_git_sha": "40-lowercase-hex",
  "source_manifest_sha256": "64-lowercase-hex",
  "git_executable": "/absolute/canonical/git",
  "git_executable_sha256": "externally-retained-64-lowercase-hex",
  "selected_density": 64,
  "worker_count": 4,
  "process_workers": {
    "host_identity": "exact-hostname",
    "device_kind": "cpu",
    "device_index": null,
    "torch_threads_per_worker": 1,
    "allow_cpu_oversubscription": false,
    "response_timeout_seconds": 3600.0,
    "startup_timeout_seconds": 120.0,
    "shutdown_timeout_seconds": 30.0,
    "max_message_bytes": 8388608
  },
  "health_probe_command": [
    "/absolute/bin/open-ecology-health-probe"
  ],
  "health_baseline_sha256": "externally-retained-64-lowercase-hex",
  "health_probe_file_sha256": "externally-retained-64-lowercase-hex",
  "artifacts": [
    {
      "learner_index": 0,
      "learner_seed": 1570849880,
      "artifact_path": "/absolute/frozen-artifact.json",
      "artifact_sha256": "64-lowercase-hex",
      "artifact_file_sha256": "64-lowercase-hex",
      "source_commit": "40-lowercase-hex",
      "terminal_authority_sha256": "64-lowercase-hex"
    }
  ]
}
```

The real file contains four registry-ordered artifact bindings. A fresh process
starts source-verified idle workers; only the qualification triplet is opened
when the first exact scope frontier is actually advanced. A single-process
debug launch uses `worker_count: 1` and `process_workers: null`:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python \
  -m evolution_sim.cli.open_ecology_campaign run \
  --launch-spec /absolute/launch.json \
  --expected-launch-spec-sha256 <externally-retained-launch-spec-sha256> \
  --through-tick 10000
```

A restarted process must name the last external receipt explicitly:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python \
  -m evolution_sim.cli.open_ecology_campaign run \
  --launch-spec /absolute/launch.json \
  --expected-launch-spec-sha256 <externally-retained-launch-spec-sha256> \
  --resume-receipt \
    /absolute/frontier-receipts/step-0002-qualification_triplet-00010000.json \
  --resume-receipt-sha256 <externally-retained-whole-file-sha256> \
  --through-tick 15000
```

Receipt-only status is a read-only operation:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python \
  -m evolution_sim.cli.open_ecology_campaign status \
  --receipt \
    /absolute/frontier-receipts/step-0002-qualification_triplet-00010000.json
```

After an external process has copied a completed evidence shard into a separate
canonical directory whose basename is its bundle ID, the campaign CLI is the
production closure path:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python \
  -m evolution_sim.cli.open_ecology_campaign seal-bundle \
  --launch-spec /absolute/launch.json \
  --expected-launch-spec-sha256 <externally-retained-launch-spec-sha256> \
  --bundle-dir /absolute/closed-bundles/bundle-0001 \
  --bundle-id bundle-0001
```

This command derives campaign and exact-source authority only from the pinned
launch specification, verifies the clean live source before and after calling
`seal_closed_bundle`, and applies the fixed campaign storage limits. It rejects
overlap with the active root, non-canonical paths, an existing marker, or source
drift. It only removes write permission and creates the immutable closure
marker; it never archives, uploads, deletes, or prunes.

Focused validation:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_open_ecology_campaign_cli \
  python.tests.test_open_ecology_campaign_coordinator \
  python.tests.test_open_ecology_process_workers
```

The receipt chain solves restart authority at completed scope barriers and the
narrow fail-closed case of an unresolved intent whose tasks are provably at its
exact predecessor or exact successor. It does not claim recovery from ambiguous
or corrupted state, scientific acceptance, the exact-source throughput gate, a
measured efficient multi-worker topology, or a completed 10,000/50,000-tick
run. The next server integration must use the bundled process launcher with the
real host-health probe and benchmark worker/thread topologies at the exact clean
source SHA before choosing the long-run allocation. The local process and
filesystem suites prove orchestration and hostile failure handling, not
production throughput or a scientific result.
