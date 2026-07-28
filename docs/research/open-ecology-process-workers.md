# Persistent open-ecology process workers

`open_ecology_process_workers.py` is the production process boundary for a
campaign whose simulator loop is CPU-bound. A worker count above one means
separate operating-system processes. It never means threads, an async wrapper,
or parent-side proxy calls into live runner objects. The complete 48-task
learner/island/H-Z-R matrix is assigned once in deterministic round-robin
order. Each child owns that immutable queue, opens a task only when the
coordinator selects it, and retains the resulting world, policy, evidence
writer continuation, and aggregate state for later 5,000-tick barriers. After a
coordinator restart, a fresh child can restore a started task only when the
parent supplies the exact aggregate `CURRENT` pins retained in the preceding
external frontier receipt. A pending task is represented exactly as tick zero,
nonterminal, with no aggregate pins.

The constructor starts idle children and performs a source, host, device, and
thread-allocation handshake. It does not open a runner or write campaign
evidence. This allows the coordinator to verify the preceding receipt and
rescan retained storage before any child mutation. Commands and results use
bounded canonical JSON envelopes over multiprocessing pipes. No world, policy,
Torch module, writer, lock, or callable is pickled across the command boundary.
The launch names one canonical absolute Git executable plus its exact SHA-256;
the launcher descriptor-validates it before spawning, passes that immutable
authority rather than an inherited command name, and each child revalidates it
before and after Git calls in a fixed environment. Both output pipes are
streamed under one byte ceiling and deadline, with the isolated command process
group terminated on a breach. The child checks
exact Git/source-manifest authority before and after each frontier, checks the
caller's expected tick and pins against its retained runner, advances its queue
sequentially, and returns a compact exact operational result: requested and
observed tick, terminal state, model-state digest, summary and milestone
digests, quiescent checkpoint, campaign-barrier frontier, and aggregate resume
pins. The parent joins every selected process, validates all results, and
exposes immutable runner views to the existing coordinator validation and
campaign-exclusive storage barrier. These views cannot advance a task.

Worker death, timeout, source drift, duplicate or reordered tasks, an omitted
task, a partial nonterminal frontier, and an invalid checkpoint/frontier pin
poison the whole launcher. No next interval is authorized. Remaining workers
are stopped so they cannot continue mutating after the parent has rejected the
batch. A normal close is a bounded graceful shutdown with an explicit
acknowledgement that no evidence was deleted. A forced stop can leave a partial
durable prefix, but it neither deletes nor rewrites that prefix and does not
invent a campaign frontier. Terminate is followed by a bounded join and then a
bounded kill/join. If any PID still reports alive after both deadlines, the
launcher creates a private, fsynced fatal marker in the campaign root and
raises; it never silently returns. Any later launcher and the host-health probe
refuse progress while that marker exists, even if the PID later disappears,
because disappearance alone does not prove what it mutated. If a worker dies
after task publication but before its bounded result reaches the parent, the
campaign's pre-mutation barrier intent and deterministic per-task attempt
authority identify the recovery scope; task `CURRENT` remains local completed
attempt evidence but never substitutes for the unresolved campaign receipt.
The parent can issue a distinct reconciliation command only for that immutable
intent. It sends the original predecessor state and original per-task attempt
digest; each child reconciles through the persistent runner's filesystem
authority, not through parent telemetry. An exact target `CURRENT` is restored,
an exact predecessor is deterministically completed while preserving any
recoverable pre-`CURRENT` attempt, and a terminal predecessor remains terminal.
The child returns a digest-bound disposition for every selected task. Any
lineage, target, prefix, attempt-authority, or task-set mismatch poisons the
launcher and publishes no campaign receipt.

The current persistent runner copies its frozen model to CPU, so the launcher
accepts only explicit CPU slots and rejects CUDA slots. This is deliberate:
assigning several nominal GPU workers to a CPU-copy runtime would make resource
reporting misleading and would not accelerate the Python world loop. Each slot
declares its local host identity and Torch thread count, and aggregate declared
threads cannot exceed host CPU count unless the launch specification explicitly
authorizes measured CPU oversubscription. The child also caps Torch intra-op
threads. A larger GPU fleet becomes useful only after the simulator/world
stepping or learner work actually has a GPU execution path; it must not be
claimed through this launcher.

The coordinator CLI builds slots from the sealed launch specification and
passes its own five-field static-assignment digest into the launcher. The
launcher recomputes the same records—task ID, worker index, worker queue index,
global queue index, and qualification priority—and refuses a mismatch. The
typical in-process construction used by the CLI is:

```python
host = local_process_worker_host_identity()
slots = build_process_worker_slots(
    tasks,
    worker_count=worker_count,
    host_identity=host,
    torch_threads_per_worker=torch_threads_per_worker,
)
launcher = PersistentProcessWorkerLauncher(
    tasks=tasks,
    campaign_root=campaign_root,
    campaign_id=campaign_id,
    repository_root=repository_root,
    source_git_sha=source_git_sha,
    source_manifest_sha256=source_manifest_sha256,
    git_executable=git_executable,
    git_executable_sha256=git_executable_sha256,
    slots=slots,
    assignment_sha256=coordinator_assignment_sha256,
)
```

At a qualification frontier the parent selects only the first H/Z/R triplet.
At catch-up it can select the remaining 45 pending tasks, and at common
frontiers it selects all 48; the immutable per-worker queues do not change.
`advance_frontier` accepts the selected IDs in full matrix order plus an exact
`ProcessWorkerExpectedState` for each and returns a
`ProcessWorkerFrontierBatch`. `batch.runner_views(tasks_by_id)` supplies the
read-only task surfaces consumed by the coordinator. The coordinator still
owns the campaign-wide exclusive scan, health evidence, immutable receipt, and
next-interval authorization; worker telemetry by itself is not scientific or
storage authority. `reconcile_frontier` is not a second advance API: it is
available only to finish the already persisted attempt authorities named by an
unresolved barrier intent, and its `ProcessWorkerReconciliationBatch` binds
those authorities and dispositions to the normal complete frontier batch.

Focused validation uses real spawned processes and hostile outcomes:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_open_ecology_process_workers
```

Those tests prove distinct child PIDs, child reuse across barriers, a fresh
launcher restoring from externally retained pins, graceful shutdown, no
mutation for duplicate/nonstatic scopes, fail-closed child death and timeout,
partial-frontier rejection, source mismatch rejection, explicit host/device
allocation, CPU thread accounting, and preservation of files created before a
failure. Mixed-state filesystem tests separately exercise the production
persistent runtime with an exact target `CURRENT`, an exact predecessor, and
recoverable pre-`CURRENT` evidence; forged and drifted variants fail closed.
The hostile shutdown test also proves that an unkillable PID writes the fatal
marker and blocks a new launcher. These are recovery/operational proofs, not
the final exact-source 48-task throughput benchmark or a completed long
scientific campaign.
