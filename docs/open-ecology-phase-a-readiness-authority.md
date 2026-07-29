# Phase-A launch-readiness authority

The base scientific protocol remains
`docs/research/open-ecology-campaign-preregistration-v1.md`. Its inherited
launch-timing correction is
`docs/research/open-ecology-launch-authority-amendment-v2.md`, and the current
prospective fixed-batch engineering authority is
`docs/research/open-ecology-launch-authority-amendment-v3.md`. The machine
campaign seals all three documents by path and SHA-256. Amendment v3 preserves
the scientific experiment while replacing only the bounded-bucket runtime and
recurrent numerical-kernel contract: the cells, seeds, network dimensions,
optimizer, genome, density schedule, thresholds, horizons, and acceptance
rules are unchanged.

The resealed base protocol also closes the prelaunch scientific-contract gaps.
Every committed PPO update must contain accepted optimization work and an
actual model-state change; its KL audit count must equal accepted plus rejected
optimizer steps, and terminal authority carries the cumulative counts.
Phase-A cells share paired policy RNG and founder-genome world identities while
retaining distinct evidence task IDs. Fixed-horizon targets execute exactly
`T` scored ticks, then use an action-free value evaluation on a disposable
clone after the real tick-`T` ecology prefix. All living agents remain in
evidence, while final-tick zero-decision newborns are counted but excluded from
targets; passive terminal reward enters the target at the same `gamma` boundary
as PPO, and deaths bootstrap zero. The clone must leave canonical world, RNG,
counter, policy, model, and genome state unchanged. Collapse detection is
rolling across every 3,000-decision span and fewer decisions are insufficient
evidence. Selection includes an exact learner-seed initialized baseline on the
same four stochastic tapes; eligibility uses the median four-tape paired delta
inside each environment followed by an equal-weight median across all 32
environments. A flattened 128-tape median cannot authorize a learner.

Phase-A training now requires dependencies 01 through 04, 09, and the
Phase-A portion of 10. In executable terms those are
`cross_surface_and_self_echo`, `runtime_genome_and_action_source`,
`critic_gradient_and_density_schedule`, `fixed_batch_equivalence_and_speed`,
`preregistration_roundtrip_fail_closed`, and
`exact_sha_phase_a_training_and_torch_ci`. Its operational gates are only
`phase_a_training_throughput`, `campaign_storage_capacity`, and
`output_lock_contention`. A single intended training host class is valid when
that exact class is tested; if multiple classes are intended, every class must
be tested and their runtime and semantic digests must agree. The throughput
gate covers the measured Phase-A training shape only and needs at least three
fresh full-shape repetitions. Storage attestation must be no more than 24 hours
old and prove both host headroom and Drive quota. Passing this authority permits
training only.

Selection has a separate boundary. Before any scientific selection seed can be
opened, dependency 08 must pass through
`causal_evaluator_rejection_battery` and
`capture_noninterference_reexecution`, the immutable terminal chains and
artifacts must be complete, and the preregistered 512-tick trained-policy plus
initialized-baseline primary pass and equal full authority reexecution must
pass `phase_a_selection_throughput`. Each artifact therefore costs 288 worlds
per pass and 576 through authorization. Phase B and Phase C likewise
require their frozen receipts and stage-specific projections. Phase D is the
first stage that requires dependency 05's 50,000-tick persistent H/Z/R
engineering triplet, dependency 06's seven-component checkpoint continuation,
dependency 07's bounded writer and event coverage, the Phase-D host contract,
full production throughput, Drive archiving, and terminal aggregation. Those
later proofs cannot block the first Phase-A update and Phase-A telemetry cannot
self-authorize them.

Drive upload proves immutable names, deterministic archive bytes, stream
readback SHA-256, and an independent `rclone check` with zero differences. The
uploader retains the local source bundle. This version implements no pruning or
deletion operation, so `verification_before_prune` is not an evidence kind and
no successful deletion is fabricated.

The authority does not trust plausible reports. A canonical SHA-256 protects
integrity but is neither a signature nor proof that a measurement ran. Every
required evidence kind needs an exact-source producer and an independent
verifier that reconstructs its substantive facts from a closed raw-evidence
bundle. The callable producer and verifier registries are immutable; readiness
derives availability from those callables rather than from hand-set booleans.
A generic `{"passed": true}` report cannot authorize anything.

The common report envelope is
`mind_v3_open_ecology_phase_a_readiness_report_v2`. It binds the evidence kind,
campaign and configuration digests, source commit and manifest, production
time, facts, a bundle-local raw-manifest reference, and a canonical digest.
The sealed preregistration source record also binds the externally supplied
SHA-256 of the archive-tool authority. The storage producer must receive that
same digest, and the independent storage verifier compares the captured
authority bytes with the preregistration pin. Replacing both the authority and
its measurement self-consistently therefore still fails closed.
`mind_v3_open_ecology_raw_evidence_manifest_v1` closes the `raw/` inventory with
ordered roles, relative paths, byte lengths, and hashes. Verification opens
directories and files descriptor-relative with no symlink following, accepts
only bounded single-link regular files, reads each file exactly once, checks
the descriptor and pathname identity before and after the read, and parses and
hashes only those captured bytes. Missing or surplus files, surplus
directories, path escape, symlinks, hardlinks, special files, concurrent
replacement, byte drift, or a changed manifest reference fail closed. The D9
verifier reconstructs its facts from the same captured bundle snapshot used to
bind the report; it never reopens a second version of the evidence.

Evidence is sealed into
`mind_v3_open_ecology_phase_a_evidence_index_v2`. A partial index is useful for
inspection, but the Phase-A authorization assembler accepts exactly the six
Phase-A dependency proofs and three Phase-A operational proofs. It rechecks the
live Git commit, clean status, source manifest, sealed protocol documents, and
exact regular-file references. The resulting
`mind_v3_open_ecology_phase_a_launch_authorization_v3` explicitly leaves
selection, Phase B, Phase C, Phase D, runtime integration, and promotion false.

The CLI is
`PYTHONPATH=python python -m evolution_sim.cli.open_ecology_phase_a`.
`preregister` requires
`--expected-archive-tool-authority-sha256`; seal that exact-source,
endpoint-bound external authority before sealing the preregistration.
`readiness` inspects progress, `build-evidence-index` binds verified reports,
`authorize` writes a new launch record, and `validate` reopens it. The six
Phase-A dependency producers are
`prove-cross-surface-and-self-echo`,
`prove-runtime-genome-and-action-source`,
`prove-critic-gradient-and-density-schedule`,
`prove-fixed-batch-equivalence-and-speed`,
`prove-preregistration-roundtrip`, and
`prove-exact-source-training`. The three operational producers are
`prove-training-throughput`, `prove-storage-capacity`, and
`prove-output-lock`. Each command publishes a new directory containing
`report.json`, `raw-evidence-manifest.json`, and a closed `raw/` inventory;
none overwrites an existing authority bundle.

The exact-source producer requires a clean checkout matching the
preregistration, the two sealed full-shaped benchmark reports, authenticated
GitHub access to the exact commit's successful non-optional
`mind-recurrent-cpu` check, and a local CUDA host capable of completing the
entire Torch-gated suite plus CUDA training smoke. The throughput producer
does not reuse a timing claim: it reruns both three-repeat, full-shaped
heritable and `zero_all` benchmark arms on CUDA while sampling GPU memory and
temperature, host RAM and swap, kernel OOMs, and NVIDIA Xid errors immediately
before spawn, during execution, and immediately after exit. Command output is
consumed incrementally under a fixed byte ceiling and deadline; breach kills
the complete child process group. The benchmark excludes artifact
serialization, so the report makes no evidence-write-error claim.

Storage measures the actual privately configured trainer campaign directory with remote
`open(O_NOFOLLOW)`/`fstatvfs`/identity checks executed by the Python binary
pinned in the sealed archive authority. The SSH binary, effective
configuration, authenticated address/host key/authentication method, remote
Python, local rclone binary, and explicit rclone configuration are all checked
against that authority. Drive capacity is queried locally with the pinned
`rclone --config ... about gdrive: --json`. Authorization requires at least
300 GiB free on Drive, at most 200 GiB projected active campaign storage, and
free space on the remote training filesystem of at least 100 GiB or 20 percent
of its capacity, whichever is larger. The lock producer uses separate
operating-system processes to prove one-writer admission, concurrent
rejection, campaign-identity rejection, and post-release reacquisition.

Having executable producers does not itself authorize Phase A. The final
bundles must still be generated on the clean, pushed exact-SHA GPU checkout,
the exact-SHA GitHub check must have completed successfully, all raw bundles
must independently verify, and one complete evidence index and launch record
must be sealed before the first training update. Persistent-runner,
checkpoint, writer, immutable Drive uploader, and terminal aggregation
evidence remain later Phase-D requirements and cannot be substituted for this
smaller launch boundary.
