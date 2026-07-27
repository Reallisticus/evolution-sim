# Phase-A launch-readiness authority

The Phase-A authority code can parse exact evidence schemas, but parsing a
self-digested report is not evidence. A canonical SHA-256 protects integrity;
it is not a signature and does not prove that the reported measurement ran.
The authority therefore fails closed until each evidence kind has an
independent verifier that reconstructs its facts from referenced raw evidence.
At this implementation head no such authority verifier is registered. With no
evidence index, `open_ecology_phase_a_launch_readiness()` reports both
`launch_evidence_index_required` and
`independent_report_authority_verifiers_required`. The capture-reexecution
producer is implemented, but its wrapper is not yet independently reverified by
the index consumer, so it also cannot authorize training by itself.

Verifier registration is an executable callable registry, not an availability
Boolean. For every report, the authority calls the registered verifier with the
exact report path, campaign contract, and authorization time. The verifier must
reconstruct exactly the evidence kind, campaign digest, configuration digest,
source identity, and substantive facts from raw evidence. A missing or
non-callable entry, a verifier exception, surplus or missing reconstructed
fields, or any reconstructed-value mismatch fails closed. Merely changing a
flag therefore cannot create authority.

The common report envelope is
`mind_v3_open_ecology_phase_a_readiness_report_v1`. It binds one evidence kind
to the exact campaign digest, configuration digest, source commit, source
manifest, production time, substantive facts, and a canonical digest. A report
cannot satisfy a gate with `{"passed": true}` or another generic status flag.
Each evidence kind has an exact fact schema and semantic checks. For example,
the sealed preregistration explicitly requires the persistent-island
engineering proof before Phase A. That proof must show the complete 48-task
schedule and one H/Z/R proof triplet reaching tick 50,000 in the same worlds
with zero resets and an unchanged backbone. It must use only
`open_ecology_proof` seeds and a source-initialized frozen actor-FiLM artifact,
with zero access to Phase-A or Phase-B artifacts, scientific selection seeds,
or scientific outcomes. This keeps the expensive preregistered endurance proof
non-circular: it validates the runner mechanism, not post-training results. A
less expensive pre-Phase-A contract would require a new preregistration version
rather than a silent readiness reinterpretation. The
checkpoint report must cover all seven adapters and match subsequent world,
action, RNG, hidden, genome, and evidence state; the throughput report must
include authoritative Phase-A and 512/2,000-tick measurements and fit the
72/96-hour resource bounds. Phase-B mixed-density projection remains a
separate Phase-B gate and cannot block or be authorized by this record. The
storage report must carry a capacity measurement no more than 24 hours older
than both authorization and the current launch validation. An authorization
whose Drive attestation has aged beyond 24 hours cannot be replayed; a fresh
capacity report and new authorization are required. Local filesystem free space
is also rechecked by the cell runner.

Evidence is first sealed into
`mind_v3_open_ecology_phase_a_evidence_index_v1`. The index requires exact,
ordered identities. A partial index is valid for progress inspection and
reports only its missing dependencies and gates; authorization still requires
complete coverage:

| Dependency | Required semantic report kind(s) |
| --- | --- |
| 01 | `cross_surface_and_self_echo` |
| 02 | `runtime_genome_and_action_source` |
| 03 | `critic_gradient_and_density_schedule` |
| 04 | `fixed_batch_equivalence_and_speed` |
| 05 | `persistent_island_50000` |
| 06 | `checkpoint_continuation_equivalence` |
| 07 | `bounded_writer_event_coverage` |
| 08 | `causal_evaluator_rejection_battery`, `capture_noninterference_reexecution` |
| 09 | `preregistration_roundtrip_fail_closed` |
| 10 | `exact_sha_operational_path`, `multi_host_equivalence_and_torch_ci` |

The operational reports are `phase_a_full_throughput`,
`campaign_storage_capacity`, `output_lock_contention`,
`immutable_drive_uploader`, `verification_before_prune`, and
`terminal_aggregate_validator`. The authorization assembler checks the live
Git commit, clean status, source manifest, and sealed preregistration before
and after reading the bundle. It requires supplied evidence-index and
authorization objects to equal their exact regular, non-hardlinked files and
rejects symlink roots, path substitution, and byte drift. All reports and the
authorization must live under one real sealed directory. Once independent
raw-evidence verifiers exist, later validation will reopen and reconstruct
every referenced report rather than trusting the index or copied booleans.
The resulting authorization schema is v2 and its claim boundary is Phase-A
training only; Phase B, runtime integration, and promotion remain false.

The CLI is available through
`PYTHONPATH=python python -m evolution_sim.cli.open_ecology_phase_a`. Use
`readiness` to inspect an index, `build-evidence-index` to bind already
produced reports after their independent verifiers exist, `authorize` to
assemble the sealed authorization, and `validate` to reopen it.
`prove-capture-reexecution` is a real producer for
the dependency-08 capture report: it reopens a primary proof on the exact live
source, executes the repository's full proof again, requires equal canonical
report bytes, and emits the semantic report wrapper. `build-evidence-index`
accepts repeated
`--dependency-report DEPENDENCY/KIND=PATH` and
`--gate-report GATE=PATH` arguments. It does not create behavioral facts.
Those facts must come from exact clean-checkout proof and benchmark producers,
and the authority still requires independent reconstruction from their raw
evidence. Hand-authoring a plausible fact dictionary and recomputing its digest
is rejected.

At the implementation head where this authority was added, Phase A is still
blocked. The earlier twelve-case capture proof belongs to source
`f4cdd86d720f22baa085152bcbd979595cf51f10` and is stale for a newer source
commit. Focused unit and integration tests exercise many execution seams, but
they are not automatically promoted into readiness reports. No current-source
50,000-tick engineering runner proof, Phase-A plus long-horizon throughput
attestation, intended multi-host equivalence report, fresh storage-capacity
attestation, or complete exact-source operational evidence bundle has been
produced. More importantly, independent raw-evidence verifiers are not
implemented. Therefore no authoritative evidence index or launch
authorization should exist yet. The next work is to implement each
producer/verifier pair from a fresh clean exact-SHA checkout, run the missing
long/host/resource proofs, and then let this authority either issue the
Phase-A-only record or fail on the smallest concrete invalid or missing proof.
