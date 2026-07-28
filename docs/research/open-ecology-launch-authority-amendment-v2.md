# Open-Ecology Launch-Authority Amendment v2

Date: 2026-07-27
Status: development-only, adopted before any Phase-A scientific outcome evidence
Base protocol: `docs/research/open-ecology-campaign-preregistration-v1.md`
Scope: launch authority and evidence timing only

## Why this amendment exists

The base preregistration correctly freezes the scientific experiment, but its
final readiness paragraph accidentally made later-stage engineering evidence a
precondition for the first training update. In particular, it required a
50,000-tick persistent H/Z/R engineering triplet, selection-path and terminal
aggregate evidence, and a verification-before-prune proof before Phase A. It
also required at least two host classes even though the same document permits
one homogeneous host class. The repository's Drive archiver deliberately has
no deletion or pruning operation, so a successful post-verification prune
cannot truthfully be demonstrated.

This is a prospective correction. No Phase-A training, selection, Phase-B,
Phase-C, or Phase-D outcome was inspected to choose it. It changes no
scientific matrix, seed, action, reward, network width, PPO parameter, genome
contract, density rule, threshold, causal intervention, arm, or run length.
The base protocol remains the authority for those facts.

## Stage-specific authority

Evidence is required before the first stage that can usefully exercise the
corresponding mechanism. Passing one stage never self-signs a later stage.
Every report named below still needs an exact-source producer, a closed raw
evidence bundle, and an independent verifier that reconstructs the facts. A
digest, Boolean, unit test, or hand-written report is not authority.

### Phase-A training authority

The first training update requires only the evidence needed to make that update
scientifically interpretable and operationally safe:

1. `cross_surface_and_self_echo` proves the shared observation/action/artifact
   surface and excludes receiver self-echo.
2. `runtime_genome_and_action_source` proves heritable and zero-genome runtime
   behavior, birth resets, missing-genome rejection, and no heuristic action
   source.
3. `critic_gradient_and_density_schedule` proves the four ablation cells,
   gradient boundary, and fixed schedules.
4. `fixed_batch_equivalence_and_speed` proves exact Phase-A-shape repeatability,
   ordered semantic equivalence, and an actual speed benefit.
5. `preregistration_roundtrip_fail_closed` independently rebuilds the machine
   campaign and rejects unknown, dirty, stale, and duplicate-key inputs.
6. `exact_sha_phase_a_training_and_torch_ci` proves a fresh clean checkout,
   exact source manifest and import root, the full-shaped training benchmark on
   every host class intended for Phase-A training, and a non-optional passing
   Torch CI lane.

The operational reports are `phase_a_training_throughput`,
`campaign_storage_capacity`, and `output_lock_contention`. The throughput report
uses only the measured Phase-A training shape and its frozen resource envelope;
it does not claim that 512/2,000-tick selection or the 2.4-million-tick Phase-D
matrix was measured. The capacity report must freshly prove both target-host
filesystem headroom and Drive quota. This authority permits Phase-A training
only.

### Phase-A selection authority

Training completion does not authorize selection. Before any scientific
selection seed is opened, the authority must additionally verify
`causal_evaluator_rejection_battery`,
`capture_noninterference_reexecution`, the complete immutable Phase-A terminal
chains and artifacts, and a separate `phase_a_selection_throughput` report over
the preregistered 512-tick primary-plus-independent-replay shape. That later
authority permits the frozen selection pass only. It cannot authorize Phase B
or choose a threshold after seeing outcomes.

### Phase-B, Phase-C, and Phase-D authority

Phase B requires the exact selected-cell receipt plus its separately measured
mixed-density 256-tick training projection. Phase C requires the complete
Phase-B terminal selection receipt. Neither authority is implemented by the
Phase-A launcher.

Before any primary Phase-D island starts, the final exact source must
additionally prove the persistent runner, seven-component checkpoint
continuation, bounded writer/event coverage, production H/Z/R throughput and
resource envelope, immutable Drive uploader with readback and independent
`rclone check`, and terminal/milestone aggregate validation. These correspond
to the former dependencies 05 through 07 and the later operational gates. They
belong here because Phase D is the first stage that depends on them.

Host authority is based on intended host classes, not an artificial minimum of
two. A one-class plan must test that one class. A multi-class plan must test
every intended class and prove the bound runtime and semantic evidence
contracts equal across them. If they differ, the classes cannot be pooled as
exact replicas; use one homogeneous class or create separately versioned
campaigns.

## Storage and deletion boundary

Drive upload authority proves immutable names, deterministic bundle bytes,
stream readback SHA-256, an independent `rclone check` with zero differences,
and retention of the local source bytes. This version exposes no local pruning
operation and authorizes no deletion. Consequently
`verification_before_prune` is removed rather than simulated. A future
retention policy may add pruning only through a new version with a recoverable
deletion contract and hostile tests.

Until the later-stage producer/verifier pairs exist, those stages remain
blocked. This amendment removes circular work from Phase-A training authority;
it does not manufacture evidence and does not launch the campaign.
