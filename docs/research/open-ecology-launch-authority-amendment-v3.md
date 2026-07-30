# Open-Ecology Launch-Authority Amendment v3

Date: 2026-07-28
Status: prospective, development-only, adopted before any Phase-A training or scientific outcome evidence
Base protocol: `docs/research/open-ecology-campaign-preregistration-v1.md`
Prior amendment: `docs/research/open-ecology-launch-authority-amendment-v2.md`
Prior amendment SHA-256: `80e5185ffedf2edd4f4850c7b475fabbaff9c192fcd4b70228477f0f5e0ebfd3`
Scope: launch authority, evidence timing, and the fixed-batch engineering runtime contract only

## Purpose and authority

Amendment v2, at the exact path and SHA-256 pinned above, remains the immutable
historical record that moved late-stage engineering evidence out of Phase-A
training authority and removed the unsupported prune claim. A machine
preregistration adopting v3 must bind those exact v2 bytes as inherited
authority as well as binding this v3 document; a v3-only seal is invalid. This
v3 amendment carries the v2 stage-specific and storage boundaries forward
unchanged. It adds one prospective engineering repair after an exact-source
D04 qualification showed that the original fixed-capacity implementation
preserved semantics but did not provide the required speed benefit.

This amendment changes no scientific matrix, seed registry, action, reward,
observation, network width, PPO parameter, genome contract, density schedule,
causal intervention, selection rule, acceptance threshold, campaign arm, or
run length. It does not authorize Phase-A launch. Every dependency,
operational-evidence requirement, verifier boundary, and stage transition from
v2 remains binding unless this document explicitly changes the fixed-batch
runtime shape below.

## Exact negative that motivated the repair

Source commit
`8d8bf674de18a4c885a3b25106c2d09af7e68560` is a frozen, valid negative D04
qualification and cannot launch, resume, or contribute evidence authority to a
later source. Its full-shape 16-world, 16-worker, 128-tick proof processed
42,300 active transitions. Scalar and fixed-capacity execution produced the
same semantic digest,
`67b83177590d5d87cc682b2405da774b02b01f37873c4a08e8886442ba761e8f`;
all 131,072 ordered comparisons passed, action mismatches were zero, and no
seed-boundary, CUDA, Xid, memory, swap, disk, or source-drift failure occurred.

The speed gate nevertheless failed. The scalar median was 17.226798899 seconds
and the fixed-capacity median was 18.6070263005 seconds, so fixed-capacity
execution was 8.0120944674 percent slower and lost both alternating-order
repetitions. The fail-closed raw manifest has SHA-256
`ec6479593bd84e50914c6aa32241d1b5dde8d1a441f82613e8285f46789c40ae`;
the observations have SHA-256
`e7abfd938d2ba15a6fcc2f485dca26e7408010161bf412b580c112ceeac6b01c`.
The preserved historical negative archive has SHA-256
`af25ab64ae33195c02279b7e73bba6a124cba474a68677cb6af9b7b585696a5c`.
These records establish a speed-only negative; they are not launch authority
and must not be transplanted into a fresh qualification.

## Bounded-bucket fixed-batch runtime v2

The repaired runtime keeps 320 as the hard per-world tick-start admission
ceiling. Its only permitted physical forward sizes are
`1, 2, 4, 8, 16, 32, 64, 128, 256, 320`. For each tick, the runtime selects the
smallest bucket greater than or equal to the number of admitted active rows and
pads only from that active count to the selected bucket. It must reject an
active count above 320; it must not split, truncate, reorder, or silently move
rows to another forward.

Row admission and order remain the exact tick-start, seed/tick/agent-hash
contract. Observation, feedback, hidden-state, genome, and action-mask values
for active rows are unchanged. Padding remains right-side zero observation,
zero feedback, zero hidden state, and stay-only action mask, but only through
the selected bucket rather than always through row 320. Padded rows remain
passive and cannot be sampled or commit recurrent state. Cross-world batching
remains forbidden.

The model forward may therefore receive a smaller physical batch dimension,
but the simulator semantics do not change. Policy sampling, recurrent-state
commitment, and action resolution remain sequential in the original
hash-permuted order. The repair may not batch sampling, resolve actions in
parallel, use post-resolution private state, introduce heuristic action
selection, or alter replay-visible results. Exact semantic comparison against
the scalar reference remains mandatory.

The recurrent model contract is
`mind_public_recurrent_masked_actor_critic_v5`. Its encoder, GRU gates, genome
FiLM projections, actor, and critic use the same batch-size-stable per-row
batched matrix-vector forward kernel. Native dense-matrix backward projections
preserve trainable gradients without changing that forward. Scalar and
bounded-batch execution must remain within the sealed D04 numerical tolerance;
exact bit equality across backend kernels is not assumed. Both modes share one
trainable architecture and the same free-running recurrent state transition; a
scalar-only reference network or a detached numerical shim is not permitted.

The synchronous runtime may pass its own tick-start snapshot and canonical
order through a one-use, identity-and-content-bound witness so the fixed-batch
collector does not repeat an already completed deep copy or order hash. The
witness must be retired even when staging fails or does not consume it, and
content drift must fail closed both before staging and before sampling. A
generic or external caller still receives the defensive copy and independent
order validation. The fixed path may likewise project the already validated
public observation directly through the same signed-int16
quantize/dequantize operation used by the serialized observation contract,
without performing the lossless struct, compression, and base64 framing round
trip. Neither shortcut may change a quantized value, schema check, action
mask, feedback row, or ordering decision. D04 must compare these active paths
against the scalar serialized reference across heterogeneous masks and
nonzero feedback and fail on any numerical, action, hidden-state, or semantic
drift. The production-collector comparison must pair transitions by exact
world, seed, tick, agent, and decision identity; require identical hidden-state
shape and bootstrap-value presence; and compare input hidden state, sampled
action log-probability, entropy, value, and bootstrap value under the
preregistered D04 numerical tolerance. Bootstrap coverage must be independently
counted in every scalar and bounded-batch timed path, must be strictly positive,
and must equal the paired numerical comparison count in every repeat. A
behavioral digest that omits those training numerics is not sufficient launch
evidence.

## Qualification and launch boundary

The 8d8 negative stays frozen. Any implementation of bounded-bucket runtime v2
must be committed under a new exact source SHA, pass the required CI lanes from
that SHA, and create fresh SHA-bound archive authority, preregistration,
runtime, storage, evidence, and guardian records. Its D04 proof must use the
unchanged full Phase-A shape and unchanged equivalence and speed thresholds,
exercise the declared bucket contract, preserve ordered semantic equivalence,
and demonstrate an actual speed benefit. Evidence from 8d8 may be cited only
as historical engineering motivation.

Passing a local test, microbenchmark, CI job, or training telemetry cannot
self-authorize launch. Phase-A training remains blocked until the fresh exact
SHA independently passes every v2 Phase-A dependency and operational gate,
including the new D04 proof, and the Mac two-party guardian validates the
complete source-bound authority chain. Selection and later phases remain
separately gated exactly as in v2.
