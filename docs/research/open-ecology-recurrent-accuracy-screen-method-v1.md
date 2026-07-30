# Open-Ecology Recurrent Accuracy-Screen Method v1

Status: prospectively frozen, development-only contract. This document is not
launch authority, D04 evidence, training authority, or a scientific result. It
may govern one deciding CPU development screen only after the implementation,
validator, this document, and their tests are committed at one exact SHA; PR CI
and literal-push CI for that SHA must both pass first. Until then, none of the
four real seeds in this document may be evaluated. A mismatch between this
document and source fails closed.

## Why the screen method changes

The exact recurrent-adapter development screen at source
`95f9400d94ba97e45ea5e3e0ed07c94b0cad8769` is a frozen negative. Its report
has exact digest
`3e66632c428da5dac2a06d37c0c2de4e9be562ea9f5a245191296e4fe62bf395`,
raw-file SHA-256
`64b3e637f584c2202e4f2312cbbdde56df9547febdb804bb76d8393617db32e8`,
and archive SHA-256
`7e2199d21a34fd3e55e59ab9c4e698b1c7f052e0e66183bc4652ff4b18a214b1`.
That source and every older retired or invalidated source may never be rerun,
qualified, launched, resumed, or used as authority.

The frozen screen had zero action, semantic, identity, hidden-shape, and
bootstrap mismatches. Its fast adapter won all five timing pairs and beat the
faster legacy batched bracket by 5.7421%. It was rejected only by the obsolete
development requirement to remain within 0.75 of two shape-dependent BMM
rounding endpoints. Equivalent batched and sliced floating-point computations
need not be bitwise identical. Neither endpoint, nor their midpoint, is an
independent accuracy oracle. The midpoint, topology fishing, parameter fishing,
hidden rounding, and any inference-only numerical shim are prohibited.

The authoritative D04 relative tolerance `1e-5`, absolute tolerance `1e-6`,
semantic and replay requirements, speed gate, RSS gate, exact-source
requirements, and later launch gates remain unchanged. The only change is that
the non-authoritative CPU sieve measures the sole candidate against an
independent accuracy oracle instead of optimizing distance to legacy rounding.

## Frozen identities and execution order

The report schema is
`mind_v3_recurrent_adapter_accuracy_development_screen_v4`. The accuracy corpus
is `open_ecology_recurrent_accuracy_corpus_v1`. The oracle is
`python_float64_products_math_fsum_single_float32_round_v2`, and its metric
schema is
`authenticated_monotone_gamma_and_d04_float32_forward_error_metrics_v4`.

The five isolated child executions occur in exactly this order, each with a
distinct 64-character lowercase hexadecimal nonce:

1. `per_row_bmm_legacy_adapter_baseline_v2`;
2. `per_row_bmm_fast_adapter_control_v3`;
3. `fixed_row_tile_4_legacy_adapter_control_v3`;
4. `fixed_row_tile_4_fast_adapter_candidate_v3`;
5. a second `per_row_bmm_legacy_adapter_baseline_v2`, recorded only as
   `baseline_confirmation`.

Only `fixed_row_tile_4_fast_adapter_candidate_v3` is selectable. The per-row
fast lane is a timing and semantic control even though it was retained from
earlier work. The fixed-row legacy lane isolates the kernel from adapter
changes. The two legacy baseline children bracket the run and must have
identical runtime bindings and semantic fingerprints. There is no fallback
candidate: if the sole candidate fails, `selected_candidate` is `null`.

Every child runs on CPU float32 with Torch intra-op and inter-op thread counts
both exactly one. Python, Torch, NumPy, OS, architecture, environment, and
`PYTHONHASHSEED` bindings are recorded and identical across children; CUDA is
unavailable to the screen by design. Each child runs two warmup pairs in order
`[scalar, batched]`, `[batched, scalar]`, followed by five timed pairs in
order `[scalar, batched]`, `[batched, scalar]`, `[scalar, batched]`,
`[batched, scalar]`, `[scalar, batched]`. Every timed sample must have positive
and equal bootstrap comparisons across its two paths, the exact expected
collector count, and the same semantic digest. A median is the third value
after sorting five integer nanosecond samples; a paired batched win means the
batched integer is strictly smaller than the scalar integer in that repeat.

The candidate is the existing shared-trainable fixed-four-row
`torch.nn.functional.linear` forward plus the fast rollout adapter. A logical
row tail is right-zero-padded to four physical rows, projected once, and sliced
back to the logical row count. The same trainable forward must be used in
rollout, burn-in reconstruction, evaluation, and gradient-enabled PPO. The
state-dict layout and dense analytic backward stay unchanged. Runtime
autotuning, population-size branches, action-aware branches, policy heuristics,
hard-coded actions, hidden-state clamps or rounding, public-observation
changes, learner action changes, and inference-only shims are forbidden.

## Independent binary64 `math.fsum` oracle

The oracle is source-bound, CPU-only, non-selectable, detached, and forbidden
from runtime or training. It accepts finite, strided CPU `torch.float32`
inputs, a rank-two weight, and an optional rank-one bias. Each binary32
component is independently converted to Python binary64; each product is
formed in binary64; `math.fsum` accurately accumulates the products and exactly
one optional binary64 bias term; the accumulated value is then rounded once by
constructing a CPU `torch.float32` result. This is a binary64
`math.fsum`-based reference, not an exact-real dot product.

The oracle implementation must not call `torch.mm`, `torch.bmm`,
`torch.matmul`, `torch.einsum`, `torch.nn.functional.linear`, an ATen matrix
kernel, or the candidate. Source tests scan its reachable source and fail on
those calls. Each invocation is bounded to at most 1,000,000 tensor components
and 1,000,000 products. The complete forward corpus below is further bounded
to 1,874 selected dots and 479,648 products, below the preregistered ceilings
of 4,096 dots and 2,500,000 products.

Every tensor digest is the project `stable_payload_digest` of:

```text
{
  "dtype": "torch.float32",
  "shape": [...],
  "component_encoding": "ieee754_binary32_big_endian_hex_v1",
  "binary32_big_endian_hex": [component_0, component_1, ...]
}
```

Components are flattened in C order and encoded with `struct.pack(">f",
value).hex()`. Each case binds the input, weight, bias, candidate output, and
oracle output digests; shapes; selected indices; counts; family; seed identity;
and algorithm versions. The oracle exact digest additionally binds every
binary64 component with `float.hex()`. Caller-supplied oracle output or
provenance is rejected: metrics recompute and authenticate the oracle
internally.

For a selected dot of length `n`, define `u = 2^-24`, `k = 2n + 1`,
`gamma_k = ku / (1 - ku)`, and

```text
S = math.fsum(abs(float32(x_i) * float32(w_i)) for i)
    + abs(float32(bias))
allowance = gamma_k * S + 0.5 * ulp32(oracle_result)
```

The standard bound is applicable only when all exact product and bias terms
are finite, every nonzero term is normal in binary32, all nonzero terms have
one sign, and the finite binary64 pre-round sum is either zero or has magnitude
at least `2^-126`. Every applicable ordinary or holdout dot must have zero
standard-bound violations, and every ordinary and holdout dot must have zero
violations of the independent D04-style condition

```text
abs(candidate - oracle) <= 1e-6 + 1e-5 * abs(oracle).
```

The corresponding ratio is the absolute error divided by the right-hand side
and must be at most `1.0`. The two requirements are conjunctive. Cancellation
cases have a separate D04-oracle ratio gate of at most `1.0`, but are excluded
from the normal same-sign gamma aggregate. Underflow cases are diagnostics
only, have `standard_forward_error_bound_applicable=false`, and may neither
veto nor rescue selection. Maximum, p99.9, RMS absolute error, ULP distance,
standard-bound ratio, D04-oracle ratio, term class, and condition ratio are
serialized without substituting a smallest-subnormal fudge term.
Condition is serialized without a nonfinite JSON value: an exactly zero
binary64 `math.fsum` pre-round result is
`{"condition_ratio_kind":"exact_zero","condition_ratio":null}`; otherwise it
is `{"condition_ratio_kind":"finite","condition_ratio":S/abs(result)}` with a
finite JSON number. For the cancellation threshold, `exact_zero` has the
mathematical infinite-condition meaning and therefore satisfies the
`2^18` lower bound.

## Deterministic indexing and value generation

The four real uint64 seeds are frozen:

| Purpose | Decimal seed |
| --- | ---: |
| ordinary finite corpus | `8206998037284159040` |
| cancellation and underflow corpus | `6319666271073447219` |
| gradient and HVP corpus | `5016224177442350447` |
| untouched one-shot holdout | `3411624975195142037` |

Tests use synthetic seeds. Test source, fixtures, snapshots, and expected
digests may not contain or evaluate any of these four decimal literals, their
derived component digests, or the real holdout receipt. CI scans the entire
`python/tests` tree for those forbidden values.

For an index candidate `i`, compute:

```text
score(i) = SHA256(
  UTF8(
    "oe-accuracy-corpus-v1\0"
    + decimal_seed + "\0"
    + case_id + "\0"
    + axis + "\0"
    + decimal_i
  )
)
```

Candidates are ordered by `(score_bytes, i)`. Rows always pin `0` and `R-1`,
deduplicate them, hash-fill to `min(R, 4)`, and then sort numerically. For 768
outputs the exact set is `G = [0, 255, 256, 511, 512, 767]`. For 256 outputs,
`0` and `255` are pinned and two indices are hash-filled. For 20 outputs, `0`
and `19` are pinned and two are hash-filled. A one-output head selects `[0]`.
Only active rows are selectable; zero padding is recorded separately.

Value generation uses unsigned 64-bit arithmetic modulo `2^64`. Let
`C = 0x9E3779B97F4A7C15`. The domain word is the first eight bytes, interpreted
big-endian, of

```text
SHA256(UTF8(
  "oe-accuracy-value-v1\0" + family + "\0" + case_id + "\0" + tensor_name
))
```

For C-order flat index `i` and draw number `j`, define
`state = seed + domain_word + C * (1 + 4*i + j) mod 2^64`, followed by:

```text
z = state + C mod 2^64
z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9 mod 2^64
z = (z xor (z >> 27)) * 0x94D049BB133111EB mod 2^64
h_j = z xor (z >> 31)
```

This is the only SplitMix64 transform. `input`, `weight`, `bias`, loss
coefficient, and HVP direction are distinct tensor domains.

The ordinary families are exact dyadic binary32 values:

- `balanced`: for input and weight, `q = 1 + (h_0 & 1023)` and the value is
  `(-1)^(h_1 & 1) * q * 2^-12`; bias is
  `(-1)^(h_1 & 1) * 2^-14`.
- `alternating`: magnitudes match `balanced`. For input coordinate `[r,k]`
  the sign is `(-1)^k`; for weight coordinate `[o,k]` it is
  `(-1)^(o+k)`; and for bias coordinate `[o]` it is `(-1)^o`. Every product
  and bias term in an output dot therefore has sign `(-1)^o`, independent of
  row.
- `mixed`: `m = 1024 + (h_0 & 1023)`. Input and weight use
  `(-1)^(h_1 & 1) * m * 2^(e-10)` with
  `e = [-12, -8, -4][h_2 mod 3]`. Bias uses the same normalized mantissa and
  sign with `e = [-14, -10, -6][h_2 mod 3]`.

The holdout is one `composite` normal family. For every component,
`h_3 mod 3` selects `balanced`, `alternating`, or `mixed`, respectively, with
the formulas above and the holdout family domain. All ordinary and holdout
cases must be finite and contain no nonzero subnormal product or bias term.
All 425 alternating dots must be standard-bound applicable. Balanced, mixed,
and composite dots are classified individually; any same-sign applicable dot
must pass the standard bound, while a mixed-sign dot is excluded from that
aggregate but still must pass the D04-oracle gate. Each of the balanced,
mixed, and composite families must contain at least one mixed-sign dot so an
accidental same-sign-only generator drift cannot pass.

## Exact operational case table

The machine-readable canonical selection manifest is
`recurrent_accuracy_selection_manifest_v1`, exposed by
`frozen_real_selection_manifest()` in the source-bound accuracy module. The
ordinary, holdout, cancellation, underflow, and gradient tables in this
document are its human rendering. Its canonical `stable_payload_digest` is
`61e6cd3897a1dcf08cb777f09c3238efe0d167c5c313bd9077dc86c87ef42efa`.
Preregistration and post-run validation must recompute that digest from the
source constant and require this literal document binding; a generator that
merely agrees with itself is insufficient.

`R` is the active logical row count, `P` is the physical padded row count,
`I` is input width, and `O` is output width. `Kr` and `Ko` are selected row and
output counts. The affine table contributes 101 dots and 26,624 products; the
bucket table contributes 276 dots and 70,656 products; and the learner table
contributes 48 dots and 12,288 products.

For `P>R`, the first `R` rows are generated active values and rows `[R,P)` are
exact positive zero. The candidate executes the complete `P x I` input and
full `O x I` weight, slices back to the first `R` rows, and only then gathers
the frozen row/output selections. The oracle evaluates the corresponding
`Kr x I` input and `Ko x I` weight submatrices. Thus the proof exercises the
declared physical shape while the bounded oracle dot/product counts below are
exactly reconstructible.

| Case ID | Role | R | P | I | O | Kr | Ko | Dots | Products |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `affine.encoder_r5` | encoder | 5 | 5 | 604 | 256 | 4 | 4 | 16 | 9,664 |
| `affine.gru_input_r9` | GRU input | 9 | 9 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `affine.gru_hidden_r17` | GRU hidden | 17 | 17 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `affine.film_scale_r3` | FiLM scale | 3 | 3 | 16 | 256 | 3 | 4 | 12 | 192 |
| `affine.film_bias_r2` | FiLM bias | 2 | 2 | 16 | 256 | 2 | 4 | 8 | 128 |
| `affine.actor_r33` | actor | 33 | 33 | 256 | 20 | 4 | 4 | 16 | 4,096 |
| `affine.value_r1` | value | 1 | 1 | 256 | 1 | 1 | 1 | 1 | 256 |
| `bucket.gru_input.r1.p1` | GRU bucket | 1 | 1 | 256 | 768 | 1 | 6 | 6 | 1,536 |
| `bucket.gru_input.r2.p2` | GRU bucket | 2 | 2 | 256 | 768 | 2 | 6 | 12 | 3,072 |
| `bucket.gru_input.r3.p4` | GRU bucket | 3 | 4 | 256 | 768 | 3 | 6 | 18 | 4,608 |
| `bucket.gru_input.r5.p8` | GRU bucket | 5 | 8 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r9.p16` | GRU bucket | 9 | 16 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r17.p32` | GRU bucket | 17 | 32 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r33.p64` | GRU bucket | 33 | 64 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r64.p64` | GRU bucket | 64 | 64 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r65.p128` | GRU bucket | 65 | 128 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r129.p256` | GRU bucket | 129 | 256 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r257.p320` | GRU bucket | 257 | 320 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r319.p320` | GRU bucket | 319 | 320 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `bucket.gru_input.r320.p320` | GRU bucket | 320 | 320 | 256 | 768 | 4 | 6 | 24 | 6,144 |
| `learner.actor.r896` | PPO actor | 896 | 896 | 256 | 20 | 4 | 4 | 16 | 4,096 |
| `learner.actor.r1920` | PPO actor | 1,920 | 1,920 | 256 | 20 | 4 | 4 | 16 | 4,096 |
| `learner.actor.r2048` | PPO actor | 2,048 | 2,048 | 256 | 20 | 4 | 4 | 16 | 4,096 |

The exact ordinary (`O`) and holdout (`H`) selections are:

| Case ID | O rows | O outputs | H rows | H outputs |
| --- | --- | --- | --- | --- |
| `affine.encoder_r5` | `[0,1,2,4]` | `[0,58,93,255]` | `[0,1,2,4]` | `[0,29,143,255]` |
| `affine.gru_input_r9` | `[0,2,4,8]` | `G` | `[0,1,3,8]` | `G` |
| `affine.gru_hidden_r17` | `[0,12,15,16]` | `G` | `[0,1,13,16]` | `G` |
| `affine.film_scale_r3` | `[0,1,2]` | `[0,25,32,255]` | `[0,1,2]` | `[0,11,218,255]` |
| `affine.film_bias_r2` | `[0,1]` | `[0,211,248,255]` | `[0,1]` | `[0,22,118,255]` |
| `affine.actor_r33` | `[0,19,25,32]` | `[0,7,14,19]` | `[0,5,26,32]` | `[0,8,13,19]` |
| `affine.value_r1` | `[0]` | `[0]` | `[0]` | `[0]` |
| `bucket.gru_input.r1.p1` | `[0]` | `G` | `[0]` | `G` |
| `bucket.gru_input.r2.p2` | `[0,1]` | `G` | `[0,1]` | `G` |
| `bucket.gru_input.r3.p4` | `[0,1,2]` | `G` | `[0,1,2]` | `G` |
| `bucket.gru_input.r5.p8` | `[0,1,3,4]` | `G` | `[0,1,2,4]` | `G` |
| `bucket.gru_input.r9.p16` | `[0,2,3,8]` | `G` | `[0,1,5,8]` | `G` |
| `bucket.gru_input.r17.p32` | `[0,2,4,16]` | `G` | `[0,12,14,16]` | `G` |
| `bucket.gru_input.r33.p64` | `[0,13,19,32]` | `G` | `[0,25,28,32]` | `G` |
| `bucket.gru_input.r64.p64` | `[0,1,54,63]` | `G` | `[0,6,7,63]` | `G` |
| `bucket.gru_input.r65.p128` | `[0,23,28,64]` | `G` | `[0,2,60,64]` | `G` |
| `bucket.gru_input.r129.p256` | `[0,69,85,128]` | `G` | `[0,80,109,128]` | `G` |
| `bucket.gru_input.r257.p320` | `[0,51,182,256]` | `G` | `[0,42,92,256]` | `G` |
| `bucket.gru_input.r319.p320` | `[0,53,213,318]` | `G` | `[0,7,239,318]` | `G` |
| `bucket.gru_input.r320.p320` | `[0,33,278,319]` | `G` | `[0,207,307,319]` | `G` |
| `learner.actor.r896` | `[0,302,821,895]` | `[0,6,8,19]` | `[0,308,493,895]` | `[0,1,17,19]` |
| `learner.actor.r1920` | `[0,1114,1616,1919]` | `[0,9,17,19]` | `[0,1095,1136,1919]` | `[0,6,15,19]` |
| `learner.actor.r2048` | `[0,147,1445,2047]` | `[0,8,18,19]` | `[0,1298,1756,2047]` | `[0,6,18,19]` |

Each of the three ordinary families evaluates all 425 operational dots and
109,568 products, for 1,275 dots and 328,704 products. The one composite
holdout evaluates 425 dots and 109,568 products.

## Cancellation and underflow tables

Cancellation uses `R=P=4` for all seven affine roles. For input width `I=2m`,
draw exact dyadics `a,b = (1024 + (h_0 & 1023)) * 2^-11`. For each row and
half-index, set `x[r,k] = x[r,k+m] = a`; for each output set
`w[o,k] = b` and `w[o,k+m] = -b`. Bias is
`(-1)^(h_1 & 1) * (1 + (h_0 & 7)) * 2^-22`. Every selected dot must have
finite normal nonzero terms and
`S / abs(binary64_math_fsum_pre_round_result) >= 2^18`. The selected rows are
`[0,1,2,3]`. Exact output selections are:

| Case ID | I | O | Outputs | Dots | Products |
| --- | ---: | ---: | --- | ---: | ---: |
| `cancellation.encoder_r4` | 604 | 256 | `[0,54,102,255]` | 16 | 9,664 |
| `cancellation.gru_input_r4` | 256 | 768 | `G` | 24 | 6,144 |
| `cancellation.gru_hidden_r4` | 256 | 768 | `G` | 24 | 6,144 |
| `cancellation.film_scale_r4` | 16 | 256 | `[0,46,78,255]` | 16 | 256 |
| `cancellation.film_bias_r4` | 16 | 256 | `[0,127,142,255]` | 16 | 256 |
| `cancellation.actor_r4` | 256 | 20 | `[0,6,17,19]` | 16 | 4,096 |
| `cancellation.value_r4` | 256 | 1 | `[0]` | 4 | 1,024 |

This is exactly 116 dots and 27,584 products. These dots must have
D04-oracle ratio at most `1.0`; standard gamma statistics and ULP distance are
reported by condition-ratio band but do not enter the normal same-sign gate.

Underflow diagnostic `U1` uses `R=P=1`, every input and weight component
`2^-70`, and bias `(-1)^(h_1 & 1) * 2^-120`. `U2` uses `R=P=1`, every input
and weight component `2^-75`, and zero bias. Their exact output selections are:

The case IDs are `underflow_u1.<role>_r1` and
`underflow_u2.<role>_r1`, where `<role>` is one of `encoder`, `gru_input`,
`gru_hidden`, `film_scale`, `film_bias`, `actor`, or `value` in table order.

| Role | U1 outputs | U2 outputs | Dots per family | Products per family |
| --- | --- | --- | ---: | ---: |
| encoder | `[0,165,207,255]` | `[0,11,164,255]` | 4 | 2,416 |
| GRU input | `G` | `G` | 6 | 1,536 |
| GRU hidden | `G` | `G` | 6 | 1,536 |
| FiLM scale | `[0,133,167,255]` | `[0,46,251,255]` | 4 | 64 |
| FiLM bias | `[0,16,36,255]` | `[0,21,175,255]` | 4 | 64 |
| actor | `[0,12,13,19]` | `[0,17,18,19]` | 4 | 1,024 |
| value | `[0]` | `[0]` | 1 | 256 |

Each underflow family is exactly 29 dots and 6,896 products. Both must be
classified as underflow diagnostics with the standard bound inapplicable.
Their D04 ratio, absolute error, ULP distance, zero/nonzero result counts, and
subnormal-term counts are reported, but no underflow outcome can change the
selection decision.

## Fixed first- and higher-order gradient probe

Gradient cases use `R=P=5`, the seven affine roles, and the ordinary
`balanced` formula with the gradient seed. The frozen manifest retains these
bounded diagnostic selections as reconstructible case metadata only; they are
not the loss support or a claim of separately evaluated spot evidence:

| Case ID | Diagnostic rows | Diagnostic outputs | Full loss terms |
| --- | --- | --- | ---: |
| `gradient.encoder_r5` | `[0,2,3,4]` | `[0,116,138,255]` | 1,280 |
| `gradient.gru_input_r5` | `[0,2,3,4]` | `G` | 3,840 |
| `gradient.gru_hidden_r5` | `[0,1,2,4]` | `G` | 3,840 |
| `gradient.film_scale_r5` | `[0,2,3,4]` | `[0,155,240,255]` | 1,280 |
| `gradient.film_bias_r5` | `[0,1,3,4]` | `[0,153,159,255]` | 1,280 |
| `gradient.actor_r5` | `[0,1,3,4]` | `[0,7,17,19]` | 100 |
| `gradient.value_r5` | `[0,1,3,4]` | `[0]` | 5 |

The gradient loss uses every one of the five physical rows and every output in
each role: exactly 11,625 loss terms. Random streams reset within each case.
Gradient input, weight, and bias values use
`family="balanced"`, the literal case ID in this table, tensor names `input`,
`weight`, and `bias`, and C-order flat indices starting from zero separately
for each tensor.

Within a case, loss terms are ordered by physical row and then output.
Let `q = row_index * O + output_index`; `q` resets to zero for every case.
Compute `h_0` and `h_1` with `family="gradient"`, the literal case ID,
`tensor_name="loss_coefficient"`, `flat_index=q`, and draws `0` and `1`,
respectively. The coefficient is

```text
lambda_q = (-1)^(h_1 & 1) * (1 + (h_0 & 7)) * 2^-3
loss = 2^-8 * sum_q lambda_q * (z_q + z_q^2 / 8).
```

Candidate and dense `torch.nn.functional.linear` reference receive independent
clones of identical CPU float32 input, weight, and bias tensors. First
derivatives are taken with `create_graph=true` with respect to input, weight,
and bias. The fixed HVP direction for every component is
`(-1)^(h_1 & 1) * (1 + (h_0 & 15)) * 2^-8`, using the distinct
`hvp_direction_<name>` tensor domain, where `<name>` is exactly `input`,
`weight`, or `bias`. For each case and each of those three tensors separately,
use `family="gradient"`, the literal case ID, C-order `flat_index` starting
from zero, and draws `0` and `1` for `h_0` and `h_1`. The component ordinal
therefore resets per case and per tensor. The HVP is the derivative of the dot
product of the first gradient and that direction with respect to the same
ordered variables.

Each order compares exactly 572,033 components:

| Role | Input components | Weight components | Bias components | Total |
| --- | ---: | ---: | ---: | ---: |
| encoder | 3,020 | 154,624 | 256 | 157,900 |
| GRU input | 1,280 | 196,608 | 768 | 198,656 |
| GRU hidden | 1,280 | 196,608 | 768 | 198,656 |
| FiLM scale | 80 | 4,096 | 256 | 4,432 |
| FiLM bias | 80 | 4,096 | 256 | 4,432 |
| actor | 1,280 | 5,120 | 20 | 6,420 |
| value | 1,280 | 256 | 1 | 1,537 |

The combined first-gradient and HVP coverage is 1,144,066 components. All
values must be finite; shapes and variable order must match exactly; no
component may be omitted. First gradients use `rtol=2e-5`, `atol=2e-6`; HVPs
use `rtol=5e-5`, `atol=5e-6`. For each component the tolerance ratio is
`abs(candidate-reference)/(atol + rtol*abs(reference))`; every ratio must be at
most `1.0`. On the identical gradient tensors, candidate inference-mode and
gradient-enabled forward outputs must be bitwise equal.

The report also serializes nonzero reference-component counts for every
input, weight, and bias tensor separately for first gradients and HVPs. In
each order, every input row, every weight output row, and every bias component
must contain a nonzero reference component, and the aggregate nonzero
reference-component share must be at least `0.95`. These active-coverage gates
prevent structural zeros from being counted as evidence for an unexercised
output row. The deciding callable must be the production
`_backend_stable_linear` under the fixed-row candidate context, so this probe
exercises `_BackendStableLinearFunction` and its dense analytic backward;
calling the fixed-row forward helper directly is nondeciding.

## One-shot holdout and retirement

The ordinary, cancellation, underflow, gradient, source, semantic, timing
control, and pre-holdout candidate evidence is completed and exact-digested
before holdout access. Only then may the parent create a private,
process-bound, one-shot capability for the candidate child.

The fourth child therefore remains alive after emitting a single
exact-digested pre-holdout frame. While that child is paused on its private
standard-input channel, the parent runs and validates the fifth, closing
baseline child. Only after the opening controls, paused candidate evidence,
and closing baseline evidence all reconstruct does the parent send the
one-shot capability to the still-live candidate child. EOF, an unexpected
frame, output before admission, or a child exit at either stage fails closed
and retires the exact source without a retry.
Both child frames are single newline- or EOF-terminated UTF-8 JSON objects read
through a nonblocking, byte-capped, deadline-driven protocol. Partial frames,
oversized frames, extra bytes, trailing-output floods, and children that do not
exit before the shared deadline fail closed rather than bypassing the timeout.

Before starting the first child, the parent atomically publishes and fsyncs
the canonical sibling attempt marker
`recurrent-kernel-development-screen.source-retired.json` with a no-replace
hard link. It binds the source and source-byte bundle, method,
preregistration, report path, parent process identity, capability commitment,
and an exact digest of its own. Its existence permanently retires that exact
source even if the parent, a child, or the machine fails before a final report
or holdout receipt exists.

Immediately before holdout admission, the parent atomically publishes and
fsyncs the canonical sibling admission marker
`recurrent-kernel-development-screen.holdout-retired.json` with a no-replace
hard link. It binds the attempt-marker digest plus the pre-holdout frame,
closing-baseline evidence, child process identity and nonce, and the same
capability commitment. The report and both markers must be absent at initial
preflight. The attempt marker must exist before any child runs; both markers
must exist and reconstruct before holdout use. They remain permanently beside
a completed or failed screen. The capability is not sent until the admission
marker is durable.

The receipt binds:

- exact 40-character source SHA and clean detached checkout;
- exact report no-clobber path;
- schema and corpus versions;
- preregistration and pre-holdout evidence digests;
- all source-module byte digests;
- parent PID and process start identity;
- child PID, start identity, and private nonce;
- attempt- and admission-marker paths and exact digests;
- access ordinal exactly `1`;
- `phase="candidate_locked_before_selection"`;
- `test_mode=false` and absence of `PYTEST_CURRENT_TEST`.

The holdout generator is reachable only through the module-private authority;
no public composite generator accepts a caller-constructed permit. The
authority reopens every exact required source path, recomputes the byte bundle,
rechecks the live child PID/start identity and 64-hex nonce, reconstructs the
durable retirement marker, and confirms the final report path is still absent.
It refuses access before the pre-holdout digest is sealed, from a different
process, with a reused or missing nonce, with a changed report path, or at
ordinal other than one. It atomically consumes and invalidates its exact
module-private capability and generation token before generating tensors. The
report must show
`holdout_access_count=1`, `holdout_consumed=true`, and
`candidate_locked_before_holdout=true`. A crash, timeout, missing report,
nonempty child stderr, source drift, negative result, or successful result
retires that holdout for the exact SHA. There is no retry, partial reuse, or
second candidate. A fresh attempt requires a new source SHA and a prospectively
new holdout contract; it may not transplant any prior evidence.

Tests exercise the access state machine only with synthetic seeds and
synthetic receipts. Post-run validation may reconstruct serialized indices,
counts, component digests, metrics, and receipt bindings, but may not execute
the candidate against the real holdout a second time.

## Strict report schema and reconstruction

The atomic JSON report has exactly these root keys:

```text
schema_version
status
development_only
launch_authorized
authority_evidence_eligible
scientific_result
source
contract
runtime
children
baseline_confirmation
corpus
accuracy
gradients
holdout_receipt
selection
lifecycle
exact_digest
```

`status` must be `completed`; the four authority/science booleans must be
`true, false, false, false` in the order
`development_only, launch_authorized, authority_evidence_eligible,
scientific_result`. `lifecycle` must state no qualification, training, runtime
artifact, runtime action-selection change, promotion, or Phase-A output.
Unknown, missing, duplicate, nonfinite, boolean-as-integer, or wrong-type
fields fail validation. The report is written to a private pending path,
fsynced, reopened, independently validated, and atomically published to a
previously absent final path with a no-replace hard link; the pending link is
then removed. Stderr must be empty.

`source.module_byte_sha256` contains the SHA-256 of raw bytes, with no newline
normalization, for exactly:

```text
python/evolution_sim/env/runtime/observations.py
python/evolution_sim/mind/policy_inputs.py
python/evolution_sim/mind/recurrent_actor_critic.py
python/evolution_sim/mind/recurrent_experiment.py
python/evolution_sim/mind/recurrent_rollout.py
python/evolution_sim/mind/recurrent_policy.py
python/evolution_sim/mind/recurrent_ppo.py
python/evolution_sim/mind/recurrent_accuracy_screen.py
python/evolution_sim/mind/open_ecology_phase_a_behavioral_evidence.py
python/evolution_sim/mind/recurrent_kernel_development_screen.py
```

It also binds this document's raw-byte SHA-256, the literal Git commit,
repository root, detached-head status, empty tracked and untracked status, and
the imported module `__file__` paths. A source module outside the exact
checkout, an unexpected symlink, or byte drift fails closed.

`corpus` is the prospective, candidate-independent corpus contract from the
preregistration: it contains the frozen selection manifest, all case records
in table order, every selected row/output index, logical and physical shapes,
per-case and aggregate dot/product counts, generator version, formulas, gates,
and seed identities/digests. Generated tensors or candidate outputs do not
belong in `corpus`.

`accuracy.public_forward` contains generated component digests, cases,
per-dot classes/metrics, and summaries for the three ordinary families,
cancellation, and both underflow families. `accuracy.holdout_forward`
contains the corresponding evidence for the one composite holdout family.
Together those are exactly seven generated forward families:
three ordinary plus composite plus cancellation plus two underflow.
Because the real holdout generator may not be called again, each holdout case
also serializes its full candidate output plus the selected input, weight,
bias, candidate result, and oracle result using the same big-endian
binary32-hexadecimal encoding as the component digest. The post-run validator
requires the receipt seed digest to equal the explicitly supplied sealed seed
contract, regenerates the full holdout input, weight, and bias from that seed
and the frozen formulas, and requires their full and selected bytes and digests
to match the record. It also requires the selected candidate result to be the
declared slice of the serialized full candidate output and verifies both
candidate-output digests. It then independently recomputes the holdout oracle
and every metric from those authenticated values without invoking the
candidate again.

`gradients` binds all input/weight/bias/output/loss/direction/gradient/HVP
component digests and exact counts. `children` contains the first four ordered
child records; `baseline_confirmation` is separate. `selection` contains every
named gate, recomputed speed and RSS ratios, reasons, and either the sole
candidate identity or `null`.

If execution fails closed before holdout admission but the parent remains able
to publish a terminal report, the candidate's accuracy/gradient fields and the
root `holdout_receipt` are `null`, selection is unauthorized, and the exact
failure record identifies the stage. If admission occurred, those fields and
the receipt must be complete; partial evidence is never represented as a
successful empty object. A process or machine failure may leave only the
durable attempt marker (and, after admission, the admission marker); that is a
valid permanent retirement state but not a report or result.

The validator does not trust summaries. It regenerates public-family indices
and tensors from this contract; recomputes their component digests and oracle
outputs; checks all case and aggregate counts; reconstructs ordinary,
cancellation, underflow, gradient, and HVP metrics; verifies the one-shot
receipt against an explicit expected seed contract; regenerates the consumed
holdout's input, weight, and bias from that seed; validates all full and
selected component records, including the serialized full candidate output;
and reconstructs every holdout metric without a second candidate execution. It
then reopens every source module, revalidates child exact digests and distinct
nonces, recomputes timing, RSS, semantic, and selection gates, and finally
recomputes `exact_digest` with that field omitted using
`stable_payload_digest`. Any difference rejects the whole report.

## Selection gates

All of the following must pass:

- exact source, module bytes, document bytes, runtime, CPU float32
  single-thread topology, seeds, child nonces, report lifecycle, and one-shot
  receipt reconstruct;
- the complete ordinary and holdout corpus has zero applicable
  standard-bound violations and zero D04-oracle tolerance violations;
- every cancellation dot has condition ratio at least `2^18` and D04-oracle
  ratio at most `1.0`;
- underflow is classified and reported only as nondeciding diagnostics;
- candidate inference and gradient-enabled outputs are bitwise equal, all
  first gradients pass `2e-5/2e-6`, all HVPs pass `5e-5/5e-6`, every declared
  tensor row/output is active, and each order's nonzero reference-component
  share is at least `0.95`;
- scalar and bounded-batch D04 core, all 13 bucket cases, and the production
  collector stay within the unchanged `1e-5/1e-6` tolerance, with zero action,
  semantic, identity, hidden-shape, or bootstrap-none mismatch;
- legacy scalar and batched paths preserve behavior semantics and remain
  timing controls; distances to their rounding endpoints are report-only;
- the candidate wins at least four of five paired scalar-versus-batched
  timings; its batched median is at least 3% faster than its own scalar median,
  the faster of the opening/closing legacy scalar medians, and the faster of
  the opening/closing legacy batched medians;
- candidate peak RSS divided by the smaller of the opening and closing legacy
  baseline peak-RSS values is at most `1.10`;
- the final baseline confirmation has the same source, runtime binding,
  semantic fingerprint, and expected legacy numerical class as the opening
  baseline;
- no authority, qualification, training, runtime artifact, action-selection
  change, promotion, or Phase-A output exists.

The full later CUDA D04 remains unchanged: 16 worlds, 16 workers, 128 ticks,
model `mind_public_recurrent_masked_actor_critic_v5`, numeric kernel
`batch_size_tolerance_stable_per_row_bmm_forward_native_gemm_backward_v1`,
13 bucket cases and 1,224 bucket comparisons, exactly 131,072 core
comparisons, exactly 42,300 scalar/batched/paired/numeric collector
transitions, exactly 10,828,800 hidden-component comparisons, positive equal
bootstrap coverage in every timed path, scalar fixed-batch steps `0`, batched
steps equal to transitions, zero semantic or identity mismatch, and batched
median strictly below scalar. The CPU screen cannot satisfy or weaken D04.

## After the one deciding screen

A negative, missing, or malformed report is checksummed, archived to Google
Drive with stream readback and independent `rclone check`, and never rerun.
A positive report is also archived and is only a development sieve. It does
not authorize launch or support a scientific claim.

Only after a positive screen may a new exact source-bound launch-authority
amendment and D04 v9 hardening be written. That later source must explicitly
bind the fixed projection, float/bool bridge and fallback, inference mode,
output materialization, decoded runtime branches, unchanged per-row learner
kernel, shared trainable forward, and exact collector counts. It must be
committed at another exact SHA, pass PR CI and literal-push CI, reseal every
authority record, and run exactly one unchanged full CUDA D04. Only a passing
fresh D04 may proceed through remaining dependency and operational gates and
the Mac two-party guardian. Scientific claims still require exact replay and
preregistered behavioral acceptance.

## References

- PyTorch, [Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
- Nicholas J. Higham,
  [The Accuracy of Floating Point Summation](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf),
  *SIAM Journal on Scientific Computing* 14(4), 1993.
