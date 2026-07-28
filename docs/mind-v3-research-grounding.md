# Mind v3 Research Grounding

Date: 2026-06-14

This note records public-research grounding and adversarial orchestration
guidance for Mind v3 controller work. It is not a lifecycle ledger, training
authorization, promotion authorization, or replacement for the durable
`AGENTS.md` and `docs/mind-v3-autonomous-evolution.md` history.

The local evidence chain remains authoritative: deterministic replay, exact
artifact digests, source-report pins, strict per-seed gates, lifecycle flags,
and durable backup checks. Public papers, libraries, and benchmarks should
shape diagnostics, metadata, adapters, and future harness design; they must not
override the repo contracts.

## Current Campaign Interpretation

As of the v196 failure response, the central failure is not simply "offline RL
needs more data." The observed mechanism was low-specificity feature coverage
collapsing applied overrides to `eat`, with exact feature-hit share at zero and
mask-only matches dominating shadow behavior. The active v197 direction is a
no-training coverage-or-abstention repair design. It should be judged as a
collapse blocker, not as evidence that slice 4 should train.

Historical campaign evidence must be reused rather than rediscovered:

- v53-v63 closed the scalar coefficient, IQL, prior-blend, extraction-only, and
  global actor-bias family as non-promotable.
- v154-v176 closed the tiny support-archive and nearest-neighbor scorer lane as
  strategically saturated.
- v180, v186, and v195 consumed training slices and failed shadow acceptance;
  none authorize runtime integration or promotion.
- v191-v193 established that same-tick occupancy drift must be handled as a
  support-evidence contract issue, not as proof that scripts requested illegal
  actions.
- v196 established that coverage and abstention must be reported directly
  before another training slice is considered.

The privately configured remote-trainer path must remain hygienic. The default
checkout was dirty and stale when inspected, so training must not run there.
Use a separate clean remote checkout from private configuration only after
verifying git status, commit, CUDA visibility, Python/Torch versions, and the
exact source/artifact digests required by the route.

## Core Policy

1. Collapse prevention is not improvement.

   A fail-closed specificity or abstention gate can prevent a v195-style action
   collapse, but it does not prove terminal survivors, broad no-regression, or
   promotion readiness.

2. Specificity is evidence type, not acceptance proof.

   Exact or high-specificity feature hits are better than mask-only hits, but
   they are not causal support, not sequence support, and not enough to
   authorize training by themselves.

3. Strict seeds are not tuning knobs.

   Once a seed, branch, or fixture has helped discover a blocker, it must be
   labeled in reports as diagnostic/red-team or campaign-target evidence. It
   may guide diagnosis under the current lane, but new architecture selection
   and hyperparameter tuning must use separately declared train/selection
   strata.

4. Canonical replay remains the source of truth.

   RLDS, Minari, TorchRL, LeRobot, Gymnasium, PettingZoo, MLflow, W&B, Trackio,
   DVC, lakeFS, Delta, and OCI can inspire adapters and metadata. They do not
   replace canonical JSON/JSONL artifacts, exact replay validation, digest pins,
   or local lifecycle flags.

5. External methods are candidate scaffolds.

   Flow policies, diffusion policies, world models, action chunks, online RL,
   curricula, and population methods may be useful in future routes. They must
   enter through deterministic harness contracts and exact replay verification,
   not through direct runtime trust.

## Offline RL And OOD Support

Relevant public work reinforces the need for support and coverage diagnostics,
but also shows that naive support gates can become too conservative or brittle:

- [CQL](https://arxiv.org/abs/2006.04779) and
  [IQL](https://arxiv.org/abs/2110.06169) motivate conservative/offline
  learning under distribution shift, but the local v53-v63 and v180/v186 lanes
  already show that these families are not enough for the carrion blocker.
- [ReFORM](https://arxiv.org/abs/2602.05051) and its
  [official implementation](https://github.com/MIT-REALM/reform) are useful
  examples of enforcing on-support flow policies by construction. They are
  architecture evidence for later work, not a v197/v198 implementation target.
- [State-Constrained Offline RL](https://openreview.net/forum?id=KcR8ykFlHA)
  argues that constraining only seen actions can over-restrict useful behavior.
  For this repo, a future diagnostic can ask whether a rejected action would
  still return to familiar public-state support, but this must not relax strict
  gates.
- [Adaptive Scaling of Policy Constraints](https://arxiv.org/abs/2508.19900)
  warns against one global constraint scale. v197-style thresholds should be
  reported by seed, fixture, action, and margin; threshold tuning on strict seeds
  should be treated as contamination.
- [Evaluation-Time Policy Switching](https://arxiv.org/abs/2503.12222) is a
  reminder that fallback behavior after abstention is itself a policy. Reports
  must show final post-abstention requested-action distributions and outcome
  deltas.
- [Flow Matching for Offline RL with Discrete Actions](https://arxiv.org/abs/2602.06138)
  and [Causal Flow Q-Learning](https://arxiv.org/abs/2602.02847) are relevant
  for future architecture review. They do not remove the need for public-feature
  audits, sequence context checks, and exact replay.

Minimal diagnostics that fit v197 without starting a new loop:

- Per-seed, per-fixture, and per-action specificity pass/fail tables.
- Exact/high-specificity hit share, mask-only hit share, miss share, and
  abstention reason counts.
- Final post-abstention requested-action distribution, not only blocked
  override share.
- Support target/outlier stats for exact/high-specificity hits: row counts,
  action-value range, and whether a small row family dominates.
- One bounded public-feature train-vs-shadow shift table, or at minimum the
  equivalent per-seed/per-fixture support table.

Avoid in v197/v198:

- Flow/diffusion/causal policy implementation.
- Learned adaptive threshold search.
- State-constrained OOD relaxation in runtime behavior.
- Policy switching runtime behavior.
- Repeated threshold sweeps after seeing strict-seed outcomes.

## Evaluation And Seed Governance

The repo's strict per-seed gates are aligned with
[RLiable / Statistical Precipice](https://arxiv.org/abs/2108.13264), the
[RLiable implementation](https://github.com/google-research/rliable), and
[Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html):
aggregate means are dangerous, especially when seeds are few or targeted.

Future reports should make seed provenance explicit before adding more
statistics:

- `seed_registry_id`
- `seed_role_by_seed`
- `split_generation_method`
- `split_digest`
- `first_seen_report`
- `allowed_uses`
- `used_for_training_or_selection`
- `selection_seed_reuse_count`

Recommended seed roles:

- `train_support`: may provide trainable rows under an explicitly authorized
  route.
- `selection`: may choose among predeclared candidate families or thresholds.
- `diagnostic_red_team`: may discover and explain failures, but must not be
  used silently as training, selection, or promotion evidence.
- `validation`: fresh evaluation after design selection.
- `lockbox`: final strict check only, never used for design iteration.

For the current carrion lane, historical target seeds can remain campaign
evidence only when the route explicitly says so. Do not generalize from six
hand-picked carrion seeds as population evidence, and do not attach confidence
intervals or significance claims unless the sampling frame is explicit.

Off-policy evaluation should remain diagnostic unless behavior-policy
probabilities, action support, and effective sample size are available. Public
work such as [Offline RL Without OPE](https://openreview.net/forum?id=LU687itn08w)
and model-selection/OPE discussions support using OPE cautiously; it is not an
acceptance gate for this simulator today.

Every Mind report should eventually include a contamination ledger:

- `candidate_family_count`
- `configs_tried`
- `artifacts_inspected`
- `diagnostic_reports_inspected`
- `red_team_inputs_used`
- `trainable_rows_from_red_team_count`
- `posthoc_metric`
- `planned_metric`
- `inferential_claims_allowed`

## Dataset And Replay Infrastructure

The highest-value infrastructure move is not a new data stack. It is to make
the current canonical contract more explicit.

Recommended next docs/harness work:

1. Add a canonical dataset contract document.

   Cover header/row/footer format, episode boundaries, seed/branch/agent
   identity, tick-start public action mask, live resolution mask, requested vs
   resolved action, same-tick occupancy drift, terminal/truncated semantics,
   trainable vs provenance-only fields, split policy, and digest calculation.

2. Add a tracker-neutral artifact lineage manifest.

   Fields should include source report digest, dataset digest, schema versions,
   exact command, git SHA and dirty state, Python/dependency versions, hardware
   class, artifact URI, backup URI, lifecycle flags, and consumed slice.
   Trackers can mirror this manifest; they must not become the authority.

3. Add a validation harness spec.

   Use [JSON Schema 2020-12](https://json-schema.org/specification) for object
   shape and code validators for cross-row invariants: footer counts, exactly
   one first row per episode, digest round-trip, mask/action consistency,
   leakage exclusion from trainable payloads, source report existence, and
   branch replay verification.

4. Add bounded property-based tests.

   [Hypothesis](https://hypothesis.readthedocs.io/) is useful for small
   deterministic worlds: replay round trip, branch replay equivalence,
   action-mask legality, summary-only no-viewer/no-events behavior, and stable
   digest under JSON round trip. Keep profiles deterministic and promote shrunk
   failures into fixed regression tests.

5. Treat adapters as lossy views until proven otherwise.

   [RLDS](https://github.com/google-research/rlds),
   [Minari](https://github.com/Farama-Foundation/Minari),
   [LeRobotDataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3),
   [Gymnasium](https://gymnasium.farama.org/), and
   [PettingZoo](https://pettingzoo.farama.org/) are useful export or trainer
   interfaces. Every adapter must carry the canonical dataset digest and a
   mapping-version digest.

Interfaces worth keeping clean:

- `CanonicalDatasetReader`
- `DatasetManifest`
- `EpisodeIterator`
- `TorchBatchAdapter`
- `DatasetExporter(format="rlds|minari|lerobot|parquet")`
- `TrackerSink`
- `ArtifactStoreRef`
- `ReplayVerifier`

Premature for now:

- Reverb/Acme servers as required infrastructure.
- Mandatory MLflow, W&B, or Trackio.
- DVC, lakeFS, Delta, or OCI as required artifact infrastructure.
- LeRobot or TFDS/RLDS migration as canonical storage.
- Great Expectations as a required validation layer.

## Alternative Controller Directions

The repeated lookup/IQL/support-table failures make it reasonable to prepare a
harness-first controller track. That does not mean jumping straight to a bigger
policy. The next viable direction is to build deterministic, replayable
interfaces that can safely host stronger controllers.

Ranked candidate directions:

1. Masked action-chunk offline-to-online RL.

   [Q-chunking](https://arxiv.org/abs/2507.07969) and
   [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) are relevant because
   carrion survival is long-horizon and sparse. The local trap is open-loop
   chunks: every tick in a chunk must be revalidated against the public action
   mask and exact replay.

2. Corrective or contrastive imitation over chunks.

   [Set-Supervised Diffusion Policy](https://arxiv.org/abs/2606.01865) shows how
   negative action chunks can matter, not only successful demonstrations. Local
   implication: failed chunks and collapse actions should become first-class
   diagnostic/training candidates in future routes. Diffusion sampling should
   not be introduced until deterministic discrete decoding is specified.

3. World model as proposal generator, not source of truth.

   [DIAMOND](https://arxiv.org/abs/2405.12399) and related model-based work are
   useful for cheap continuation search. A learned model may propose candidates,
   but no candidate counts unless exact replay in the real simulator validates
   it. The report type should be "model-proposed, replay-verified."

4. GPU-vectorized online RL with hard invalid-action masking.

   [Invalid Action Masking](https://arxiv.org/abs/2006.14171) is directly
   relevant. Penalty-only invalid-action handling is a poor fit for this repo.
   The policy logits and any sequence/chunk decoder must be hard-masked with
   public action masks.

5. Targeted automated curricula.

   Curriculum and generated-task systems can help sparse survival. They are only
   useful if fixture provenance, difficulty, train/curriculum/held-out role, and
   strict canonical evaluation remain separate.

6. Population or evolutionary search.

   Population methods can search architectures and hyperparameters, but strict
   gate seeds cannot become the selection surface. Freeze candidates before
   blind shadow evaluation.

7. Domain-specific behavior pretraining.

   Robot/VLA systems such as Octo/OpenVLA are not direct fits. Borrow the
   pattern: broad behavior pretraining over public observations and actions,
   then tightly scoped finetuning. Do not import robot pretrained weights.

8. Multi-agent/open-ended JAX infrastructure.

   Multi-agent training may eventually matter because this is artificial life.
   Any centralized-training/decentralized-execution experiment must prove that
   runtime policy features use only public observation and public mask fields.

Harness preconditions before GPU controller training:

- Deterministic vectorized environment adapter with public observation and
  public action mask only.
- Exact replay recorder for every rollout, including RNG seeds and checkpoint
  digest.
- Hard action-mask application in logits and chunk/sequence decoding.
- Train, curriculum, selection, validation, and strict-held-out seed split with
  no accidental reuse.
- Frozen shadow evaluator that reports terminal survivors, alive/birth deltas,
  dominant requested-action share, heuristic source count, invalid actions, and
  abstentions.
- Policy artifact schema covering config, dataset digest, code version, GPU
  determinism flags, and replay trace references.
- Negative-example schema for failed chunks and collapse actions.
- CPU smoke tests before GPU: deterministic reset, one-step mask legality,
  replay equivalence, and chunk executor equivalence.
- Explicit GPU run budget and slice accounting.
- No runtime simulator lookahead or hidden-state features unless the report is
  explicitly diagnostics-only.

## Suggested Route Shape

Use the following as orchestration guidance, not as authorization:

1. Finish v197 as a no-training coverage/abstention report.

   It should answer whether the v196 collapse can be blocked without broad
   regressions and without relaxing the strict gates. It should not implement a
   new controller family.

2. If v197 closes the collapse cleanly, pre-register v198 before any slice-4
   route.

   v198 should likely be a harness/data-contract route, not training:
   `v198_harness_for_masked_online_and_chunk_policies_no_training`.

3. Only after the harness route passes, run a small pilot.

   Candidate route:
   `v199_masked_action_chunk_pilot_no_runtime_integration`, with exact replay,
   per-tick mask validation, negative-example handling, and no promotion claims.

4. Keep slice accounting explicit.

   Training slices already consumed by v180, v186, and v195 remain spent.
   Nothing in this research note authorizes slice 4.

## Orchestrator Prompting Pattern

For future coders or agents, include these fields in the prompt:

- Exact current route and whether it is training, diagnostics-only, support
  generation, audit-only, or harness-only.
- Required upstream report and dataset digests.
- Historical dedupe list: which prior lanes are closed and must not be
  rediscovered.
- Explicit forbidden actions: no runtime integration, no promotion, no gate
  relaxation, no unsupported training slice, no dirty trainer checkout.
- Write scope and files likely involved.
- Required output report path, classification string, lifecycle flags, and
  exact digest command.
- Required validation commands.
- Remote trainer policy: use only the clean checkout; verify git status,
  commit, CUDA, Python, Torch, and dependency state before running.
- Second-pass adversarial check: ask what would make this result misleading,
  what hidden overfit or contamination could exist, and which metrics could
  create false confidence.

Good agent split for research-heavy turns:

- Offline RL/OOD adversary: challenge support, specificity, and abstention
  assumptions.
- Controller architecture scout: find alternatives beyond another lookup table.
- Evaluation auditor: focus on seed roles, contamination, OPE limits, and
  report-card requirements.
- Dataset/replay infrastructure auditor: keep canonical artifacts durable while
  borrowing external standards only where they help.

Run agents twice when the stakes justify it: first pass for breadth, second
pass to challenge their own recommendations and rank what is safe to adopt now.

## Source Map

Offline support and constraints:

- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)
- [Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
- [ReFORM](https://arxiv.org/abs/2602.05051)
- [ReFORM implementation](https://github.com/MIT-REALM/reform)
- [State-Constrained Offline RL](https://openreview.net/forum?id=KcR8ykFlHA)
- [Adaptive Scaling of Policy Constraints](https://arxiv.org/abs/2508.19900)
- [Evaluation-Time Policy Switching](https://arxiv.org/abs/2503.12222)
- [Flow Matching for Offline RL with Discrete Actions](https://arxiv.org/abs/2602.06138)
- [Causal Flow Q-Learning](https://arxiv.org/abs/2602.02847)

Controller alternatives:

- [Q-chunking](https://arxiv.org/abs/2507.07969)
- [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544)
- [Set-Supervised Diffusion Policy](https://arxiv.org/abs/2606.01865)
- [DIAMOND](https://arxiv.org/abs/2405.12399)
- [Invalid Action Masking](https://arxiv.org/abs/2006.14171)

Evaluation and reproducibility:

- [RLiable / Statistical Precipice](https://arxiv.org/abs/2108.13264)
- [RLiable repository](https://github.com/google-research/rliable)
- [Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)
- [Offline RL Without OPE](https://openreview.net/forum?id=LU687itn08w)

Dataset, replay, and adapters:

- [RLDS](https://github.com/google-research/rlds)
- [Minari](https://github.com/Farama-Foundation/Minari)
- [LeRobotDataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [Gymnasium](https://gymnasium.farama.org/)
- [PettingZoo](https://pettingzoo.farama.org/)
- [JSON Schema](https://json-schema.org/specification)
- [Hypothesis](https://hypothesis.readthedocs.io/)
