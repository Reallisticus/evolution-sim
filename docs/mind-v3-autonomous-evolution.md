# Mind v3 Autonomous Evolution

Mind v3 is the target controller architecture for this simulator. It removes
heuristic action selection from the agent runtime. The simulator still defines
the world actuator API and physics: actions such as `move_north`, `eat`,
`drink`, `attack_*`, `stay`, `mate`, and signals remain legal world operations.
Mind v3 decides among legal actions without heuristic fallback, hard guards, or
confidence delegation.

Development remains feature-gated until v3 proves survival and reproduction,
but that gate is a safety boundary, not a product direction. The heuristic is
only an external comparison baseline for v3.

The goal is not to discover one hyper-optimized monoculture that dominates the
world. The target is durable ecological diversity: multiple lineages using
terrain, diet, movement, communication, and reproduction pressures differently
enough that emergent inter-species behavior can be observed rather than erased
by a single global winner.

## Current First Slice

- `MindV3EvolutionPolicy` is a pure autonomous policy implementing the existing
  `Policy` protocol.
- Founders receive bounded controller metadata in
  `agent.mind_inheritance_metadata`.
- Children inherit and mutate bounded controller metadata through reproduction.
- Runtime action selection uses masked controller scores over policy-visible
  observations.
- Within-run reward-modulated controller updates apply a short eligibility
  trace, so later resource/survival reward can credit recent actions without
  hard-coded navigation behavior.
- `collect_trajectory --mind-v3-autonomous-evolution` records v3 trajectories
  with zero heuristic action sources.
- `sim:mind:v3:evaluate` compares heuristic baseline and Mind v3 directly on
  survival, births, deaths, and action-source counts.
- `sim:mind:v3:evolve` runs a small heuristic-free population search over
  founder controller templates and writes the best controller metadata back into
  the report.
- `sim:mind:v3:evaluate --founder-template <report>` and runtime
  `--mind-v3-founder-template <report>` flags can replay a searched controller
  template through evaluator, trajectory collection, and headless replay paths.

## Non-Goals

- No post-hoc extraction caps.
- No fallback-rate micro-optimization.
- No heuristic guard or confidence delegate inside the Mind v3 runtime.
- No default-runtime promotion until autonomous survival and reproduction are
  measured across held-out seeds and longer horizons.
- No unlimited parent memory or unbounded model copying through reproduction.

## Acceptance Boundary

The first Mind v3 milestone is not promotion-level survival. It is honest
autonomous measurement:

- Mind v3 action-source counts contain zero heuristic actions.
- Every founder has bounded inherited controller metadata.
- Every child produced under v3 inherits bounded controller metadata.
- The evaluator reports survival, births, deaths, action-source counts, and
  policy ids.
- A multi-seed long-horizon report is recorded even if survival is poor.

## First Baseline Result

The first two-seed 120-tick smoke reached zero heuristic action sources, but it
collapsed: Mind v3 alive mean `0.0`, births mean `0.0`; the heuristic baseline
on the same seeds reached alive mean `49.5`, births mean `44.5`. This is a real
autonomous baseline and a clear blocker. The next v3 work should add a stronger
population-search loop over inherited controller state rather than weakening
the no-heuristic boundary.

## Population Search Smoke

The first searched-template smoke is intentionally small:

- Search command:
  `npm run sim:mind:v3:evolve -- --seeds 5,13 --ticks 80 --population-size 4 --generations 2 --output output/mind/mind-v3-evolution-search-smoke.json`
- Baseline v3 comparison at 80 ticks:
  `output/mind/mind-v3-autonomous-evolution-80tick-baseline.json`
- Searched-template evaluation:
  `output/mind/mind-v3-evolution-search-template-eval.json`

Checked result on seeds `5,13` at `80` ticks:

- Raw Mind v3: alive mean `0.5`, births mean `0.0`,
  heuristic action-source count `0`.
- Searched template: alive mean `4.0`, births mean `0.5`, deaths mean `16.5`,
  heuristic action-source count `0`.
- Heuristic baseline on the same evaluation: alive mean `28.0`, births mean
  `18.0`.

This is a real improvement over the raw v3 founder initialization, but it is not
a working autonomous mind. The gap to the heuristic baseline remains large:
`-24.0` alive mean and `-17.5` births mean at `80` ticks. The next useful v3
milestone is to scale selection pressure and diversity, not to reintroduce
heuristic fallback.

## Quality-Diversity Search

Mind v3 search now has a resumable quality-diversity loop:

- `sim:mind:v3:evolve --resume-from <report>` resumes from durable
  `resume.next_candidates` and `resume.rng_state`.
- Search reports include a `quality_diversity_archive_v2` with balanced,
  survival, births, resource-use, scavenger-lane, movement, and low-death
  elites, plus `behavior_niches` keyed by dominant terrain, trophic role, and
  meat mode.
- Next-generation parents are drawn from both metric elites and behavior-niche
  lineage templates, so a niche can seed children even when it is not the
  single best global scorer.
- Scoring is `survival_dominant_quality_diversity_v4`: final alive count and
  survival area dominate, while resource and movement events are bounded
  behavior descriptors rather than primary targets. The v4 score also gives a
  bounded bonus for active behavior niches and a bounded penalty for
  dominant-action collapse.
- Optional `--holdout-seeds` evaluates the selected controller after search.
- Founder search candidates are stratified across the built-in
  `specialization_profile` priors instead of relying on random coverage. Search
  artifacts can export an
  `archive_diverse_founder_template_pool_with_identity_diagnostics_v2` founder
  pool, and runtime/evaluation paths initialize founders deterministically
  across that pool instead of cloning every founder from one selected
  controller. Reports include template-pool fingerprint counts so fixture
  rerank cannot silently pretend identical pools are distinct candidates. Pool
  entries must either clear
  `founder_template_pool_min_alive_agents_mean` or have births, so known-weak
  explored profiles stay diagnostic rather than being forced into the promoted
  founder pool.

Checked result at `120` ticks:

- Search:
  `output/mind/mind-v3-quality-diversity-survival-search.json`
- Standard evaluation:
  `output/mind/mind-v3-quality-diversity-survival-120tick-eval.json`
- Result across seeds `5,13,19,29,37`: searched v3 alive mean `0.6`,
  births mean `0.2`, heuristic action-source count `0`; heuristic baseline
  alive mean `45.6`, births mean `39.0`.

This failed as a long-horizon controller milestone. The search found resource
loops and small birth events, but did not preserve enough agents through `120`
ticks.

Lineage-selection rerun on the same `80,120` curriculum command, before
behavior-niche archive parents:

- Search: `output/mind/audit/mind-v3-lineage-curriculum-80-120-search.json`
- The `80`-tick training best reached score `18.5695`, alive mean `4.0`,
  births mean `0.6667`, movement rate `0.0506`, and zero heuristic actions.
- The same selected template failed `80`-tick holdout: alive mean `0.5`, births
  mean `0.0`, movement rate `0.0`, resource rate `1.0`, and only one unique
  requested action on average.

This is direct evidence of overfit local resource loops. Runtime lineage
selection is necessary, but not sufficient; the archive must preserve
behavioral niches explicitly instead of relying on a single balanced parent.

Behavior-niche archive rerun on the same command:

- Search: `output/mind/audit/mind-v3-niche-curriculum-80-120-search.json`
- The `80`-tick stage passed holdout: alive mean `3.5`, births mean `0.0`,
  movement rate `0.0111`, resource rate `0.9889`, zero heuristic actions.
- The `120`-tick stage failed holdout: alive mean `0.0`, births mean `0.0`,
  movement rate `0.0`, resource rate `1.0`, zero heuristic actions.
- Five-seed eval of the selected template reached `3.6` alive mean and `0.4`
  births at `80` ticks versus heuristic `27.8` / `16.0`, and `0.2` alive mean
  and `0.4` births at `120` ticks versus heuristic `45.6` / `39.0`.

This is a real archive/selection correction, not a promotion-quality
controller. The next bottleneck is controller behavior: the selected policies
still overfit immediate eat/drink loops and do not discover durable movement,
mate timing, or longer-horizon survival strategies.

Behavior-gated curriculum rerun with explicit holdout diversity floors:

- Search:
  `output/mind/audit/mind-v3-diversity-gated-curriculum-80-120-search.json`
- Command floors: alive `1.0`, births `0.0`, movement rate `0.005`, unique
  requested actions `2`, behavior niches `4`.
- The `80`-tick stage passed: alive mean `3.5`, births mean `0.0`, movement
  rate `0.0111`, unique requested actions mean `5.0`, and behavior niche count
  `11`.
- The `120`-tick stage failed: alive mean `0.0`, births mean `0.0`, movement
  rate `0.0`, unique requested actions mean `1.0`, and behavior niche count
  `9`.

This gate does not improve the controller. It prevents us from mistaking
single-action local resource loops for ecological progress.

Action-collapse diagnostics are now first-class report fields:

- each run, aggregate, and candidate can report `requested_action_counts` and
  `resolved_action_counts`;
- holdout aggregates report `dominant_requested_action`,
  `dominant_requested_action_count`, and `dominant_requested_action_share`;
- holdout aggregates also report `active_behavior_niche_count` and
  `active_behavior_niche_keys`, where a niche only counts as active if a lineage
  elite was alive, moved, or reproduced;
- curriculum gates can enforce
  `--curriculum-max-holdout-dominant-action-share` and
  `--curriculum-min-holdout-active-behavior-niches`.

This is an audit hardening slice, not a controller improvement. It closes the
obscure gap where static terrain/trophic/meat niches could look diverse while
the holdout behavior had collapsed to one action.

Diversity-scored strict curriculum rerun:

- Search:
  `output/mind/audit/mind-v3-diversity-scored-strict-curriculum-80-120-search.json`
- Five-seed eval at `80`:
  `output/mind/audit/mind-v3-diversity-scored-strict-80tick-eval.json`
- Five-seed eval at `120`:
  `output/mind/audit/mind-v3-diversity-scored-strict-120tick-eval.json`
- The `80`-tick stage passed: holdout alive mean `3.5`, births mean `0.0`,
  movement rate `0.0111`, unique requested actions mean `5.0`, dominant action
  share `0.5689`, behavior niches `11`, and active behavior niches `8`.
- The `120`-tick stage failed: holdout alive mean `0.0`, births mean `0.0`,
  movement rate `0.0`, unique requested actions mean `1.0`, dominant action
  share `1.0`, behavior niches `9`, and active behavior niches `0`.
- Five-seed eval of the selected last-passing template reached `3.6` alive mean
  and `0.4` births at `80` ticks versus heuristic `27.8` / `16.0`, and `0.2`
  alive mean and `0.4` births at `120` ticks versus heuristic `45.6` / `39.0`.

This is another falsification of search-only progress. Diversity-aware scoring
and stricter gates prevent hidden action collapse from passing, but they do not
give the current v3 controller the credit assignment or capacity needed for
long-horizon survival.

Checked result at `80` ticks:

- Search:
  `output/mind/mind-v3-quality-diversity-survival-80tick-search.json`
- Raw v3 comparison:
  `output/mind/mind-v3-raw-80tick-five-seed-eval.json`
- Searched-template evaluation:
  `output/mind/mind-v3-quality-diversity-survival-80tick-eval.json`
- Result across seeds `5,13,19,29,37`: raw v3 alive mean `0.8`, births mean
  `0.0`; searched v3 alive mean `3.8`, births mean `0.2`; heuristic baseline
  alive mean `27.8`, births mean `16.0`.

This is real end-to-end progress, but still not close to a working autonomous
mind. The next v3 milestone should make curriculum explicit: search at a
shorter horizon, promote only controllers that pass holdout survival, then
extend to `120` and `240` ticks. Reproduction remains the weak signal.

## Holdout-Gated Curriculum Search

Mind v3 search now supports an explicit tick curriculum:

- `sim:mind:v3:evolve --curriculum-ticks 80,120 --holdout-seeds 29,37`
  runs full search stages in order.
- A stage advances only if the selected controller clears the configured
  holdout floors (`--curriculum-min-holdout-alive`,
  `--curriculum-min-holdout-births`,
  `--curriculum-min-holdout-movement-rate`,
  `--curriculum-min-holdout-unique-actions`, and
  `--curriculum-min-holdout-behavior-niches`).
- The final report remains a loadable `mind_v3_evolution_search_v1` report,
  with additional `curriculum` and `stages` fields.

Checked run:

- Search:
  `output/mind/mind-v3-curriculum-80-120-search.json`
- Evaluation at `80` ticks:
  `output/mind/mind-v3-curriculum-80tick-eval.json`
- Evaluation at `120` ticks:
  `output/mind/mind-v3-curriculum-120tick-eval.json`
- Command:
  `npm run sim:mind:v3:evolve -- --seeds 5,13,19 --holdout-seeds 29,37 --curriculum-ticks 80,120 --curriculum-min-holdout-alive 1.0 --curriculum-min-holdout-births 0.0 --population-size 12 --generations 3 --search-seed 733 --output output/mind/mind-v3-curriculum-80-120-search.json`

Result:

- Stage `80` passed holdout survival: holdout alive mean `2.5`, births mean
  `0.0`, heuristic action-source count `0`.
- Stage `120` failed the holdout alive floor: holdout alive mean `0.5`, births
  mean `0.0`, heuristic action-source count `0`.
- The report selected the last passing stage (`80` ticks) rather than the
  failed longer-horizon candidate.
- Five-seed evaluation of the selected template at `80` ticks reached alive
  mean `3.4`, births mean `0.2`, versus heuristic `27.8` / `16.0`.
- Five-seed evaluation of the selected template at `120` ticks reached alive
  mean `0.4`, births mean `0.2`, versus heuristic `45.6` / `39.0`.

This is not a promotion improvement over the previous best `80`-tick searched
template (`3.8` alive mean, `0.2` births mean), but it is a useful falsification
of the current controller/search scale: the same discovered behavior does not
survive horizon extension. The next v3 work should target stronger closed-loop
credit assignment or richer controller capacity before running larger
curricula.

## Need-Gated Navigation Controller v4

2026-05-11 audit update: the contract and architecture now match the code. The
Mind v3 contract previously claimed `policy_visible_self_state_only` even
though the current controller already consumed local patch and bounded
navigation inputs. The contract now declares
`policy_visible_self_local_patch_navigation`, keeps
`mind_inheritance_available` excluded, and lists the derived interaction
features explicitly.

New founders now use
`need_gated_local_navigation_feature_projection_linear_action_head_v4`. It keeps
the deterministic linear action head, but expands the projection from sixteen
to twenty-four hidden features by adding thirst-gated water direction,
plant-hunger direction, meat-hunger carrion direction, and predatory-hunger prey
direction. These are products of already policy-visible observation fields; the
controller gets no wider FOV and no privileged fixture/world state.

This is a capacity fix, not a promotion claim. It addresses the concrete
limitation that a linear head over separate `thirst` and `navigation.water.dy`
features cannot robustly express "move toward water only when thirsty" without
an interaction feature. Search/evaluation still has to prove that the new
capacity improves survival, reproduction, and controlled-fixture behavior.

## Local/Navigation Controller v3

2026-05-10 audit update: the v3 controller-capacity bottleneck was verified in
code, not inferred from docs. The `homeostatic_feature_projection_linear_action_head_v2`
founder architecture scored only self-state features; changing local patch food,
fresh-kill/carcass, and navigation vectors produced zero action-score delta.

New founders now use
`local_navigation_feature_projection_linear_action_head_v3`. The inherited
action head remains bounded and heuristic-free, but its feature projection now
uses only policy-visible observation-input fields from self state, the center
local patch, and bounded water/plant/carrion/prey navigation targets. Legacy
`fixed_random_feature_projection_linear_action_head_v1` and
`homeostatic_feature_projection_linear_action_head_v2` metadata remain loadable
with their original eight-unit layout; new v3 founders use sixteen hidden
features and `diverse_local_navigation_action_prior_v2`.

Measured smoke results are mixed and should not be treated as promotion
evidence:

- Raw five-seed v3 founder eval at `80` ticks:
  `2.2` alive / `0.2` births, heuristic action-source count `0`.
- Raw five-seed v3 founder eval at `120` ticks:
  `0.0` alive / `0.2` births, heuristic action-source count `0`.
- A small `80`-tick search smoke over seeds `5,13` found an in-sample candidate
  with `3.5` alive / `1.5` births and a noncollapsed training action mix, but
  holdout seeds `19,29` still collapsed to `eat` with dominant requested-action
  share `0.9958` and `0.0` births.

Conclusion: the representation bug is fixed, but the current reward/search loop
still overfits immediate intake. The next v3 work should target objective and
credit assignment pressure, not more documentation or larger blind sweeps.

## Previous Homeostatic Controller v2

Mind v3 founder controllers previously used
`homeostatic_feature_projection_linear_action_head_v2` for new searches. This
keeps bounded inherited action-head weights and masked argmax action selection,
but replaces the old fixed pseudo-random projection with eight explicit
homeostatic features over policy-visible observation inputs: energy need,
hydration need, injury/risk pressure, reproduction drive, local resource
context, water context, movement pressure, and trophic/mode/signal context.
The feature projection is intentionally limited to explicit self-state indices
from the encoded policy input. A visibility audit removed
`mind_inheritance_available` from this projection because it describes
controller-state availability rather than ecological perception; regression
tests now require score invariance when patch/navigation inputs or that
controller-private availability bit are changed.

A second audit found a credit-assignment boundary bug: v3 transition updates
were crediting `resolved_action` first. In resolution races, where a requested
action was legal in the observation mask but became invalid before execution,
that updated `stay` instead of the policy-selected action. The original fix was
`policy_valid_requested_action_short_eligibility_trace_v2`: valid requested
actions received credit, while truly invalid requested actions still fell back
to the resolved action. This was later extended to
`policy_valid_requested_action_horizon_eligibility_trace_v3` for longer
movement-to-resource paths. Rerunning the selected template after the original
boundary fix preserved
the visibility-hardened metrics: `6.2` alive / `0.4` births at `80` ticks and
`0.8` / `0.4` at `120` ticks, with zero heuristic action-source counts.

New founders also record `diverse_homeostatic_action_prior_v1` and a bounded
`specialization_profile` (`forager`, `hydration_seeker`, `disperser`,
`reproducer`, or `predator_scavenger`). This is an initialization and
representation bias, not heuristic fallback: runtime decisions are still
controller scores over legal actions, and action-source counts remain zero for
heuristics.

Checked result:

- Raw v2-founder five-seed eval at `80` ticks:
  `output/mind/audit/mind-v3-homeostatic-raw-80tick-eval.json`
- Raw v2-founder five-seed eval at `120` ticks:
  `output/mind/audit/mind-v3-homeostatic-raw-120tick-eval.json`
- Curriculum search:
  `output/mind/audit/mind-v3-homeostatic-curriculum-80-120-search.json`
- Selected-template five-seed eval at `80`:
  `output/mind/audit/mind-v3-homeostatic-curriculum-selected-80tick-eval.json`
- Selected-template five-seed eval at `120`:
  `output/mind/audit/mind-v3-homeostatic-curriculum-selected-120tick-eval.json`

Measured outcome:

- Raw v2 founders improved the raw `80`-tick baseline to alive mean `1.4` and
  births mean `0.2` with zero heuristic actions. The raw `120`-tick result
  still collapsed to alive mean `0.0`, births mean `0.2`.
- The `80,120` holdout-gated curriculum completed both stages. Stage `80`
  holdout reached alive mean `4.0`, births mean `0.0`, movement rate `0.0306`,
  unique requested actions mean `5.0`, dominant action share `0.4949`, and
  active behavior niches `10`.
- Stage `120` holdout reached alive mean `1.5`, births mean `0.5`, movement
  rate `0.0131`, unique requested actions mean `4.0`, dominant action share
  `0.5238`, and active behavior niches `7`.
- Five-seed selected-template eval reached alive mean `6.0`, births mean `0.4`
  at `80` ticks, and alive mean `1.0`, births mean `0.4` at `120` ticks, with
  zero heuristic action-source counts.
- After the visibility hardening that removed `mind_inheritance_available` from
  the scorer, the same selected-template rerun reached alive mean `6.2`, births
  mean `0.4` at `80` ticks and alive mean `0.8`, births mean `0.4` at `120`
  ticks, still with zero heuristic action-source counts. The conclusion is
  unchanged: improved over action collapse, far below heuristic, and not
  promotion quality.
- After the requested-action credit hardening, the selected-template rerun
  remained at alive mean `6.2`, births mean `0.4` at `80` ticks and alive mean
  `0.8`, births mean `0.4` at `120` ticks, with zero heuristic action-source
  counts:
  `output/mind/audit/mind-v3-credit-hardened-selected-80tick-eval.json` and
  `output/mind/audit/mind-v3-credit-hardened-selected-120tick-eval.json`.

This is the first v3 slice here that clears the `120`-tick curriculum holdout
instead of merely hardening the gate. It is still far below the heuristic
baseline (`27.8` / `16.0` at `80`; `45.6` / `39.0` at `120`) and should not be
called a working autonomous mind. The next v3 slice should extend the same
representation into stronger credit assignment and reproduction/movement
pressure, then test `120,240` rather than only replaying `80,120`.

## Runtime Lineage Selection

The search loop now records runtime lineage elites instead of treating the
pre-rollout founder template as the only inheritable controller:

- each candidate report carries `controller_lineage_elites` selected from agents
  that survived, reproduced, lived longest, moved, or occupied distinct terrain
  and trophic niches;
- each candidate carries `behavior_descriptors` for terrain occupancy, dominant
  terrain, trophic roles, meat modes, requested actions, lineage count, and
  runtime species/ecotype counts;
- the archive carries bounded `behavior_niches` keyed by
  `(dominant_terrain, trophic_role, meat_mode)`;
- the selected candidate `controller_metadata` is promoted from the best
  runtime lineage elite, while `founder_template_metadata` preserves the
  original pre-run template for auditability;
- promoted search artifacts also carry `founder_template_pool` and
  `founder_template_pool_specialization_profile_counts`; loader/evaluator paths
  consume that pool so evaluation can preserve viable founder-profile diversity
  instead of silently collapsing back to one controller profile;
- next-generation mutation draws from metric elites and behavior-niche lineage
  templates, so terrain/diet/movement variants can seed future candidates
  instead of being collapsed into one balanced parent.

This does not solve the current survival/reproduction gap by itself. It fixes a
search-contract bug: within-run controller adaptation and ecological niche
evidence now have a path into subsequent search generations.

May 10 audit result:

- Search:
  `output/mind/mind-v3-eligible-diverse-pool-80-smoke.json`
- The selected 80-tick artifact exports an 8-template founder pool with
  `hydration_seeker: 6`, `disperser: 1`, and `reproducer: 1`.
- Holdout at 80 ticks (`19,29`) reached `4.0` alive, `0.0` births, dominant
  action `drink` at `0.5712`, active behavior niches `8`, and heuristic action
  source count `0`.
- Five-seed eval at 80 ticks reached `3.6` alive, `0.0` births, dominant action
  share `0.5887`, and `5.8` unique requested actions mean.
- Five-seed eval at 120 ticks reached `0.6` alive, `0.0` births, dominant action
  share `0.5813`, and `5.8` unique requested actions mean.

This is a foundation improvement, not a reproduction breakthrough. It improves
profile diversity and action-collapse diagnostics relative to a single-template
search artifact (`3.4` alive, dominant `drink` share `0.789` at 80 ticks), but
births remain absent and horizon transfer is still failing.

## Reproduction Viability Gate Hardening

The next audit found a more subtle fake-progress path: the v3 score could
reward in-sample terminal reproduction viability while the curriculum gate only
checked held-out alive/birth/action/niche floors. A candidate could therefore
look directionally better in the search aggregate and still have no held-out
biological path to reproduction.

The scoring policy is now
`reproductive_viability_quality_diversity_v6`: alive pressure is lower, terminal
energy/health/matched-diet viability is weighted more directly, an alive
candidate with zero terminal reproduction viability receives an explicit
dead-end penalty, and seed-brittle births/terminal viability lose score. The
curriculum gate also gained held-out floors:

- `--curriculum-min-holdout-energy-viability`
- `--curriculum-min-holdout-health-viability`
- `--curriculum-min-holdout-matched-diet-viability`
- `--curriculum-min-holdout-biologically-ready`

Smoke evidence is deliberately not promotion evidence:

- `output/mind/mind-v3-repro-deadend-penalty-80-smoke.json` selected a
  `disperser`-heavy pool rather than the previous hydration-survival dead end.
- Its holdout reached `3.0` alive, `0.0` births, dominant `drink` share
  `0.3636`, and `6.5` unique requested actions mean.
- The same holdout had terminal energy viability `0.0`, health viability
  `0.125`, matched-diet viability `0.125`, and biologically ready mean `0.0`.
- Five-seed eval reached only `2.4` alive / `0.0` births at `80` ticks and
  `0.2` alive / `0.0` births at `120`.
- A dense own-state reward-shaping probe
  (`output/mind/mind-v3-repro-signal-v2-80-smoke.json`) produced one-seed
  in-sample births but collapsed on holdout (`0.5` alive, `0.0` births, and
  zero terminal viability). That runtime reward change was not retained.
- The retained v6 score rerun
  (`output/mind/mind-v3-robust-repro-score-v6-80-smoke.json`) rejects that
  one-seed birth artifact and selects the robustly less-bad `g0-c2` candidate,
  but its holdout still has terminal energy viability `0.0` and no births.

Conclusion: this slice improves the search contract and reduces action
collapse, but it does not improve the controller. Any future claim of
reproduction pressure must clear nonzero held-out viability floors, not just an
in-sample score component.

## Controlled Ecology Fixture Evaluation

The evaluator now has an opt-in controlled fixture suite instead of relying only
on default-world aggregate survival. `mind_v3_evaluate` can run
`--fixture-suite basic`, which builds real `SimulationWorld` instances with
hand-seeded herbivore, hunter, scavenger, and omnivore founder genomes across
plant-only, carrion-only, prey-rich, and mixed-stable arenas. The fixture path
uses the same runtime policy, action resolution, diet/combat/resource systems,
and `reproduction_end` summary as ordinary runs; it does not compute fixture
outcomes out of band.

The evaluation report now also includes terminal reproduction failure
attribution for normal and fixture runs:

- ready and biologically-ready terminal agents;
- biological blockers (`age`, `cooldown`, `energy`, `hydration`, `health`,
  `matched_diet`);
- blocker shares and terminal viability shares;
- blocker/readiness breakdowns by trophic role and meat mode;
- diet, combat, fresh-kill, carcass, and animal-resource opportunity summaries.

May 10 smoke artifacts:

- `output/mind/mind-v3-controlled-fixture-v1-smoke.json`
- `output/mind/mind-v3-controlled-fixture-v1-80-smoke.json`

The 80-tick fixture smoke is a useful falsification, not a promotion:

- Open world seed `5`: heuristic `37` alive / `19` births, v3 `3` alive /
  `0` births, v3 terminal energy viability `0.0` and matched-diet viability
  `0.0`.
- Plant-only fixture: heuristic `22` alive / `34` births, v3 `1` alive /
  `3` births, v3 terminal energy and hydration viability `0.0`.
- Carrion-only fixture: heuristic `2` alive / `12` births, v3 `1` alive /
  `2` births, v3 terminal energy and hydration viability `0.0`.
- Prey-rich fixture: heuristic `35` alive / `43` births, v3 `5` alive /
  `3` births, v3 terminal hydration viability `0.0`.
- Mixed-stable fixture: heuristic and v3 both ended with `9` alive, but births
  were `9` vs `2`, and v3 terminal health viability was `0.0`.

This makes the next bottleneck sharper: v3 is not merely failing generic
survival. It is failing to keep enough energy/hydration/diet/health readiness
inside terrain/resource contexts where the heuristic can reproduce.

The fixture suite is now wired into both `mind_v3_evaluate` and
`mind_v3_evolve` as an explicit hard regression surface. When
`--fixture-suite basic` is enabled, reports include `fixture_gate` with floors
for alive count, births, mixed-stable births, terminal energy/hydration/health
and matched-diet viability, and biologically-ready terminal agents. Curriculum
search treats a failed fixture gate as a stage stop; non-curriculum search
records the gate verdict and blockers on the selected candidate.

The v3 in-run update signal also changed from component-weighted reward only
to observed outcome-delta readiness credit. The current policy,
`outcome_delta_action_conditioned_readiness_signal_v3`, credits real
post-action deltas in energy/hydration/health distance to reproduction
readiness, plus terminal reproduction/death outcomes. It also adds
action-conditioned pressure: repeated no-gain `eat` is penalized, while
`eat`, `drink`, and movement receive extra credit only when the finalized
transition improves readiness-relevant state. This does not add decision-time
field of view; the controller still chooses from the existing policy-visible
observation and only receives the observed `before`/`after` and `outcome`
record after the action resolves.

May 10 reproduction-delta artifact:

- Search: `output/mind/mind-v3-repro-delta-fixture-gated-80-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-repro-delta-fixture-gated-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-repro-delta-fixture-gated-120-eval.json`
- Two-seed fixture eval: `output/mind/mind-v3-repro-delta-fixture-gated-80-fixtures.json`

Result: this is real controller progress, not promotion. The selected 80-tick
search candidate reached `11.0` alive / `3.3333` births on train seeds and
`8.5` alive / `1.0` births on holdout seeds with zero heuristic actions. The
five-seed eval improved to `9.0` alive / `1.4` births at `80` ticks and `4.4`
alive / `1.8` births at `120`, compared with the previous v3 fixture-era
result of roughly `3.4` / `0.2` at `80` and `0.4` / `0.2` at `120`. The
fixture gate still failed because `carrion_only` ended with `0.0` alive, and
all contexts still show zero biologically-ready terminal agents. The dominant
requested action is still `eat` around `0.64` on open five-seed eval, so the
next bottleneck is not "add more gates"; it is action/behavior credit pressure
that gets hydration, health, movement, and context-specific resource use into
the same reproduction-ready trajectory.

May 10 fixture-rerank/action-credit artifact:

- Search: `output/mind/mind-v3-fixture-rerank-action-credit-80-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-fixture-rerank-action-credit-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-fixture-rerank-action-credit-120-eval.json`
- Two-seed fixture eval: `output/mind/mind-v3-fixture-rerank-action-credit-80-fixtures.json`

This slice adds `--fixture-rerank-top-k`: the search score nominates a bounded
top-K set, then each nominee is evaluated on holdout and controlled fixtures.
Selection prefers fixture gate pass, fewer fixture blockers, mixed-stable
births, fixture aggregate survival/births, holdout births/survival, and lower
holdout action collapse before falling back to the original search score.

Result: selected candidate `g2-c9` reached `12.6667` alive / `4.0` births on
train seeds and `8.5` alive / `3.0` births on holdout seeds. The reranker
rejected raw top candidate `g2-c2` despite stronger train metrics because it
still had a fixture blocker. Five-seed eval improved to `9.8` alive / `3.4`
births at `80` and `4.8` alive / `4.2` births at `120`, with zero heuristic
actions. The two-seed fixture eval passed the configured fixture gate, and
dominant open-world action shifted from `eat` to `drink` around `0.46`-`0.48`.
This is still not promotion: heuristic remains far ahead, terminal
biologically-ready agents are still effectively absent, and mixed-stable
energy/health viability remains weak.

May 10 sustained-core/composite-repair artifact:

- Search: `output/mind/mind-v3-composite-repair-v8-top8-fixture2-80-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-composite-repair-v8-fixture2-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-composite-repair-v8-fixture2-120-eval.json`
- Two-seed fixture eval: `output/mind/mind-v3-composite-repair-v8-fixture2-80-fixtures.json`

This slice fixes two hidden selection issues. First, v3 search's terminal
reproduction viability now includes hydration, matching the real biological
blockers and the fixture gate. Second, top-K rerank can add a bounded composite
founder-template repair candidate: a high-holdout candidate that fails a
fixture blocker may be evaluated with a real archived gate-passing donor
template pool. The repair is selected only after normal holdout and fixture
rollouts; it does not add decision-time FOV or synthetic fixture knowledge.

The selected candidate `g2-c2+repair-g0-c4` uses an 11-template composite pool.
It reached `16.0` train alive / `5.3333` train births and `9.5` holdout alive /
`3.0` holdout births. The two-seed rerank fixture gate passed: plant-only
`11.5` / `14.0`, carrion-only `3.0` / `7.5`, prey-rich `17.5` / `15.0`, and
mixed-stable `14.5` / `4.0`.

Transfer is mixed but useful: five-seed `80` eval regressed versus the previous
best to `9.0` alive / `2.2` births, while five-seed `120` eval improved to
`7.6` alive / `5.0` births. This is horizon-transfer progress, not promotion.
Terminal biologically-ready agents remain `0.0`, and energy remains the main
open-world viability bottleneck.

May 10 real-energy/action-credit artifact:

- Search: `output/mind/mind-v3-real-energy-credit-v10-top8-fixture2-80-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-real-energy-credit-v10-fixture2-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-real-energy-credit-v10-fixture2-120-eval.json`
- Two-seed fixture eval: `output/mind/mind-v3-real-energy-credit-v10-fixture2-80-fixtures.json`

This slice audited the energy blocker below the prior readiness proxy. Search
now records terminal energy-requirement satisfaction from the real
`reproduction_end.energy_readiness_by_meat_mode` totals, and
`mind_v3_evaluate` exposes the same aggregate fields directly in eval reports
under `terminal_reproduction_failure_attribution_v2`. The online reward signal
changed to `real_energy_proxy_action_conditioned_readiness_signal_v4`: it uses
a higher observable energy-ratio target, stronger energy-progress credit, and
a no-gain drink penalty. This does not add decision-time FOV or fixture
knowledge; it only changes how observed post-action self-state deltas are
credited.

Result: selected candidate `g0-c1` reached `17.6667` train alive / `7.0` train
births and `11.0` holdout alive / `4.5` holdout births. The two-seed fixture
gate passed. Five-seed eval improved versus v9 from `9.0` alive / `2.2` births
to `10.8` / `3.8` at `80`, and from `7.6` / `5.0` to `8.8` / `6.8` at `120`.
Real terminal energy satisfaction improved to `0.774` at `80` and `0.744` at
`120`; energy viability improved to `0.2826` and `0.3666`. The tradeoff is also
real: dominant action shifted from `drink` to `eat` at about `0.46`, and
hydration viability dropped to `0.5599` at `80` and `0.5691` at `120`.
Terminal biologically-ready agents remain `0.0`, so this is a real bottleneck
move, not promotion.

Operational note: this two-stage population search is CPU-bound simulator
rollout work. The v3 search CLI now has `--rollout-workers` for process-level
CPU candidate and holdout rollout parallelism, and the report records
`search.rollout_execution` with requested/resolved worker counts. CUDA will
help torch/IQL learner runs, but it will not materially accelerate this specific
v3 search unless the controller training becomes tensor-heavy. For guarded IQL
gate comparisons, `mind_gate` now reports phase timings and can parallelize
train/held-out diagnostics with `--artifact-diagnostics-workers` plus
independent held-out/runtime-mode evaluation entries with
`--evaluation-workers` before moving work to the RTX host.

May 10 balanced-bottleneck/action-collapse artifacts:

- v11 search: `output/mind/mind-v3-balanced-bottleneck-v11-top8-fixture2-80-search.json`
- v11 five-seed eval at `80`: `output/mind/mind-v3-balanced-bottleneck-v11-fixture2-80-eval.json`
- v11 five-seed eval at `120`: `output/mind/mind-v3-balanced-bottleneck-v11-fixture2-120-eval.json`
- v12 top-8 search: `output/mind/mind-v3-balanced-bottleneck-v12-top8-fixture2-80-search.json`
- v12 top-8 five-seed eval at `80`: `output/mind/mind-v3-balanced-bottleneck-v12-fixture2-80-eval.json`
- v12 top-8 five-seed eval at `120`: `output/mind/mind-v3-balanced-bottleneck-v12-fixture2-120-eval.json`
- v12 top-16 search: `output/mind/mind-v3-balanced-bottleneck-v12-top16-fixture2-80-search.json`

This slice made the search and evaluation reports expose terminal balanced
reproduction readiness, energy/hydration balance, energy/hydration gap, and
temporal core-readiness blocker attribution. The online reward changed to
`balanced_bottleneck_hydration_guard_readiness_signal_v6`: it credits observed
before/after self-state deltas, gives full positive credit to the currently
limiting core field, downweights non-limiting positive deltas, and penalizes
`eat` when hydration is limiting and eating does not improve hydration or the
balanced core score. This still does not add decision-time FOV or fixture
knowledge.

v11 selected `g0-c4+repair-g2-c6` and passed the configured two-seed fixture
gate, but five-seed eval exposed the hidden failure: `80` ticks reached `13.2`
alive / `4.2` births and `120` ticks reached `7.6` / `5.8`, while dominant
requested action was still `eat` at about `0.76` and terminal biologically-ready
agents remained nearly absent. v12 selected `g2-c10`; it reduced the v11 eat
collapse to about `0.62` and improved open-world five-seed `120` eval versus
v10 from `8.8` alive / `6.8` births to `14.0` / `8.6`. The tradeoff is real:
hydration regressed versus v10, terminal biologically-ready agents remain
`0.0`, and the two-seed fixture gate failed on `carrion_only`.

A larger v12 top-16 rerank did not fix that by search coverage. It selected
`g3-c14`, but every rerank nominee still failed the fixture gate; the selected
candidate failed `carrion_only` with `0.5` alive and `0.0` hydration viability.
Treat v12 as a partial open-world/horizon donor, not a promotion. The next
non-toy improvement should preserve an explicit carrion/scavenger quality-
diversity lane while keeping the balanced anti-collapse pressure.

May 10 scavenger-lane v13 artifact:

- Search: `output/mind/mind-v3-scavenger-lane-v13-top8-fixture2-80-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-scavenger-lane-v13-fixture2-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-scavenger-lane-v13-fixture2-120-eval.json`

This slice split out an explicit `scavenger` founder profile instead of relying
on the mixed `predator_scavenger` prior, added a named `scavenger_lane` archive
elite, guaranteed scavenger-lane nominees enter fixture rerank beyond the raw
global top-K, and allowed composite repair to use the best scavenger-lane donor
when no fixture-passing donor exists. Lane scoring is recomputed from founder
profiles, behavior descriptors, and lineage elites; metadata tags alone do not
create lane credit.

Final comparable top-8 search selected `g0-c7+repair-g0-c4`, a composite whose
donor was chosen for `scavenger_lane`, with a founder pool of `12` templates
including `4` `scavenger` and `1` `predator_scavenger` templates. The fixture
gate still failed, but the failure narrowed from v12's `carrion_only` alive and
hydration blockers to one blocker: `carrion_only` energy viability `0.0` below
the `0.2` floor. In the same `carrion_only` fixture, v13 had `1.0` alive,
`2.5` births, hydration viability `0.5`, matched-diet viability `0.5`, and
`14` animal-resource consumption events across the two fixture seeds. That is
real carrion-path behavior, not only a report-level metric shift.

Open-world transfer improved too: five-seed eval reached `16.6` alive / `6.2`
births at `80` and `16.0` / `11.8` at `120`, versus v12 `14.4` / `3.8` and
`14.0` / `8.6`. Terminal biologically-ready agents finally became nonzero at
`120` (`1.0` mean), but v13 is still not a promotion: the hard fixture gate
fails, dominant requested action is still `eat` at about `0.66`, and the
primary temporal blocker shifts to hydration at `120`.

May 10 carrion-rerank v14 artifacts:

- Search: `output/mind/mind-v3-carrion-rerank-v14-top8-fixture2-80-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-carrion-rerank-v14-fixture2-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-carrion-rerank-v14-fixture2-120-eval.json`
- `120` fixture eval: `output/mind/mind-v3-carrion-rerank-v14-fixture2-120-gated-eval.json`

This slice parallelized fixture rerank CPU evaluation across candidates and
made the carrion blocker first-class in selection diagnostics. Rerank report
metadata now records `fixture_rerank.execution` with requested, initial, and
repair worker counts. Fixture summaries expose observed carcass/fresh-kill
consumption events and gained energy, and the selector can prioritize terminal
energy satisfaction and observed animal-resource intake when `carrion_only` is
the active blocker.

The reward changed to
`balanced_bottleneck_observed_carrion_readiness_signal_v7`: animal-resource
`eat` gets extra credit only when the finalized action outcome says the agent
ate carcass or fresh kill and before/after self-state readiness improved.
Zero-delta `eat` is treated as no-gain. This still does not add decision-time
FOV or fixture knowledge.

The comparable v14 search selected `g1-c0` and passed the configured `80`-tick
two-seed fixture gate. In `carrion_only`, v14 reached `1.5` alive, `4.0`
births, energy viability `0.5`, energy satisfaction `0.6471`, hydration
viability `0.75`, matched-diet viability `0.75`, and `11.0` observed
animal-resource consumption events per fixture run. It also reduced broad
eval dominant `eat` from v13's about `0.66` to about `0.51`.

This is not a promotion. Five-seed open-world eval regressed versus v13 to
`14.0` alive / `6.0` births at `80` and `14.6` / `11.4` at `120`. The
`120`-tick fixture gate still fails on `carrion_only`: `1.0` alive and `5.0`
births remain, but energy viability, hydration viability, and matched-diet
viability fall below the configured floors. Alternate `80`-gate-passing
candidates `g1-c6` and `g1-c10` were worse on the `120` carrion fixture,
reaching zero carrion survivors. The remaining blocker is multi-horizon
carrion robustness, not a missing fixture rerank nominee.

May 10 multi-horizon rerank v16 artifacts:

- Search: `output/mind/mind-v3-multihorizon-rerank-v16-top8-80-120-search.json`
- Five-seed eval at `80`: `output/mind/mind-v3-multihorizon-rerank-v16-80-eval.json`
- Five-seed eval at `120`: `output/mind/mind-v3-multihorizon-rerank-v16-120-eval.json`
- Rejected challenger eval at `120`: `output/mind/mind-v3-multihorizon-rerank-v15-120-eval.json`

The reranker can now evaluate top-K candidates against multiple fixture
horizons via `--fixture-rerank-ticks`. The report stores combined fixture
blockers, per-horizon summaries, and the selection key used to choose among
candidate failures. This closes the audit gap where a candidate could pass the
80-tick fixture and then fail the 120-tick fixture only after a separate manual
run.

The first multi-horizon selection exposed a useful but unsafe tradeoff. v15
selected `g2-c3+repair-g0-c4`, which improved broad five-seed eval to `17.4`
alive / `14.0` births at `120`, but it regressed the `80` carrion fixture
energy gate. The retained v16 key therefore prefers partial horizon pass
coverage before smaller blocker count when no candidate passes all horizons.
With that promotion-safe rule, v16 selects `g1-c0`: broad eval remains `14.0`
alive / `6.0` births at `80` and `14.6` / `11.4` at `120`, while the combined
80/120 fixture gate honestly fails on `120` carrion-only energy, hydration, and
matched-diet viability. This is a useful falsification and a useful donor
signal, not a promotion event.

May 11 bridge-repair v18 artifacts:

- Search: `output/mind/mind-v3-bridge-lite-rerank-v18-top8-80-120-search.json`
- Five-seed eval at `120`: `output/mind/mind-v3-bridge-lite-rerank-v18-120-eval.json`
- Full-donor bridge check: `output/mind/mind-v3-bridge-rerank-v17-top8-80-120-search.json`

This slice tested the obvious next hypothesis from v16: use an 80-fixture-safe
candidate as the primary and borrow only bounded donor templates from the
rejected v15-style open-world challenger. The reranker now has a second repair
phase whose donor reason is `promotion_safe_bridge`; it only uses primaries
that passed the first fixture horizon and evaluates donor-template limits `1`,
`2`, and `4` before selection.

The hypothesis failed cleanly. v17 full-donor bridges destroyed the 80 carrion
fixture. v18 bounded bridges were still not safe: the smallest one-template
bridge improved holdout births but reached `0.0` carrion-only survivors at the
limiting horizon and failed the combined gate. The selector correctly kept
`g1-c0`, and v18 `120` eval stayed at `14.6` alive / `11.4` births. Treat this
as a useful negative result: simple founder-template mixing cannot transfer the
v15 open-world gains into a promotion-safe controller.

May 11 contextual-founder v19 artifacts:

- Search:
  `output/mind/mind-v3-contextual-template-v19-top8-80-120-search.json`
- Five-seed eval at `80`:
  `output/mind/mind-v3-contextual-template-v19-80-eval.json`
- Five-seed eval at `120`:
  `output/mind/mind-v3-contextual-template-v19-120-eval.json`
- Contextual reevaluation of the prior v18 artifact with fixtures:
  `output/mind/mind-v3-contextual-template-v19-80-fixture-eval.json` and
  `output/mind/mind-v3-contextual-template-v19-120-fixture-eval.json`

This audit found a real hidden bug in the v3 founder-template pool: templates
were assigned by `agent_id % pool_size`. That made bridge repair brittle because
adding a donor template could reshuffle unrelated species onto different
controllers. Runtime founder assignment is now
`contextual_trophic_founder_template_assignment_v1`: when the policy has a
template pool, founders select templates from their own trophic role and meat
mode context. Those are self/genome-derived fields already visible in policy
observations, not external FOV or fixture-specific knowledge.

The fix is real but not a promotion. Re-evaluating the prior v18 artifact under
contextual assignment improved broad `120` eval to `18.0` alive / `12.6` births
and kept heuristic actions at zero, but the `120` fixture gate still failed on
`carrion_only` energy and matched-diet viability. A fresh v19 search with the
same hard 80/120 fixture rerank selected `g1-c0` with strong search-seed
performance (`21.6667` alive / `9.3333` births) but worse five-seed eval
(`14.4` / `4.0` at `80`, `11.4` / `7.8` at `120`) and no first-horizon-safe
carrion candidate for bridge repair. This means the next step should not be a
larger blind scalar search. It should add artifact warm-start or fixture-archive
seeding plus true multi-objective selection so the v18 context benefit is not
lost while searching.

May 11 warm-start and delayed-credit v21 artifacts:

- Sequential warm-start falsification:
  `output/mind/mind-v3-warm-start-v20-top8-80-120-search.json`
- Round-robin warm-start search:
  `output/mind/mind-v3-warm-start-v20-roundrobin-top8-80-120-search.json`
- Round-robin warm-start evals:
  `output/mind/mind-v3-warm-start-v20-roundrobin-80-eval.json` and
  `output/mind/mind-v3-warm-start-v20-roundrobin-120-eval.json`
- Delayed-credit search:
  `output/mind/mind-v3-delayed-credit-v21-top8-80-120-search.json`
- Delayed-credit evals:
  `output/mind/mind-v3-delayed-credit-v21-80-eval.json` and
  `output/mind/mind-v3-delayed-credit-v21-120-eval.json`
- Rejected hydration-risk probe:
  `output/mind/mind-v3-hydration-risk-v22-top8-80-120-search.json`
- Need-gated navigation search:
  `output/mind/mind-v3-need-gated-v23-top8-80-120-search.json`
- Need-gated navigation evals:
  `output/mind/mind-v3-need-gated-v23-80-eval.json` and
  `output/mind/mind-v3-need-gated-v23-120-eval.json`

The first warm-start implementation exposed another hidden provenance bug:
when multiple `--warm-start-report` inputs were supplied, the importer filled
the candidate limit from the first report and never sampled the second report.
Warm-start import is now round-robin across source reports, and fixture rerank
always includes warm-start nominees even when scalar score would drop them.
Reports record warm-start provenance only when warm-start input is actually
used; normal fresh searches keep `warm_start: null`.

The v20 round-robin artifact confirmed that prior archives can improve the
open-world floor, but not enough by themselves: five-seed eval reached `15.4`
alive / `4.4` births at `80` and `11.0` / `8.0` at `120`, with the combined
fixture gate still failing on carrion-only terminal viability. The next
controller bottleneck was delayed navigation credit. The v3 eligibility trace
is now `policy_valid_requested_action_horizon_eligibility_trace_v3`, length
`12` with decay `0.84`, so delayed drink/resource outcomes can credit longer
movement paths without expanding decision-time FOV.

That change produced real transfer rather than paper-only progress. v21 search
selected `g2-c2` with `21.3333` alive / `10.3333` births. Five-seed eval moved
from v20 `11.0` alive / `8.0` births at `120` to `15.8` / `12.2`, with
dominant `eat` down to `0.4388` and zero heuristic actions. It is still not a
promotion: the fixture gate fails on carrion-only hydration at `80`, and
carrion-only energy/hydration/health/matched-diet viability at `120`.

The hydration-risk v22 probe was intentionally rejected. Penalizing observed
carcass `eat` under hydration debt lowered broad search to `14.3333` alive /
`7.3333` births and made carrion fixture viability worse (`0.0` across the
carrion floors). That penalty is not in the default path.

The v23 architecture slice adds need-gated navigation interactions and fixture
pool identity diagnostics. This is mixed but useful progress, not a promotion.
With the same warm-start reports and multi-horizon fixture floors, old v3
warm-start controllers still dominated scalar search score, while a v4
candidate won final rerank by fewer fixture blockers and better holdout. The
selected v23 artifact reached five-seed eval `14.0` alive / `7.0` births at
`80` and `16.4` / `15.2` at `120`, improving v21 births at both horizons but
still failing the combined controlled fixture gate. The next slice should put
fixture outcomes into parent selection/search pressure; another scalar-only
search will likely keep rediscovering broad-eval gains that do not survive the
carrion fixture.

The next search slice starts that fixture-in-loop pressure. When
`--fixture-suite basic` is enabled, `sim:mind:v3:evolve` now defaults to a
bounded per-generation fixture nominee pass via `--fixture-selection-top-k 4`.
The nominee pool is deliberately small but no longer scalar-only: it preserves a
broad-score nominee, a scavenger/ecology-lane nominee, a current-architecture
nominee, and a controlled-readiness nominee before filling remaining slots by
the existing prefilter. Those fixture outcomes add a
`fixture_selection_pressure` score component and a `fixture_selection` archive
elite, so candidates with fewer and less severe controlled-fixture blockers can
seed the next generation before final top-K rerank.

The v25 scoring policy also records fixture blocker pressure. Raw blocker count
still matters, but search pressure now includes the size of each failed
floor gap, with extra weight for carrion-only failures and alive-floor failures.
That is aimed directly at the v23 split: broad-world reproduction improved, but
final rerank candidates stayed at zero carrion-only alive/viability across
80/120 ticks. This remains search-time selection pressure only: runtime
controllers still receive no fixture identity, wider field of view, or heuristic
fallback. Use `--fixture-selection-top-k 0` to recover the previous post-hoc-only
fixture path for audits.

The first bounded v25 slice is diagnostic rather than promotion-grade:
`output/mind/mind-v3-fixture-pressure-v25-top8-80-120-search.json` selected
`g1-c5+repair-g0-c2`, a v4 repair candidate with zero heuristic actions. It
reduced the combined 80/120 fixture blockers from v23's `12` to `9`, improved
80-tick carrion-only alive from `0.0` to `0.5` inside the search rerank, and
made blocker pressure observable in generation reports. Direct five-seed eval
still fails the fixture gate: at 80 ticks it reaches `14.6` alive / `6.2` births
against heuristic `29.4` / `15.6` with five blockers, and at 120 ticks it
reaches `12.2` / `10.8` against heuristic `52.2` / `42.4` with six blockers.
The remaining hard failure is still carrion-only terminal viability at the
longer horizon, so the next slice should target controller credit/selection for
sustained carrion-only survival rather than further broad-world scalar reward.

The v26 slice adds one runtime-learning change and one fixture diagnostic:
movement actions now receive a small capped reward when the action actually
moves toward an already policy-visible need-gated navigation target, and
fixture summaries record carrion-only alive-agent ticks per tick so extinction
time is not hidden behind the terminal alive count. This keeps the controller
honest: no fixture identity, no wider field of view, and no hidden global
resource knowledge.

`output/mind/mind-v3-visible-nav-v26-top8-80-120-search.json` selected
`g0-c3`, still v4 and still zero-heuristic. The broad result improved:
holdout reached `20.5` alive / `9.0` births, direct five-seed eval reached
`17.8` / `8.2` at 80 ticks and `20.2` / `16.4` at 120 ticks. This is the
first recent slice that improves the direct 120-tick broad eval over both v23
and v25. The controlled fixture did not clear: combined 80/120 blockers stayed
at `9`, direct evals still fail with `5` blockers at 80 and `6` at 120, and
carrion-only terminal alive remains `0.0` at 120. The useful signal is that
carrion-only consumption improved inside rerank (`16.5` animal-resource events,
`3.5369` gained energy, `3.6667` alive-agent ticks/tick minimum), but energy is
still the primary temporal readiness blocker and terminal carrion survival is
not sustained. The next slice should separate "find/eat carrion" from "survive
after carrion contact", likely by adding fixture pressure for hazard/occupant
blocked carrion opportunity and post-contact survival, not by increasing the
eat/carrion reward again.

## May 11 Deep-System Audit Direction

The deep system audit (`output/audits/deep-system-audit-2026-05-11.md`) changes
the Mind v3 direction. Foundation, replay, viewer contracts, and the v1/v2
guarded safety floor are strong enough to keep using as the measurement
boundary. Mind v3 also remains the correct strategic track because it removes
heuristic action selection from the agent runtime. The blocker is the controller
class, not the absence of another scalar reward term.

Current v3 founders use a hand-shaped twenty-four-feature projection and a
linear action head with bounded selected-action updates. That architecture is
useful for interpretable falsification, but the evidence now says it is
underpowered for delayed ecological credit: long-horizon energy/hydration
balance, carrion-only survival, reproduction timing, role specialization, and
durable lineage diversity. The v23-v26 sequence improved broad five-seed
survival/birth metrics and reduced action collapse, but it still split from the
controlled fixture gate. In particular, v26 reached `20.2` alive / `16.4`
births at `120` ticks while still failing the combined fixture gate with
carrion-only terminal alive at `0.0`.

The next v3 phase is therefore a deterministic autonomous-controller
training/evolution platform:

- keep the no-heuristic runtime boundary;
- add `mind_horizon_labels_v1` and fixture blocker labels before changing the
  promoted controller path;
- split ecological policy inputs from controller diagnostics so private fields
  such as `mind_inheritance_available` cannot leak into stronger learners;
- add a small deterministic neural or compact recurrent v3 artifact beside the
  current linear controller, with masked logits and horizon/readiness heads;
- use the quality-diversity archive, fixture lanes, and rerank machinery as
  selection infrastructure, not as a substitute for policy capacity.

The next five implementation tasks are ordered to keep the transition
measurable:

1. Generate `mind_horizon_labels_v1` from existing trajectory/evaluation data:
   survival by horizon, reproduced-by-horizon, terminal energy/hydration/health
   and matched-diet viability, carrion-contact survival, and alive-agent ticks.
2. Extract structured fixture blocker labels, including carrion opportunity,
   hazard/occupant blocking, and post-contact survival.
3. Split policy-visible ecological input from diagnostic/controller-private
   fields and add contract tests for the split.
4. Add a deterministic `MindV3NeuralArtifactPolicy` inference path as an
   opt-in artifact type; keep the linear v3 controller as the baseline.
5. Train/evolve the first tiny neural artifact against horizon and fixture
   labels, then compare it against the current linear v3 on the same
   `80`/`120` broad and controlled-fixture matrix.

Do not treat this as permission to skip the current gates. The existing linear
v3 controller remains the honest baseline until the stronger artifact improves
held-out broad outcomes and fixture blocker pressure together.

Initial implementation milestone:

- `sim:mind:horizon-labels` writes `mind_horizon_labels_v1` reports from one or
  more trajectory JSONL files. Labels include observed/censored survival and
  reproduction outcomes at configured future horizons, terminal/target
  energy-hydration-health viability, matched-diet when decodable from the
  observation input, and post-contact animal-resource survival.
- `sim:mind:fixture-labels` writes `mind_fixture_blocker_labels_v1` reports from
  Mind v3 eval/search reports. Labels convert controlled fixture floors into
  per-fixture metric gaps and pressure, including carrion-only and alive-floor
  weighting.
- `mind_ecological_policy_input_v1` defines the safe ecological policy vector
  for stronger v3 artifacts by dropping controller-private diagnostics such as
  `self.mind_inheritance_available` while retaining self ecology, local patch,
  and navigation inputs.

This milestone is complete: v3 trajectory data and fixture-gated v3 reports can
produce loadable label reports, focused tests cover the contract, and the next
artifact path consumes `mind_ecological_policy_input_v1` instead of the raw
observation vector.

Second implementation milestone:

- `sim:mind:v3:train-neural` writes a frozen
  `mind_v3_neural_policy_artifact_v1` artifact from trajectory JSONL plus a
  matching `mind_horizon_labels_v1` report. Optional
  `mind_fixture_blocker_labels_v1` input contributes bounded global fixture
  pressure, recorded as `global_fixture_floor_gap_action_bias_v1`, without
  introducing fixture-specific world knowledge at runtime.
- The artifact uses a deterministic pure-Python fixed-projection MLP with
  masked action logits plus serialized survival and reproduction horizon heads.
  Runtime inference is dependency-free and uses only the ecological input
  contract; `self.mind_inheritance_available` remains excluded.
- `MindV3EvolutionPolicy` now accepts this artifact as an opt-in frozen backend.
  The linear inherited controller remains the default and the baseline. Neural
  artifacts do not mutate weights inside simulation ticks; they can be used via
  `sim:mind:v3:evaluate -- --neural-artifact ...` and can be compared against
  the current linear controller with `--compare-linear-baseline`,
  `sim:run -- --mind-v3-autonomous-evolution --mind-v3-neural-artifact ...`,
  and `sim:trajectory -- --mind-v3-autonomous-evolution
  --mind-v3-neural-artifact ...`.

This milestone is complete: a small artifact can be trained from real v3
trajectory labels, loaded by the v3 evaluator, and smoke-evaluated without
heuristic fallback or controller-private input leakage. The first comparable
run found a real failure mode: raw neural action logits collapsed back toward
`eat`, and the initial frozen-artifact runtime accidentally disabled the online
linear-controller update path that the v3 safety floor still depends on.

Third implementation milestone:

- The neural trainer now records a damped action-frequency prior
  (`action_prior_log_weight=0.18`) and stronger contextual prototype scale
  (`prototype_weight_scale=2.0`) so the artifact is less directly dominated by
  logged action frequency.
- Fixture blocker labels now feed the artifact through the actual
  `pressure` field. The previous implementation read a stale
  `pressure_weight` key, so `fixture_action_bias_delta` stayed zero even when
  the fixture label report contained carrion-only blockers.
- The v3 runtime now uses
  `linear_controller_margin_guarded_neural_residual_v2` when a frozen neural
  artifact is loaded. Neural weights remain immutable during a run, but the
  inherited linear anchor controller keeps receiving the same bounded
  reward-modulated updates as the linear baseline. The neural residual is
  bounded and shadowed when the artifact's local top action is the known
  collapsed `eat` mode or would override a nontrivial linear-anchor margin.

Measured on the v26 template/artifact slice:

- Artifact: `output/mind/mind-v3-v26-neural-artifact-v2.json`
- 80-tick broad comparison:
  `output/mind/mind-v3-v26-neural-guarded-online-anchor-compare-80.json`
  produced v3-neural-anchor `19.0` alive / `11.0` births versus linear
  `20.5` / `10.0`; dominant requested action share was `0.4224`.
- 120-tick broad comparison:
  `output/mind/mind-v3-v26-neural-guarded-online-anchor-compare-120.json`
  produced v3-neural-anchor `16.0` alive / `16.0` births versus linear
  `23.5` / `18.5`; dominant requested action share was `0.4264`.
- 120-tick controlled fixture comparison:
  `output/mind/mind-v3-v26-neural-guarded-online-anchor-fixture-120.json`
  reduced the hard fixture blockers to `5`, all in `carrion_only`, with
  `blocker_count_delta=0` versus the linear baseline.

Fourth implementation milestone:

- Runtime decision diagnostics now expose the neural top action, linear-anchor
  action, anchored action, score margins, residual shadow reason, residual
  applied/shadowed counts, and linear-to-anchored transition counts. The
  evaluator aggregates these under `mind_v3_neural_anchor_diagnostics_v1` for
  broad runs and controlled fixtures.
- `sim:mind:v3:evaluate --fixture-names ...` can run a bounded fixture subset,
  so carrion-only drilldowns do not require rerunning every controlled arena.

Measured on the v26 template/artifact slice:

- Drilldown:
  `output/mind/mind-v3-v26-neural-margin-guard-carrion-120.json`
- Broad `120`-tick comparison moved from the previous neural-anchor `16.0`
  alive / `16.0` births to `19.0` / `16.0`; the linear baseline on the same
  seeds remained better at `23.5` / `18.5`.
- The margin guard reduced residual action changes from `12.08%` in the
  prior diagnostic run to `8.69%` broad and `10.64%` on carrion-only. Most
  residuals are now shadowed by the collapse guard or linear-margin guard.
- Carrion-only did not improve: terminal alive stayed `0.0`, births were
  `2.5` versus linear `3.0`, and the fixture gate still failed on alive,
  energy, hydration, health, and matched-diet viability.

This is not a promotion. It is a safety-ratchet milestone: the neural path no
longer collapses broad action mix, and the margin guard limits cases where the
neural residual harms a confident linear action. It still depends heavily on
the online linear anchor and still fails the carrion-only controlled fixture.
The next v3 milestone is explicit: reduce anchor dependence only when the
artifact is within `2` alive agents of the linear baseline at 120 ticks, matches
or beats linear births, keeps dominant action share at or below `0.50`, and
does not add fixture blockers. Because this slice did not reduce the
carrion-only blocker set, do not spend another iteration merely retuning the
pure-Python residual scale; the next slice must either train against carrion
fixture labels directly or move to the torch/IQL or vectorized rollout training
path.

Fifth implementation milestone:

- Neural artifacts now record
  `contextual_fixture_floor_gap_action_bias_v2`. It keeps the previous bounded
  global fixture action delta for compatibility, and adds
  `policy_visible_carrion_water_context_bias_v1`, a small context-gated action
  bias derived only from the ecological policy input vector: energy,
  hydration, matched-diet need, center-patch carcass/fresh-kill, and visible
  water/carrion navigation vectors.
- This is explicitly not fixture identity at runtime. The artifact stores only
  aggregate fixture pressure from the training label report; per-decision
  scoring still sees the ordinary policy-visible observation.

Measured on the same v26 template/artifact inputs:

- Artifact:
  `output/mind/mind-v3-v26-neural-context-bias-artifact-v3.json`
- Drilldown:
  `output/mind/mind-v3-v26-neural-context-bias-carrion-120.json`
- Broad `120`-tick comparison stayed at `19.0` alive / `16.0` births versus
  linear `23.5` / `18.5`.
- Carrion-only was unchanged: `0.0` alive / `2.5` births, five blockers
  (`alive`, `energy`, `hydration`, `health`, `matched_diet`), and primary
  temporal readiness blocker `energy`.
- Neural top-action counts changed only marginally on carrion-only (`eat`
  `160 -> 159`, `move_east` `470 -> 471`), so the context bias is not strong
  enough to create the missing post-carrion survival sequence through the
  current residual/anchor path.

Keep the mechanism as a bounded artifact feature, but treat the measured result
as a stop signal for this family of hand-shaped residual work. The next
productive slice is to collect controlled fixture trajectories for horizon
labels or run a stronger torch/vectorized learner that can train directly on
carrion-only sequences.

Sixth implementation milestone:

- `sim:mind:v3:evaluate --trajectory-output-dir <dir>` now writes each broad
  and controlled-fixture run as a trajectory JSONL.gz file while keeping the
  evaluator in summary-only mode. Reports include `trajectory_path` per run.
- The exported files use the same trajectory contract and loader as
  `sim:trajectory`, so fixture runs can feed `mind_horizon_labels_v1` directly.

Smoke evidence:

- Evaluation:
  `output/mind/mind-v3-v27-fixture-trajectory-export-smoke.json`
- Exported carrion trajectory:
  `output/trajectories/mind-v3-v27-fixture-export-smoke/fixture-carrion-only-mind-v3-29-40.jsonl.gz`
- Horizon labels:
  `output/mind/mind-v3-v27-carrion-fixture-horizon-labels-smoke.json`
- The exported carrion trajectory produced `393` labels at horizons
  `10,20,40`; observed labels were `383/393`, `373/393`, and `353/393`.

This is the first direct bridge from controlled carrion failures to training
labels. The next neural/learner slice should train with these fixture
trajectories included, not only with broad-world v3 trajectories plus aggregate
fixture blocker pressure.

Seventh implementation milestone:

- `sim:mind:v3:train-neural` now accepts repeated
  `--trajectory-weight <positive-float>` arguments, one per trajectory input.
  The artifact records `trajectory_weight_multipliers` plus separate horizon,
  trajectory, and final sample-weight summaries. This lets controlled fixture
  or distillation trajectories be made comparable to broad open-world
  trajectories without adding runtime fixture identity, hidden world state, or
  heuristic delegation.
- First falsification run:
  `output/mind/mind-v3-v28-broad-carrion-neural-artifact.json` trained on the
  two v26 broad v3 trajectories plus two failing carrion-only v3 fixture
  trajectories. The label set increased animal-resource contacts from `34` to
  `126`, but the artifact remained below the linear anchor: `19.0` alive /
  `9.0` births at `80` versus linear `20.5` / `10.0`, and `22.0` /
  `17.0` at `120` versus linear `23.5` / `18.5`. Carrion-only stayed at
  `0.0` alive / `2.5` births with five blockers.
- Source-balanced distillation run:
  `output/mind/mind-v3-v29-source-balanced-neural-artifact.json` trained on
  broad v3 trajectories at weight `1`, controlled carrion heuristic fixture
  trajectories at weight `4`, and failing controlled carrion v3 trajectories
  at weight `1`. The label set recorded `584` animal-resource contacts.
  Broad eval improved to `20.0` alive / `11.5` births at `80` versus linear
  `20.5` / `10.0`, and `23.0` / `19.5` at `120` versus linear `23.5` /
  `18.5`. Dominant action share stayed below `0.47` and heuristic action
  sources stayed zero.
- The hard blocker remains controlled carrion survival. The source-balanced
  artifact still produced `0.0` carrion-only alive / `2.5` births with the
  same five blockers: alive, energy, hydration, health, and matched diet. At
  `120`, neural top actions shifted strongly toward movement on carrion-only,
  but `542` fixture decisions were still shadowed by the linear-margin guard.

This is useful progress because broad reproduction moved without action
collapse, and it is also a clear stop signal for treating source weighting as
the main fix. The next milestone for this track is explicit: keep the broad
`120` result within `1.0` alive of the linear anchor, match or beat linear
births, keep dominant action share at or below `0.50`, and reduce the
carrion-only blocker set below five or produce nonzero terminal carrion-only
alive. If the next bounded anchor/learner slice cannot move the carrion fixture,
switch effort to torch/IQL or vectorized rollout training rather than another
pure scalar-weight pass.

Eighth implementation milestone:

- Bounded neural-control leverage was tested and rejected as a runtime change.
  The first v30 variant let policy-visible carrion/water context relax the
  neural residual scale and linear-margin guard. It was too broad:
  `output/mind/mind-v3-v30-contextual-anchor-leverage-80.json` fell to
  `15.5` alive / `7.5` births versus linear `20.5` / `10.0`, and
  `output/mind/mind-v3-v30-contextual-anchor-leverage-120.json` fell to
  `11.0` / `10.0` versus linear `23.5` / `18.5`. Carrion-only still had
  `0.0` alive, five blockers, and only `3.5` births, so the broad regression
  bought no controlled-fixture progress.
- A tighter direction-aligned variant only relaxed leverage when the neural
  movement top action aligned with policy-visible carrion or water navigation.
  That made the intervention rare in broad worlds (`1.21%` of decisions at
  `80`, `0.86%` at `120`) and recovered most broad performance:
  `output/mind/mind-v3-v30-directional-context-leverage-80.json` produced
  `20.0` alive / `11.5` births, and
  `output/mind/mind-v3-v30-directional-context-leverage-120.json` produced
  `22.5` / `19.0`. The fixture result still did not move: carrion-only stayed
  at `0.0` alive / `2.5` births with the same alive, energy, hydration,
  health, and matched-diet blockers.
- The tested code path was not kept. The existing default remains
  `linear_controller_margin_guarded_neural_residual_v2`; v30 is documented as
  evidence that small anchor/residual leverage changes are exhausted for the
  carrion blocker. The next implementation milestone should either build a
  stronger deterministic autonomous policy artifact from horizon/fixture labels
  or move to torch/IQL/vectorized rollout training. Do not spend another slice
  on scalar trajectory weights, residual scale, or margin guard thresholds
  unless it is part of that stronger learner path and has a predeclared
  carrion-fixture pass/fail threshold.

Ninth implementation milestone:

- v31 adds an opt-in direct deterministic artifact mode:
  `deterministic_horizon_fixture_policy_v2`, trained by
  `sim:mind:v3:train-neural --artifact-mode horizon-fixture`. The artifact
  still uses ecological policy inputs only, immutable pure-Python JSON weights,
  horizon labels, fixture labels, and the same source-balanced v29 trajectory
  multipliers. It bypasses the linear-margin anchor at runtime and records
  `frozen_horizon_fixture_policy_artifact` diagnostics. The default runtime
  remains `linear_controller_margin_guarded_neural_residual_v2`.
- `sim:mind:v3:evaluate` can now include `--anchored-neural-artifact` so one
  report compares the current linear v3 controller, the existing anchored
  neural path, and a primary experimental artifact on the same seeds, founder
  template, and controlled fixture matrix.
- The comparable v31 artifact is
  `output/mind/mind-v3-v31-horizon-fixture-policy-artifact.json`, trained from
  the same six source-balanced v29 trajectories with hidden width `64`. The
  comparable reports are
  `output/mind/mind-v3-v31-horizon-fixture-policy-80.json` and
  `output/mind/mind-v3-v31-horizon-fixture-policy-120.json`, both using
  founder template `output/mind/mind-v3-visible-nav-v26-top8-80-120-search.json`.
- v31 is not accepted. At `80`, the direct policy produced `1.5` alive /
  `0.0` births with dominant action share `0.5155`, versus linear `20.5` /
  `9.0` and anchored neural `21.0` / `10.5`. Carrion-only had nonzero terminal
  alive (`0.5`) and higher births (`5.5`), but still had five blockers. At
  `120`, the direct policy collapsed to `0.0` alive / `0.0` births with
  dominant action share `0.5136`, versus linear `23.0` / `17.5` and anchored
  neural `20.5` / `18.0`. Carrion-only remained `0.0` alive with five blockers,
  although births stayed at `5.5`. Heuristic runtime actions stayed zero.

This closes the bounded pure-Python direct-artifact slice. It proves that
stronger action-value capacity can touch carrion behavior, but the current
horizon-fixture artifact cannot preserve broad ecology. Do not tune this v31
artifact with another local score-weight loop. The next track should be
torch/IQL or vectorized rollout training, where the learner can optimize
temporal value and action support with stronger function approximation and a
proper held-out broad-plus-fixture gate.

Tenth implementation milestone:

- `sim:mind:v3:carrion-autopsy` now builds a deterministic post-carrion-contact
  failure report from trajectory JSONL.gz files. It groups each agent timeline
  after first carcass/fresh-kill consumption, records concrete action/state
  excerpts, classifies the terminal path, and aggregates death causes,
  terminal bottlenecks, post-contact action counts, drink counts, low-gain eat
  loops, movement counts, and post-contact reproduction events. This is a
  diagnostic surface only; it does not change policy, reward, replay, fixture,
  or trainer behavior.
- A fresh v31 carrion-only trajectory export is
  `output/mind/mind-v3-v31-carrion-autopsy-eval-120.json`, with trajectories
  under `output/trajectories/mind-v3-v31-carrion-autopsy/`. The direct v31
  autopsy report is
  `output/mind/mind-v3-v31-carrion-autopsy-direct-120.json`; anchored neural
  and linear comparison reports are
  `output/mind/mind-v3-v31-carrion-autopsy-anchored-120.json` and
  `output/mind/mind-v3-v31-carrion-autopsy-linear-120.json`.
- The concrete failure split is sharper than the aggregate fixture blockers.
  Direct v31 had seven post-contact episodes, all dead within the 120-tick
  window, zero drinks, seven post-contact reproduction events, and dominant
  path `low_gain_eat_energy_depletion_after_carrion_contact` (`3/7`). Its
  post-contact action counts were heavily `eat`-dominated (`337` eats, `10`
  moves), but animal-resource gain totaled only `2.2928`; the first retained
  example consumed fresh kill at tick `2`, reproduced, then repeatedly ate
  low-gain plant food until energy depletion at tick `47`.
- Anchored neural and linear did not share that exact failure. Anchored neural
  had seven post-contact episodes, all energy-depletion deaths, zero drinks,
  and dominant path `movement_energy_depletion_after_carrion_contact` (`7/7`).
  Linear had ten post-contact episodes with the same dominant path (`10/10`).
  This means the carrion fixture is exposing at least two failure modes:
  direct v31 over-selects low-gain eating after contact, while the anchor
  family continues movement after contact until energy death. Neither policy
  sequence learns the needed "eat enough, stop spending energy, hydrate, then
  survive" loop.

Adjusted next tasks after the autopsy:

1. The counterfactual rollout labeler is now the active diagnostic slice. It
   should replay bounded policy-visible scripts such as eat-until-sated,
   move-to-water, drink, stay/rest, and mixed carrion-water recovery against
   carrion fixture seeds. Acceptance: prove at least one legal sequence
   produces 120-tick carrion-only survival, or document that fixture mechanics
   make survival unreachable.
2. If a survivable sequence exists, feed those counterfactual labels into the
   existing torch/IQL path with explicit energy, hydration, health,
   matched-diet, and terminal-alive constraints. Do not route them through the
   v31 fixed-projection artifact.
3. Keep the same hard gate for any learner artifact: broad 120 alive within
   about `1.0` of linear, broad 120 births not worse than linear, dominant
   action share `<= 0.50`, zero heuristic runtime actions, and carrion-only
   blocker count reduced or terminal alive nonzero.
4. If no bounded counterfactual plan can keep agents alive, stop learner work
   and audit fixture/world mechanics directly. The blocker would then be the
   environment/curriculum, not policy capacity.
5. If counterfactual survival exists but torch/IQL cannot learn it, move to
   vectorized rollout training or a model-based learner rather than another
   scalar weighting, residual-scale, or anchor-margin pass.

Eleventh implementation milestone:

- `sim:mind:v3:carrion-counterfactual` now runs deterministic, policy-visible
  carrion-water recovery scripts against the controlled `carrion_only` fixture
  and writes `mind_v3_carrion_counterfactual_rollout_v1` reports. The command
  can also export trajectories for the scripted plans. This is a diagnostic
  labeler, not a runtime policy promotion path.
- The first v32 diagnostic report is
  `output/mind/mind-v3-carrion-counterfactual-v32-120.json`, with trajectories
  under `output/trajectories/mind-v3-carrion-counterfactual-v32/`. It used
  fixture seeds `29,37`, horizon `120`, and recorded the v31 direct autopsy
  report as source context.
- The answer is clear: carrion-only survival is mechanically possible under
  legal policy-visible action sequences. `hydration_safe_carrion_cycle` kept
  `3` agents alive on both fixture seeds, with `3.0` alive mean, `11.0` births
  mean, zero heuristic action sources, and dominant action share `0.2528`.
  `water_first_recovery` and `conserve_after_carrion` also produced nonzero
  terminal alive on at least one run, while naive `carrion_then_water` still
  died out.
- Autopsying the successful hydration-cycle trajectories produced
  `output/mind/mind-v3-carrion-counterfactual-v32-hydration-cycle-autopsy.json`.
  It found `22` post-contact episodes, `4` survived contact windows,
  post-contact survival rate `0.1818`, `65` drinks, `87` animal-resource
  events, and `21.825` animal-resource gain. The dominant remaining death path
  is still `movement_energy_depletion_after_carrion_contact` (`15/18` deaths),
  so the sequence is not "solved"; it is a positive feasibility label for
  constrained learner training.

Adjusted next tasks after the counterfactual result:

1. The successful counterfactual trajectories now need to become explicit
   action/value labels for constrained torch/IQL. The label contract must
   preserve source script, fixture seed, horizon, terminal alive, energy,
   hydration, health, matched diet, action support, and animal-resource gain.
2. Train an opt-in torch/IQL artifact on broad trajectories plus the
   counterfactual carrion labels with constraints for terminal alive and
   homeostatic state. Do not alter the default linear controller during this
   slice.
3. Evaluate linear default, anchored neural, and the torch/IQL candidate on the
   same broad `120` and carrion fixture matrix. Acceptance stays: broad alive
   within about `1.0` of linear, births not worse than linear, dominant action
   share `<= 0.50`, zero heuristic runtime actions, and carrion-only blocker
   count reduced or terminal alive nonzero.
4. If torch/IQL cannot learn the hydration-cycle behavior despite positive
   labels, move to vectorized rollout/model-based training. Do not return to
   residual-scale, anchor-margin, or scalar trajectory-weight tuning.

Twelfth implementation milestone:

- `sim:mind:v3:carrion-counterfactual-labels` now builds
  `mind_v3_carrion_counterfactual_labels_v1` reports from scripted
  counterfactual trajectory JSONL.gz files. The labels align every trajectory
  decision with policy-visible action support, source script, fixture seed,
  horizon targets, and a rollout-terminal target. This still does not alter
  runtime policy or trainer behavior.
- The first v33 label report is
  `output/mind/mind-v3-carrion-counterfactual-v33-hydration-cycle-labels.json`,
  generated from the two successful `hydration_safe_carrion_cycle` trajectories
  with horizons `20,40,80,120` and primary horizon `120`.
- The report contains `1155` labels, all with legal logged actions. Logged
  action counts are `stay=322`, `move_north=200`, `eat=186`, `drink=141`,
  `move_east=126`, `move_south=90`, and `move_west=90`. The 120-tick horizon
  itself is censored for final survivors because the last decision tick is
  `119`, so primary observed terminal-alive rate is `0.0`; the added
  rollout-terminal target preserves the real end-of-run result: `46` terminal
  agent timelines, `6` terminal-alive agents, alive-agent rate `0.130435`, and
  `124` labels whose owning agent survives to the rollout terminal state.
- The strongest positive labels now expose the desired sequence explicitly. One
  example at tick `64` requests `drink`, has legal action support, and reaches
  terminal alive at tick `119` with energy `0.4531`, hydration `0.965`, health
  `0.9803`, matched diet `1.0`, `1.8204` animal-resource gain to terminal, and
  rollout action-value score `0.89062`.

Adjusted next tasks after the label result:

1. The counterfactual label report is now wired into an opt-in torch/IQL data
   path. Keep the original trajectories as the transition source, but use the
   labels for row weights, terminal/homeostatic constraints, and action-support
   diagnostics. Do not change default Mind v3 runtime selection.
2. Train one bounded torch/IQL candidate using broad trajectories plus these
   hydration-cycle labels. Acceptance remains broad 120 within about `1.0`
   alive of linear, births not worse than linear, dominant action share
   `<= 0.50`, zero heuristic runtime actions, and carrion-only blocker count
   reduced or terminal alive nonzero.
3. If the labeled torch/IQL slice cannot move carrion, stop this path and move
   to vectorized rollout/model-based training rather than adding another
   weight/anchor/search tweak.

Thirteenth implementation milestone:

- `sim:mind:train` and `sim:mind:gate` now accept
  `--torch-iql-counterfactual-labels` plus
  `--torch-iql-counterfactual-label-weight-scale` for `torch-discrete-iql`.
  This is opt-in only; no default Mind v3 runtime, trainer, or promotion path
  changes.
- Training records now carry in-memory source path and dataset-record index
  context, and the counterfactual labels align primarily by trajectory path and
  record index. This avoids a silent miss when broad trajectories are prepended
  to the counterfactual trajectories during mixed training.
- The torch/IQL hook uses the label report for three signals: row weights,
  auxiliary logged-action value targets from the rollout-terminal score, and
  auxiliary terminal homeostatic viability targets for both the state viability
  head and logged-action viability head. The artifact training metrics record
  matched-label count, action support failures, terminal-alive label count,
  animal-resource gain, source scripts, action counts, and label digest.
- This slice still has not produced a trained torch candidate in the local CPU
  environment. The next acceptance-bearing run is on a torch-capable machine:
  train one candidate on broad trajectories plus
  `output/mind/mind-v3-carrion-counterfactual-v33-hydration-cycle-labels.json`,
  then evaluate broad `120` and carrion-only fixture gates beside linear and
  anchored neural.

Fourteenth implementation milestone:

- `sim:mind:v3:labeled-iql-slice` now provides the missing acceptance report
  for the next torch run. It evaluates three policy surfaces under the same
  broad and controlled-fixture horizons: the current Mind v3 linear default,
  an optional anchored Mind v3 neural artifact, and a learned torch/IQL
  candidate artifact loaded through the heuristic-free `autonomous` runtime.
- The report writes a single acceptance gate:
  broad `120` alive regression versus linear must be `<= 1.0`, broad births
  must match or beat linear, candidate dominant requested-action share must be
  `<= 0.50`, candidate heuristic runtime actions must be `0`, and the
  carrion-only fixture must either produce nonzero terminal alive agents or
  reduce blocker count versus the linear baseline.
- This is still not a promotion or result claim. The local machine used for
  this slice does not have torch installed, so the acceptance-bearing run
  remains an RTX-host task. The value of this slice is that the next candidate
  cannot hide behind a broad-only policy eval or a post-hoc fixture check.

Concrete RTX sequence for v34:

```bash
npm run sim:mind:gate -- \
  --reuse-trajectories \
  --trainer torch-discrete-iql \
  --torch-device cuda \
  --extra-trajectory output/trajectories/mind-v3-carrion-counterfactual-v32/counterfactual-hydration_safe_carrion_cycle-seed-29.jsonl.gz \
  --extra-trajectory output/trajectories/mind-v3-carrion-counterfactual-v32/counterfactual-hydration_safe_carrion_cycle-seed-37.jsonl.gz \
  --torch-iql-counterfactual-labels output/mind/mind-v3-carrion-counterfactual-v33-hydration-cycle-labels.json \
  --artifact-output output/mind/mind-v3-v34-labeled-iql-artifact.json \
  --output output/mind/mind-v3-v34-labeled-iql-train-gate.json \
  --artifact-diagnostics-workers 8 \
  --evaluation-workers 2

npm run sim:mind:v3:labeled-iql-slice -- \
  --candidate-artifact output/mind/mind-v3-v34-labeled-iql-artifact.json \
  --enable-mind \
  --mind-runtime-mode autonomous \
  --founder-template output/mind/mind-v3-need-gated-v23-top8-80-120-search.json \
  --anchored-neural-artifact output/mind/mind-v3-v29-source-balanced-neural-artifact.json \
  --seeds 5,13,19,29,37 \
  --ticks 120 \
  --fixture-names carrion_only \
  --fixture-seeds 29,37 \
  --fixture-ticks 120 \
  --output output/mind/mind-v3-v34-labeled-iql-slice-report.json \
  --experiment-ledger-output output/mind/mind-v3-experiment-ledger.jsonl
```

Use the same founder template as the comparison lineage when making a strict
apples-to-apples report. Omit `--founder-template` only when the goal is to
evaluate against the current compiled linear default. Omit
`--anchored-neural-artifact` only when the v29 anchored reference is not part of
the question.

RTX v34 result:

- The CUDA train-gate completed on `gpu4070` and wrote
  `output/mind/mind-v3-v34-labeled-iql-artifact.json` plus
  `output/mind/mind-v3-v34-labeled-iql-train-gate.json`. The strict control
  gate passed but remains not promoted: hard guard `0.0998`, heuristic delegate
  `0.3716`, total fallback `0.4714`, safe deviation `0.0`, alive delta `0.0`,
  births delta `0.0`.
- The acceptance slice wrote
  `output/mind/mind-v3-v34-labeled-iql-slice-report.json` and failed. Broad
  `120` did not regress alive agents and improved births versus linear
  (`+3.6`), with zero heuristic runtime actions, but requested actions were
  over-concentrated (`0.6867` dominant share versus the `0.50` cap) and the
  carrion-only fixture still did not move (`0.0` terminal alive, no blocker
  reduction).
- This closes the labeled torch/IQL counterfactual-label path as a controller
  candidate for this milestone. The next productive slice should follow the
  existing boundary above: vectorized rollout/model-based training or a
  different credit/data path, not another local anchor or scalar-weight tweak.

RTX v35 rollout-search scout:

- A small RTX scout used the existing autonomous evolution CLI with process
  rollout workers, basic controlled fixtures, and the `80,120` fixture rerank:
  `sim:mind:v3:evolve --seeds 5,13,19 --holdout-seeds 29,37
  --curriculum-ticks 80,120 --population-size 8 --generations 2
  --rollout-workers 8 --fixture-suite basic --fixture-selection-top-k 4
  --fixture-rerank-top-k 4 --fixture-rerank-ticks 80,120`.
- The run completed on `gpu4070` and wrote
  `output/mind/mind-v3-v35-rollout-search-smoke.json`. The selected candidate
  was `g1-c1`: score `105.6799`, train-seed alive mean `15.6667`, births mean
  `7.0`, zero heuristic action sources, dominant requested action `eat` at
  `0.4444`, and `8.0` unique requested actions on average. Held-out seeds
  `29,37` at `80` ticks stayed under the action-collapse cap (`eat` at
  `0.4071`), with alive mean `8.5`, births mean `4.0`, and zero heuristic
  action sources.
- The fixture gate still failed. Every reranked candidate hit the
  `carrion_only` alive floor at `120` ticks. The selected candidate passed the
  `80`-tick fixture horizon, but `carrion_only@120` had `0.0` alive mean
  despite `5.0` births mean, so the curriculum stopped at
  `fixture_gate_floor`.
- The v35 lesson is that broad heuristic-free survival and births are no
  longer the only blocker; the hard target is now the `80 -> 120` carrion-only
  survival transition. Next work should keep `carrion_only@120` as the smallest
  acceptance target and move to policy capacity plus temporal-credit data,
  vectorized rollout training, or a model-based learner rather than another
  scalar-weight search.

v36 temporal-credit audit:

- The first v36 diagnostic regenerated `carrion_only@120` trajectories for the
  v35 selected candidate with `--trajectory-output-dir`, then built horizon
  labels, fixture labels, and a carrion autopsy. The fixture reproduction signal
  was not enough: candidate `g1-c1` still had `0.0` fixture alive mean at `120`
  despite `5.0` births mean, and autopsy found `12/12` post-contact episodes
  died after carrion contact, dominantly by
  `movement_energy_depletion_after_carrion_contact`.
- A small direct `horizon-fixture` neural artifact trained on the v35
  carrion-failure trajectories plus the earlier hydration-cycle counterfactual
  trajectories did not recover the target. It regressed open held-out seeds
  from `6.5` alive and `6.5` births to `0.0/0.0`, and regressed
  `carrion_only@120` births from `5.0` to `1.0`, with zero heuristic runtime
  actions.
- `sim:mind:v3:temporal-credit-audit` now gates this failure mode before
  training. On `output/mind/mind-v3-v36-temporal-credit-horizon-labels.json`,
  the audit wrote `output/mind/mind-v3-v36-temporal-credit-audit.json` and
  failed readiness: `120`-tick survivor count `0`, `120` post-contact survivor
  count `0`, while the `80` horizon still had `11` survivors. This means the
  current data is not merely weak; it lacks positive `120`-horizon survival
  support for the target transition.
- The next implementation target is to create positive `carrion_only@120`
  support before training again. Do not run another learner on this label set
  unless the temporal-credit audit passes. The useful directions are exact or
  approximate state-branching rollout data, a vectorized rollout/search loop
  that can discover terminal carrion survivors, or a model-based learner that
  can optimize the `80 -> 120` survival transition explicitly.

v37 branch-and-explore working slice:

- `sim:mind:v3:carrion-branch-explore` adds the first exact simulator branch
  surface for the carrion blocker. It replays the controlled `carrion_only`
  fixture under a policy-visible base script until an animal-resource contact
  record, deep-copies the exact post-contact simulator state in-process, fans
  out continuation scripts from that branch, and can write per-branch trajectory
  JSONL.gz files.
- This is not yet the full quality-diverse archive. It is the smallest
  reproducible v37 example: branch state digests, continuation run summaries,
  zero-heuristic action checks, per-seed positive terminal survivor coverage,
  and deterministic replay verification from each copied branch state in one
  report.
- Local working example:
  `npm run sim:mind:v3:carrion-branch-explore -- --seeds 29,37 --ticks 120
  --trajectory-output-dir output/trajectories/mind-v3-v37-branch-explore-smoke
  --output output/mind/mind-v3-v37-branch-explore-smoke.json`.

v38 recovery-archive working slice:

- `sim:mind:v3:carrion-recovery-archive` builds the first quality-diverse
  recovery archive on top of v37 branch reports. It can consume an existing
  branch report or generate one, descriptors each branch continuation by
  branch-tick band, contact energy/hydration bins, resource gain, terminal
  alive bin, births bin, continuation script, and dominant requested action,
  then keeps one quality elite per descriptor cell.
- The archive also exports a balanced JSONL dataset of survivor and failure
  elites for the future distillation step. This is not a learned policy yet;
  it is the missing data-support surface: compact labeled recovery examples
  with source trajectory paths, terminal outcome labels, descriptor metadata,
  quality scores, and zero-heuristic checks.
- Working example from the v37 smoke branch report:
  `npm run sim:mind:v3:carrion-recovery-archive -- --branch-report
  output/mind/mind-v3-v37-branch-explore-smoke.json --dataset-output
  output/mind/mind-v3-v38-recovery-archive-dataset.jsonl --output
  output/mind/mind-v3-v38-recovery-archive-smoke.json`. The smoke result has
  `5` descriptor cells: `3` survivor cells, `2` failure cells, `5` dataset
  records, and archive acceptance passed.

v39 outcome-telemetry working slice:

- Branch, archive, and broad Mind v3 evaluator reports now include
  `outcome_metrics`, a compact run/aggregate block that makes each rollout
  auditable without reading trajectory JSONL by hand.
- The metrics expose terminal survivor/extinction counts, total births and
  deaths, births by parent/child meat mode, terminal alive agents by meat mode,
  carcass and fresh-kill consumption counts, animal-resource energy gained,
  scavenger-specific animal-resource consumption, and the number of runs with
  reproduction or scavenging.
- The carrion branch and recovery CLIs print those counters directly. The
  recovery dataset records also carry the elite `outcome_metrics`, so the next
  distillation step can filter or weight examples by actual ecological outcome,
  not only by alive/birth labels.
- Local working example:
  `npm run sim:mind:v3:carrion-branch-explore -- --seeds 29,37 --ticks 120
  --trajectory-output-dir output/trajectories/mind-v3-v39-outcome-branch
  --output output/mind/mind-v3-v39-outcome-branch-explore.json`, followed by
  `npm run sim:mind:v3:carrion-recovery-archive -- --branch-report
  output/mind/mind-v3-v39-outcome-branch-explore.json --dataset-output
  output/mind/mind-v3-v39-outcome-recovery-dataset.jsonl --output
  output/mind/mind-v3-v39-outcome-recovery-archive.json`.
- The branch example produced `8` branch runs: `5` terminal-survivor runs,
  `3` extinct runs, `9` terminal alive agents total, `64` births, `151`
  deaths, `256` carcass consumption events, and `210` scavenger carcass
  consumption events, with zero fresh-kill consumption and replay acceptance
  passed.
- The archived elite dataset has `5` records: `3` survivor elites and `2`
  failure elites. Across those elites it records `5` terminal alive agents,
  `40` births, `95` deaths, `150` carcass consumption events, and `129`
  scavenger carcass consumption events. The best survivor is still
  `hydration_safe_carrion_cycle` on seed `29`: `3` terminal survivors, `12`
  births, `39` scavenger carcass events, and `9.9947` scavenger carcass energy
  gained.

v40 recovery-distillation working slice:

- `sim:mind:v3:carrion-recovery-distill` is the first reproducible bridge from
  the recovery archive into a runnable Mind v3 neural artifact. It loads the
  archived elite trajectory paths, builds horizon labels, applies an
  outcome-weighted trajectory policy, trains a deterministic artifact, and
  evaluates that artifact beside the Mind v3 linear baseline on both broad
  open-world seeds and the `carrion_only` fixture.
- The default artifact mode is `anchored-neural`, not direct
  `horizon-fixture`, because the direct archive-distilled controller collapses
  in open evaluation. This slice is a working data/training/evaluation path,
  not a promotion candidate.
- Local working example:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v39-outcome-recovery-archive.json --horizon-output
  output/mind/mind-v3-v40-recovery-distill-horizon-labels.json
  --artifact-output output/mind/mind-v3-v40-recovery-distill-artifact.json
  --evaluation-output output/mind/mind-v3-v40-recovery-distill-evaluation.json
  --output output/mind/mind-v3-v40-recovery-distill-report.json`.
- The run selected all `5` archived elite trajectories and trained on `2382`
  records. Horizon labeling covered `2429` records; the 20-tick labels had
  `2382` observed records, `0.292611` survival rate, and `0.266163`
  reproduction rate, while 80/120-tick survival remained `0.0` and 120-tick
  reproduction was `0.332481`.
- Data-path acceptance passed: the artifact trained, evaluation completed, and
  the candidate used `0` heuristic runtime actions. Promotion failed because
  broad open-world performance regressed versus the linear baseline: candidate
  `9.0` alive mean and `5.5` births mean versus linear `15.0` alive mean and
  `12.5` births mean across seeds `29,37`.
- The carrion-only fixture shows the useful local signal to preserve next:
  candidate and linear both ended extinct across `2` fixture runs, but the
  candidate produced `7` births versus linear `5`, `19` scavenger carcass
  events versus linear `12`, and `5` scavenger parent/child births versus
  linear `4`.

v41 residual-safe recovery-distillation slice:

- Recovery-distilled artifacts now carry artifact-scoped anchored-neural
  residual controls. Existing generic anchored-neural artifacts keep the global
  `0.05` residual scale and `0.015` linear-override margin; the recovery
  distillation path defaults to a safer `0.03` scale and `0.008` override
  margin.
- Local working example:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v39-outcome-recovery-archive.json --horizon-output
  output/mind/mind-v3-v41-residual-safe-recovery-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v41-residual-safe-recovery-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v41-residual-safe-recovery-distill-evaluation.json
  --output
  output/mind/mind-v3-v41-residual-safe-recovery-distill-report.json`.
- This removes the v40 broad-regression blocker on the same seeds and archive:
  local data-path acceptance passed, local promotion-candidate acceptance
  passed, and the candidate used `0` heuristic runtime actions. Open-world
  candidate performance was `19.5` alive mean and `17.5` births mean versus
  linear `15.0` alive mean and `12.5` births mean, a `+4.5` alive and `+5.0`
  births delta across seeds `29,37`.
- The carrion-only fixture still ends extinct across both candidate and linear
  runs, so the full carrion blocker is not solved. The safer residual keeps a
  narrower positive scavenging signal: candidate and linear both produced `5`
  births, while the candidate produced `23` animal-resource events versus
  linear `22`, `19` carcass events versus linear `18`, and `16` scavenger
  carcass events versus linear `12`.
- The next target is larger recovery-archive coverage and held-out seed
  validation, not simply increasing residual strength. The v40 setting had
  stronger carrion fixture births but broke broad survival; v41 shows the
  acceptance surface can catch that tradeoff.

v42 held-out recovery-distillation validation slice:

- Command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v39-outcome-recovery-archive.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds 29,37,41
  --eval-ticks 120 --fixture-ticks 120 --horizon-output
  output/mind/mind-v3-v42-heldout-recovery-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v42-heldout-recovery-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v42-heldout-recovery-distill-evaluation.json --output
  output/mind/mind-v3-v42-heldout-recovery-distill-report.json`.
- Outputs:
  `output/mind/mind-v3-v42-heldout-recovery-distill-horizon-labels.json`,
  `output/mind/mind-v3-v42-heldout-recovery-distill-artifact.json`,
  `output/mind/mind-v3-v42-heldout-recovery-distill-evaluation.json`, and
  `output/mind/mind-v3-v42-heldout-recovery-distill-report.json`.
- Data-path acceptance passed and promotion-candidate acceptance passed under
  the current broad-plus-fixture criteria. The candidate used `0` heuristic
  runtime actions.
- The training data did not change: `5` selected recovery trajectories,
  `2382` trained records, `3` survivor records, `2` failure records, and
  survivor cells only from seed `29`.
- Broad open-world held-out result across seeds `5,13,19,29,37,41` at `120`
  ticks: candidate `17.8333` alive mean, `14.8333` births mean, and `17.0`
  deaths mean versus linear `13.8333` alive mean, `12.3333` births mean, and
  `18.5` deaths mean. The candidate beat or matched linear on alive and births
  for every tested seed. The weakest broad seed was `41`, where candidate and
  linear tied at `8` alive and `8` births.
- Carrion-only fixture result across seeds `29,37,41` at `120` ticks:
  candidate and linear both ended extinct on all `3` runs. Candidate births
  mean was `2.6667` versus linear `2.0`; candidate scavenger carcass events
  totaled `21` versus linear `16`; candidate animal-resource events totaled
  `28` versus linear `26`.
- The first fixture break is `carrion_only@29`: candidate and linear both ended
  at `0` alive, candidate matched linear on births (`3` each), matched
  scavenger carcass events (`9` each), and had fewer total animal-resource
  events (`12` versus `15`). This is the clearest signal that the current
  recovery archive is under-covered rather than under-tuned.
- v42 confirms the v41 conclusion on a broader matrix: residual-safe anchoring
  protects broad open-world behavior, but it does not produce held-out
  carrion-only terminal survivors. The next slice should be archive expansion,
  not residual tuning or RTX torch/IQL on the current five-trajectory archive.

v43 recovery-archive expansion slice:

- Command:
  `npm run sim:mind:v3:carrion-recovery-archive -- --seeds 29,37,41
  --ticks 120 --max-branch-points-per-seed 3 --trajectory-output-dir
  output/trajectories/mind-v3-v43-archive-expansion-branch
  --max-dataset-records-per-class 12 --min-survivor-cells 2
  --min-failure-cells 1 --dataset-output
  output/mind/mind-v3-v43-archive-expansion-dataset.jsonl --output
  output/mind/mind-v3-v43-archive-expansion-report.json`.
- Outputs:
  `output/mind/mind-v3-v43-archive-expansion-report.json`,
  `output/mind/mind-v3-v43-archive-expansion-dataset.jsonl`, and branch
  trajectories under
  `output/trajectories/mind-v3-v43-archive-expansion-branch/`.
- Archive acceptance passed with replay verification. The archive contains
  `23` cells, `20` survivor cells, `3` failure cells, and `15` dataset records
  (`12` survivor, `3` failure).
- This expands the v39 recovery archive from `8` source branch runs to `36`,
  from `5` cells to `23`, and from survivor support on seed `29` only to
  survivor cells on seeds `29`, `37`, and `41`.
- Source branch outcomes: `32/36` runs ended with terminal survivors, `4/36`
  ended extinct, all `36` runs produced births and animal-resource
  consumption, and all animal-resource use was carcass use (`1345` carcass
  events, `0` fresh-kill events). Source runs produced `65` terminal alive
  agents, `327` births, and `1114` scavenger carcass events.
- Archived elite outcomes: `20/23` elites ended with terminal survivors,
  `3/23` ended extinct, with `39` terminal alive agents, `196` births, and
  `666` scavenger carcass events. Terminal survivors were all scavengers.
- Best survivor elite: seed `29`, branch
  `carrion-only-seed-29-branch-2-tick-2-agent-12`,
  `conserve_after_carrion`, `3` alive agents, `14` births, `36` scavenger
  carcass events, `0` heuristic runtime actions.
- Best failure elite: seed `37`, branch
  `carrion-only-seed-37-branch-0-tick-0-agent-9`, `carrion_then_water`,
  `0` alive agents, `13` births, `45` scavenger carcass events, `0`
  heuristic runtime actions. This is now a useful contrast case rather than
  the only seed-specific signal.
- v43 satisfies the archive-expansion prerequisite for re-distillation: positive
  `carrion_only@120` survivor support now exists from more than one seed. The
  next step is to distill from this expanded archive under the same
  residual-safe defaults and broad-plus-carrion acceptance matrix.

v43 expanded-archive recovery-distillation slice:

- Command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v43-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds 29,37,41
  --eval-ticks 120 --fixture-ticks 120 --horizon-output
  output/mind/mind-v3-v43-expanded-archive-recovery-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v43-expanded-archive-recovery-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v43-expanded-archive-recovery-distill-evaluation.json
  --output
  output/mind/mind-v3-v43-expanded-archive-recovery-distill-report.json`.
- Outputs:
  `output/mind/mind-v3-v43-expanded-archive-recovery-distill-horizon-labels.json`,
  `output/mind/mind-v3-v43-expanded-archive-recovery-distill-artifact.json`,
  `output/mind/mind-v3-v43-expanded-archive-recovery-distill-evaluation.json`,
  and
  `output/mind/mind-v3-v43-expanded-archive-recovery-distill-report.json`.
- Data-path acceptance passed, but promotion-candidate acceptance failed with
  blockers `open_alive_regression_vs_linear` and
  `open_birth_regression_vs_linear`. The candidate still used `0` heuristic
  runtime actions.
- Training used `15` selected recovery trajectories and `7796` trained records,
  versus v42's `5` selected trajectories and `2382` trained records.
- Broad open-world result across seeds `5,13,19,29,37,41` at `120` ticks:
  candidate `12.3333` alive mean, `10.6667` births mean, and `18.3333`
  deaths mean versus linear `13.8333` alive mean, `12.3333` births mean, and
  `18.5` deaths mean. The candidate regressed by `-1.5` alive mean and
  `-1.6666` births mean versus linear. It also regressed versus the v42
  candidate by `-5.5` alive mean and `-4.1666` births mean.
- Broad seed deltas versus linear: seed `5` `+2` alive / `+2` births, seed
  `13` `+7` alive / `+6` births, seed `19` `-6` alive / `-7` births, seed
  `29` `-3` alive / `-2` births, seed `37` `-8` alive / `-8` births, seed
  `41` `-1` alive / `-1` births. The first broad break is seed `19`; the
  worst broad break is seed `37`.
- Carrion-only fixture result across seeds `29,37,41` at `120` ticks:
  candidate and linear both ended extinct on all `3` runs. Candidate births
  mean matched linear at `2.0`, candidate scavenger carcass events totaled
  `18` versus linear `16`, and candidate animal-resource events totaled `25`
  versus linear `26`.
- Carrion-only seed details: seed `29` matched linear on alive and births but
  had fewer animal-resource events (`12` versus `15`); seed `37` matched alive
  and births and improved scavenger carcass events (`5` versus `3`); seed `41`
  matched linear on alive, births, and scavenger carcass events. The fixture
  first break remains terminal extinction, not lack of branch survivor support.
- Anchor diagnostics show the failure is not a large runtime takeover. On broad
  open seeds, only `401/11192` candidate decisions changed the linear anchor
  (`3.58%`). Those changes shifted action mix toward `drink` (`+96`) and
  `stay` (`+58`) while reducing movement (`-158` combined cardinal moves),
  enough to hurt broad survival and births on seeds `19`, `29`, `37`, and
  `41`.
- Interpretation: archive expansion succeeded, but direct residual
  distillation from the expanded carrion recovery archive is now too
  fixture-shaped for broad open-world acceptance. The next slice should add
  training/evaluation separation or context-gated residual selection, not
  increase residual strength and not train RTX torch/IQL blindly on the same
  mixed archive.

v44 context-gated expanded-archive recovery-distillation slice:

- Code change:
  recovery-distilled anchored-neural artifacts can now carry an artifact-scoped
  `neural_residual_context_gate`. The default recovery-distill gate is
  `visible_carrion_scavenger_v1`, which applies the residual only for
  policy-visible scavenger/carrion contexts. Older artifacts without the field
  keep the previous unrestricted anchored-residual behavior.
- Command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v43-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds 29,37,41
  --eval-ticks 120 --fixture-ticks 120 --neural-residual-context-gate
  visible_carrion_scavenger_v1 --horizon-output
  output/mind/mind-v3-v44-context-gated-recovery-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v44-context-gated-recovery-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v44-context-gated-recovery-distill-evaluation.json
  --output
  output/mind/mind-v3-v44-context-gated-recovery-distill-report.json`.
- Outputs:
  `output/mind/mind-v3-v44-context-gated-recovery-distill-horizon-labels.json`,
  `output/mind/mind-v3-v44-context-gated-recovery-distill-artifact.json`,
  `output/mind/mind-v3-v44-context-gated-recovery-distill-evaluation.json`,
  and `output/mind/mind-v3-v44-context-gated-recovery-distill-report.json`.
- Data-path acceptance passed and promotion-candidate acceptance passed under
  the current broad-plus-fixture criteria. The candidate used `0` heuristic
  runtime actions.
- Training data stayed the same as v43 expanded-distill: `15` selected recovery
  trajectories and `7796` trained records from an archive with `12` survivor
  records, `3` failure records, and survivor cells on seeds `29`, `37`, and
  `41`.
- Broad open-world result across seeds `5,13,19,29,37,41` at `120` ticks:
  candidate `14.3333` alive mean, `12.6667` births mean, and `18.3333`
  deaths mean versus linear `13.8333` alive mean, `12.3333` births mean, and
  `18.5` deaths mean. The candidate is `+0.5` alive mean and `+0.3334` births
  mean versus linear.
- Broad seed deltas versus linear: seed `5` `0` alive / `0` births, seed `13`
  `0` alive / `-1` births, seed `19` `0` alive / `0` births, seed `29` `0`
  alive / `0` births, seed `37` `+3` alive / `+3` births, seed `41` `0`
  alive / `0` births. The current aggregate gate passes, but seed `13` is a
  residual birth regression that should remain visible in the next stricter
  seed-level acceptance check.
- Carrion-only fixture result across seeds `29,37,41` at `120` ticks:
  candidate and linear both ended extinct on all `3` runs. Candidate births
  mean matched linear at `2.0`; candidate animal-resource events totaled `35`
  versus linear `26`; candidate scavenger carcass events totaled `28` versus
  linear `16`.
- Carrion-only seed details: seed `29` matched linear on alive and births and
  improved scavenger carcass events (`10` versus `9`) while reducing total
  animal-resource events (`13` versus `15`); seed `37` matched alive and births
  while improving animal-resource events (`18` versus `7`) and scavenger
  carcass events (`14` versus `3`); seed `41` matched linear exactly on alive,
  births, and scavenger carcass events.
- Anchor diagnostics show the gate did what it was meant to do. In broad open
  seeds, changed-linear decisions fell from v43's `401/11192` (`3.58%`) to
  `38/11519` (`0.33%`). The residual was shadowed by the context gate on
  `10582` broad decisions and applied on `157`. In the carrion-only fixture,
  it applied on `8` decisions and lifted scavenger carcass events without
  broad open-world regression.
- Interpretation: v44 repairs the v43 negative-transfer failure by making the
  recovery residual context-selective. It still does not solve terminal
  carrion-only survival, so the next task is stricter acceptance and better
  recovery-state carryover or policy memory, not a larger ungated residual.

v45 strict seed-level recovery-distillation acceptance slice:

- Code change:
  recovery-distill evaluation now records paired open-seed deltas versus the
  linear baseline under `candidate_vs_linear_per_seed`. Promotion acceptance
  now blocks any broad open seed with negative alive or birth delta, even when
  aggregate means are positive. The CLI also prints minimum per-seed alive and
  birth deltas.
- Command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v43-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds 29,37,41
  --eval-ticks 120 --fixture-ticks 120 --neural-residual-context-gate
  visible_carrion_scavenger_v1 --horizon-output
  output/mind/mind-v3-v45-strict-acceptance-recovery-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v45-strict-acceptance-recovery-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v45-strict-acceptance-recovery-distill-evaluation.json
  --output
  output/mind/mind-v3-v45-strict-acceptance-recovery-distill-report.json`.
- Outputs:
  `output/mind/mind-v3-v45-strict-acceptance-recovery-distill-horizon-labels.json`,
  `output/mind/mind-v3-v45-strict-acceptance-recovery-distill-artifact.json`,
  `output/mind/mind-v3-v45-strict-acceptance-recovery-distill-evaluation.json`,
  and `output/mind/mind-v3-v45-strict-acceptance-recovery-distill-report.json`.
- Data-path acceptance passed, but promotion-candidate acceptance failed under
  the stricter seed-level gate. The blocker is
  `open_seed_13_birth_regression_vs_linear`. The candidate still used `0`
  heuristic runtime actions.
- Training stayed unchanged from v44: `15` selected recovery trajectories,
  `7796` trained records, anchored-neural artifact mode, residual scale
  `0.03`, override margin `0.008`, and context gate
  `visible_carrion_scavenger_v1`.
- Broad open-world aggregate result across seeds `5,13,19,29,37,41` at `120`
  ticks remains positive versus linear: candidate `14.3333` alive mean and
  `12.6667` births mean versus linear `13.8333` alive mean and `12.3333`
  births mean. Aggregate deltas are `+0.5` alive and `+0.3334` births.
- Broad seed deltas versus linear: seed `5` `0` alive / `0` births, seed `13`
  `0` alive / `-1` births, seed `19` `0` alive / `0` births, seed `29` `0`
  alive / `0` births, seed `37` `+3` alive / `+3` births, seed `41` `0`
  alive / `0` births. Minimum alive delta is `0`; minimum birth delta is `-1`.
- Carrion-only fixture result remains unchanged from v44 in aggregate:
  candidate and linear both ended extinct on all `3` fixture seeds, births mean
  matched at `2.0`, and candidate scavenger carcass events totaled `28` versus
  linear `16`.
- Carrion-only seed details: seed `29` candidate `0` alive / `3` births with
  `10` scavenger carcass events versus linear `0` / `3` and `9`; seed `37`
  candidate `0` / `2` with `14` scavenger carcass events versus linear `0` /
  `2` and `3`; seed `41` matched linear at `0` / `1` and `4` scavenger
  carcass events.
- Interpretation: v45 is a gate-hardening success and a candidate-promotion
  failure. The aggregate v44 candidate was useful, but the stricter acceptance
  now correctly prevents a seed-specific birth regression from being hidden by
  seed `37` gains. The next candidate work should not tune residual scale; it
  should either make the gate/support criterion more selective or expand the
  archive so recovery behavior does not borrow survival from a narrow seed
  niche while still failing terminal carrion-only survival.

v46 strict-gate hardening and archive-expansion slice:

- Code changes:
  strict recovery-distill acceptance now requires paired per-seed delta coverage
  for every open evaluation seed; missing, duplicate, or unexpected seed rows
  block promotion. Neural artifacts now reject unknown
  `neural_residual_context_gate` values instead of silently shadowing all
  residuals. A new `water_rescue_carrion_cycle` continuation script was added
  to expand the recovery archive with a water-first rescue behavior for
  critically dehydrated states.
- Seed `13` diagnosis:
  `output/mind/mind-v3-v46-seed13-diagnosis.json` compared the v45 candidate
  against linear with trajectory diagnostics. The candidate changed only `9`
  current-anchor decisions; the first cumulative birth divergence happened at
  tick `53`. The late changed decisions clustered on agent `18`, where the
  candidate stayed around the map edge while the pure linear run reached water.
  This supports adding water-rescue continuation support, but does not explain
  the failure as broad heuristic delegation or a large residual takeover.
- Archive command:
  `npm run sim:mind:v3:carrion-recovery-archive -- --seeds
  13,19,29,37,41,43 --ticks 120 --max-branch-points-per-seed 4
  --trajectory-output-dir
  output/trajectories/mind-v3-v46-strict-archive-expansion-branch
  --max-dataset-records-per-class 16 --min-survivor-cells 4
  --min-failure-cells 1 --dataset-output
  output/mind/mind-v3-v46-strict-archive-expansion-dataset.jsonl --output
  output/mind/mind-v3-v46-strict-archive-expansion-report.json`.
- Archive result:
  acceptance passed with replay verification. The archive has `55` cells,
  `43` survivor cells, `12` failure cells, and `28` exported dataset records
  (`16` survivor, `12` failure). Survivor cells now exist for all six fixture
  seeds `13,19,29,37,41,43`. The source branch acceptance reports `103`
  successful branch runs across all `6` target seeds.
- Distill command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v46-strict-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds
  13,19,29,37,41,43 --eval-ticks 120 --fixture-ticks 120
  --neural-residual-context-gate visible_carrion_scavenger_v1 --horizon-output
  output/mind/mind-v3-v46-strict-archive-recovery-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v46-strict-archive-recovery-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v46-strict-archive-recovery-distill-evaluation.json
  --output
  output/mind/mind-v3-v46-strict-archive-recovery-distill-report.json`.
- Distill result:
  data-path acceptance passed, but promotion failed. The blockers are
  `open_seed_5_alive_regression_vs_linear`,
  `open_seed_5_birth_regression_vs_linear`, and
  `open_seed_13_birth_regression_vs_linear`. Broad aggregate deltas stayed
  positive versus linear (`+0.3334` alive mean, `+0.1667` births mean), but the
  strict per-seed floor caught seed `5` and seed `13` regressions.
- Carrion-only fixture result:
  expanded fixture seeds still ended extinct for both candidate and linear on
  all `6` runs. Candidate births mean was `2.3333` versus linear `2.1667`;
  candidate scavenger carcass events totaled `56` versus linear `45`.
- Interpretation:
  v46 proves the archive can now hold broad multi-seed survivor continuations,
  but the current feed-forward residual distill still cannot convert those
  branch-state survivors into terminal `carrion_only@120` survivors from
  initial fixture state. Do not run RTX/IQL yet; the next slice should add
  recovery-phase/state carryover or branch-state-conditioned policy inputs, then
  repeat the strict seed-level gate.

v47 recovery-phase carryover slice:

- Code changes:
  neural artifacts now support a serialized
  `visible_carrion_or_recovery_phase_v1` residual context gate plus
  `neural_residual_recovery_phase_ticks`. The runtime policy tracks a
  policy-visible post-contact recovery phase after actual carcass or fresh-kill
  consumption and exposes gate reason/remaining-tick diagnostics. The recovery
  distill CLI now accepts and reports the recovery-phase duration.
- Distill command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v46-strict-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds
  13,19,29,37,41,43 --eval-ticks 120 --fixture-ticks 120
  --neural-residual-context-gate visible_carrion_or_recovery_phase_v1
  --neural-residual-recovery-phase-ticks 8 --horizon-output
  output/mind/mind-v3-v47-recovery-phase-distill-horizon-labels.json
  --artifact-output output/mind/mind-v3-v47-recovery-phase-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v47-recovery-phase-distill-evaluation.json --output
  output/mind/mind-v3-v47-recovery-phase-distill-report.json`.
- Distill result:
  data-path acceptance passed, but promotion still failed. The blockers stayed
  `open_seed_5_alive_regression_vs_linear`,
  `open_seed_5_birth_regression_vs_linear`, and
  `open_seed_13_birth_regression_vs_linear`. Broad aggregate deltas remained
  positive versus linear (`+0.3334` alive mean, `+0.1667` births mean), with
  `0` heuristic runtime actions, but the strict per-seed floor remained
  negative (`-1` alive, `-1` births).
- Carrion-only fixture result:
  expanded fixture seeds still ended extinct for both candidate and linear on
  all `6` runs. Candidate births mean was `2.3333` versus linear `2.1667`;
  candidate scavenger carcass events totaled `56` versus linear `45`.
- Diagnostics:
  `output/mind/mind-v3-v47-regression-diagnosis.json` rechecked failing open
  seeds `5` and `13` with the v47 artifact. The two-seed slice remained
  negative versus linear (`-0.5` alive mean, `-1.0` births mean). Runtime
  residual application increased only modestly under the carryover gate, so
  the missing piece is not simply remembering that carrion contact happened.
- Interpretation:
  v47 is a contract/observability improvement and a candidate-promotion
  failure. The next useful slice should make the branch-state condition itself
  visible to scoring, for example a compact branch-state-conditioned
  value/support head or recovery-phase action-value bias derived from survivor
  continuations. Do not start RTX/IQL until that branch-state-conditioned
  surface passes the strict seed-level gate and produces positive
  `carrion_only@120` terminal survivors.

v48 branch-state-conditioned recovery action bias:

- Code changes:
  neural artifacts now accept a serialized `recovery_phase_action_bias` payload
  with policy `branch_survivor_failure_action_log_odds_v1`. The carrion
  distiller derives this payload from survivor-vs-failure action frequencies
  during post-contact recovery windows and the runtime scorer applies it only
  when `neural_residual_recovery_phase_remaining > 0`. Validation now rejects
  unknown recovery-bias policies. During audit, the log-odds centering was
  fixed to ignore unobserved action-contract entries so inactive
  attack/mate/signal actions cannot clamp all observed recovery actions
  positive.
- Distill command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v46-strict-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds
  13,19,29,37,41,43 --eval-ticks 120 --fixture-ticks 120
  --neural-residual-context-gate visible_carrion_or_recovery_phase_v1
  --neural-residual-recovery-phase-ticks 8 --horizon-output
  output/mind/mind-v3-v48-branch-state-bias-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v48-branch-state-bias-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v48-branch-state-bias-distill-evaluation.json --output
  output/mind/mind-v3-v48-branch-state-bias-distill-report.json`.
- Distill result:
  data-path acceptance passed, but strict promotion failed with the same open
  blockers as v47: `open_seed_5_alive_regression_vs_linear`,
  `open_seed_5_birth_regression_vs_linear`, and
  `open_seed_13_birth_regression_vs_linear`. Broad aggregate deltas versus
  linear remained positive (`+0.3334` alive mean, `+0.1667` births mean) with
  `0` heuristic runtime actions, but the minimum seed deltas stayed negative
  (`-1` alive, `-1` births).
- Recovery-bias payload:
  `record_count=3291`, `survivor_total_weight=10013.03`, and
  `failure_total_weight=192.5`. The final action-bias vector was small and
  differentiated: `move_east +0.044308`, `stay +0.039837`,
  `move_west +0.036883`, `drink +0.001513`, `eat -0.005664`,
  `move_north -0.03084`, and `move_south -0.086037`; unobserved actions stayed
  at `0`.
- Carrion-only fixture result:
  all `6` candidate and linear fixture runs still ended extinct. Candidate
  births mean was `2.3333` versus linear `2.1667`; candidate scavenger carcass
  events totaled `56` versus linear `45`.
- Diagnosis:
  `output/mind/mind-v3-v48-regression-diagnosis.json` rechecked seeds `5` and
  `13`. The two-seed slice stayed negative versus linear (`-0.5` alive mean,
  `-1.0` births mean). Seed `5` shifted agent `5`'s first birth from tick `49`
  to tick `52`, and the linear-only offspring birth at tick `83` did not occur.
  Seed `13` was not one isolated lost birth; the candidate shifted the timing
  and positions of several births, with linear-only births at ticks `53`, `79`,
  `85`, `95`, `103`, `108`, `112`, and `115` and candidate-only births at
  ticks `54`, `77`, `84`, `105`, `107`, `107`, and `114`.
- Interpretation:
  v48 proves the artifact/scorer can carry a branch-conditioned action prior,
  but the learned bias is too weak and too indirect to repair strict
  open-seed regressions or terminal carrion-only extinction. The next audit
  should determine whether residual overrides near local animal resources are
  causing delayed reproduction; if not, stop adding small residual nudges and
  expand the exact branch archive toward positive `carrion_only@120` terminal
  survivors.

v49 local-resource eat guard audit:

- Code changes:
  the anchored neural policy now suppresses residual overrides when the linear
  anchor's top action is `eat`, `eat` is legal, and current-tile carcass or
  fresh-kill energy is visible. The guard reports
  `linear_local_animal_resource_eat_guard` as the residual shadow reason.
- Distill command:
  `npm run sim:mind:v3:carrion-recovery-distill -- --archive-report
  output/mind/mind-v3-v46-strict-archive-expansion-report.json --eval-seeds
  5,13,19,29,37,41 --fixture-names carrion_only --fixture-seeds
  13,19,29,37,41,43 --eval-ticks 120 --fixture-ticks 120
  --neural-residual-context-gate visible_carrion_or_recovery_phase_v1
  --neural-residual-recovery-phase-ticks 8 --horizon-output
  output/mind/mind-v3-v49-local-resource-eat-guard-distill-horizon-labels.json
  --artifact-output
  output/mind/mind-v3-v49-local-resource-eat-guard-distill-artifact.json
  --evaluation-output
  output/mind/mind-v3-v49-local-resource-eat-guard-distill-evaluation.json
  --output output/mind/mind-v3-v49-local-resource-eat-guard-distill-report.json`.
- Distill result:
  data-path acceptance passed, but strict promotion still failed with the same
  blockers and same aggregate metrics as v48. The guard fired only once across
  the broad-open run (`linear_local_animal_resource_eat_guard: 1`), while
  residual application stayed at `168` applied and `11289` shadowed decisions.
  Therefore the v48/v49 regressions are not primarily caused by overriding a
  linear `eat` on current-tile animal resource.
- Diagnosis:
  `output/mind/mind-v3-v49-regression-diagnosis.json` repeated the paired
  seed `5` and `13` diagnosis. The two-seed slice remained `-0.5` alive mean
  and `-1.0` births mean versus linear. The same birth-timing pattern remained:
  seed `5` lost one downstream offspring birth after delaying agent `5`, and
  seed `13` shifted multiple branch positions/timings rather than exposing one
  protected local-eat decision.
- Interpretation:
  v49 is a useful negative audit: a narrow local-resource eat guard is
  behaviorally correct and tested, but it does not solve the strict gate. The
  next cycle should return to data generation: expand branch points,
  continuation scripts, and fixture seeds until the archive contains positive
  terminal `carrion_only@120` survivors from multiple seeds. Do not start
  RTX/IQL yet.

v50 audit and shadowed-residual repair:

- Project-wide audit finding:
  post-run trajectory exports from `mind_v3_evaluate --trajectory-output-dir`
  were writing retained `world.trajectory_records` without reattaching
  retained `policy_decision_diagnostics` or per-record `policy_update_trace`.
  That made v48/v49 diagnosis trajectories weaker than the JSON aggregate
  implied. The evaluator now enriches retained records before writing, and the
  world keeps update traces aligned with retained trajectory records.
- Scorer bug:
  `_blend_neural_with_linear_anchor` rounded blended scores to four decimals
  before action selection. With `neural_residual_effective_scale=0`, tiny
  linear-anchor margins could collapse to ties and flip to alphabetical
  actions, so diagnostics could report a shadowed residual while behavior still
  changed. The blend now preserves raw score precision for action selection,
  and tests cover the zero-scale low-margin case.
- Re-run:
  `output/mind/mind-v3-v50-shadowed-residual-fix-report.json` used the same
  v46 archive and strict six-seed matrix after the scorer repair. Data-path
  acceptance passed, seed `13` was repaired, and the only open strict blocker
  left was seed `5` (`-1` alive, `-1` birth versus linear). Aggregate broad
  deltas stayed positive (`+0.3334` alive mean, `+0.3334` births mean).
- Fixture result:
  `carrion_only@120` still had zero terminal survivors on all six fixture
  seeds. This made the next step data support, not another local residual
  guard.

v51 counterfactual-augmented archive slice:

- Counterfactual coverage:
  `output/mind/mind-v3-v51-counterfactual.json` ran all five scripted
  counterfactual policies on fixture seeds `13,19,29,37,41,43`. The
  `hydration_safe_carrion_cycle` script survived all six seeds
  (`2.5` alive mean, `10.5` births mean), and the combined report had `19`
  survivor runs out of `30`. This satisfies the "positive
  `carrion_only@120` survivors from multiple seeds" data-support condition for
  an experimental learner.
- Archive code:
  recovery archives can now append full counterfactual fixture trajectories via
  `--counterfactual-report` and can require a minimum number of distinct
  counterfactual survivor seeds via `--min-counterfactual-survivor-seeds`.
  The v51 archive
  `output/mind/mind-v3-v51-counterfactual-augmented-archive-report.json`
  passed with `55` branch cells plus `30` counterfactual full-fixture records:
  `74` dataset records total, `51` survivor records, `23` failure records, and
  counterfactual survivor coverage on all six target fixture seeds.
- Distill result:
  `output/mind/mind-v3-v51-counterfactual-augmented-distill-report.json`
  trained on `39,128` records from `74` selected trajectories. Data-path
  acceptance passed, but promotion failed on seed `5` (`-1` alive, `-1` birth)
  and a `carrion_only` scavenger-event regression. The fixture still had zero
  terminal survivors.
- Interpretation:
  adding full survivor scripts to the supervised residual dataset is not enough
  for the current feed-forward anchored residual. The residual changed only a
  small number of fixture decisions and did not reproduce the long water/carrion
  cycle.

v52 recovery-window probe:

- Probe:
  `output/mind/mind-v3-v52-recovery32-probe-artifact.json` changed only
  `neural_residual_recovery_phase_ticks` from `8` to `32` on the v51 artifact
  and evaluated it in
  `output/mind/mind-v3-v52-recovery32-probe-evaluation.json`.
- Result:
  broad aggregate deltas stayed positive (`+0.3334` alive mean and births
  mean), but seed `5` still regressed (`-1` alive, `-1` birth), seed `13`
  regressed by `-1` alive, and `carrion_only@120` still had zero terminal
  survivors.
- Boundary:
  this is the practical RTX/IQL boundary, not a promotion boundary. The data now
  contains multiple full-horizon carrion-only survivor trajectories, and the
  deterministic residual path has failed after the scoring bug fix, archive
  augmentation, and longer recovery-memory probe. The next work should be an
  offline value/sequence learner over this expanded archive, with the same
  strict seed-level and fixture gates before any promotion.

v53-v56 RTX/IQL pilot:

- Counterfactual labels:
  `output/mind/mind-v3-v53-counterfactual-labels.json` generated `16,112`
  label records from the `30` full-horizon v51 counterfactual trajectories.
  The primary horizon remained `120`, and the labels were used only by the
  opt-in torch/IQL path.
- v53 base IQL:
  `output/mind/mind-v3-v53-counterfactual-augmented-iql-train-gate.json`
  trained `torch-discrete-iql` on the expanded archive using the RTX trainer.
  The train gate passed but was not promotable (`hard_guard=0.1144`,
  `heuristic_delegate=0.3683`, `total_fallback=0.4827`, no alive/birth
  deltas). The labeled autonomous slice
  `output/mind/mind-v3-v53-counterfactual-augmented-iql-slice-report.json`
  improved broad open outcomes (`14.6667` alive, `15.0` births versus linear
  `13.8333`/`12.3333`) and produced nonzero `carrion_only` terminal survivors
  (`0.6667` alive mean), but failed acceptance because dominant requested
  `eat` share was `0.6644` above the `0.5` cap.
- v54 action-distribution IQL:
  `output/mind/mind-v3-v54-action-distribution-iql-train-gate.json` enabled
  `--torch-iql-action-distribution-regularization`. The train gate again
  passed but was not promotable (`hard_guard=0.096`, `total_fallback=0.4827`).
  The slice improved broad outcomes further (`15.3333` alive, `15.5` births)
  and retained slight carrion movement (`0.1667` alive mean), but still failed
  only on dominant `eat` share (`0.641`).
- v55 contextual-prior IQL:
  `output/mind/mind-v3-v55-contextual-prior-iql-train-gate.json` kept action
  distribution regularization and added contextual behavior-prior
  regularization with a clean split: the `30` full counterfactual trajectories
  stayed in training, while `44` branch trajectories were used as calibration
  records. The train gate passed but was not promotable
  (`hard_guard=0.0979`, `total_fallback=0.4794`). The slice had the strongest
  broad gains (`18.5` alive, `19.3333` births) and nonzero carrion movement
  (`0.1667` alive mean), but failed on dominant `eat` share (`0.6307`).
- v56 blend probe:
  `output/mind/mind-v3-v56-blend97-iql-slice-report.json` changed only the v55
  artifact's `neural_actor_prior_blend_weight` to `0.97`. It did not fix
  action concentration (`eat` share `0.6326`) and removed carrion movement
  (`0.0` alive mean), so higher prior blending is not the right acceptance
  path.
- Interpretation:
  the project is now past the "ready for RTX/IQL" boundary. RTX/IQL learned a
  behaviorally useful signal that the deterministic residual could not:
  nonzero terminal survivors on the `carrion_only@120` fixture plus broad open
  gains without heuristic runtime actions. The blocker has changed from data
  support to learner-side action concentration. The next cycle should improve
  the IQL actor/extraction objective or support constraints to reduce dominant
  `eat` share below `0.5` while preserving positive carrion-only survivors;
  do not relax the acceptance gate.

v57-v63 strict IQL audit, coefficient probes, and rollout calibration:

- Audit fixes before continuing:
  the recovery archive now distinguishes counterfactual survivor rows that have
  trainable trajectory paths from report-only survivor evidence. The
  `counterfactual_survivor_seeds` acceptance input is now path-backed, while
  `counterfactual_report_survivor_seed_count` keeps report coverage visible.
  The labeled IQL slice also inherited the v45 per-seed no-regression rule, so
  broad open promotion is blocked by any held-out seed with a negative alive or
  birth delta versus linear.
- v57 contextual-supported IQL:
  `output/mind/mind-v3-v57-contextual-supported-iql-train-gate.json` passed the
  train gate but was not promotable. The strict recomputed slice
  `output/mind/mind-v3-v57-contextual-supported-iql-slice-report.json` failed
  on dominant `eat` share `0.6444` and seed-level regressions on seeds `13`,
  `37`, and `41` (`min_alive_delta=-16`, `min_births_delta=-8`). Broad means
  were `13.6667` alive and `13.3333` births versus linear `13.8333`/`12.3333`;
  `carrion_only` alive mean was `0.1667`.
- v58 margin-contextual IQL:
  `output/mind/mind-v3-v58-margin-contextual-iql-train-gate.json` passed the
  train gate but was not promotable. The strict recomputed slice
  `output/mind/mind-v3-v58-margin-contextual-iql-slice-report.json` improved
  broad means to `19.8333` alive and `20.0` births, but failed on dominant
  `eat` share `0.6453` and seed `37` alive/birth regression
  (`min_alive_delta=-18`, `min_births_delta=-13`). `carrion_only` alive mean
  remained `0.1667`.
- v59 sharp-action IQL:
  `output/mind/mind-v3-v59-sharp-action-iql-train-gate.json` used the new
  action-distribution loss-weight and temperature overrides
  (`loss_weight=1.25`, `temperature=0.08`). It passed the train gate but was
  not promotable. The strict recomputed slice
  `output/mind/mind-v3-v59-sharp-action-iql-slice-report.json` lowered
  dominant `eat` share only to `0.6315`, improved broad means to `16.5` alive
  and `18.5` births, and moved the fixture to `0.3333` alive mean, but failed
  seeds `19`, `37`, and `41` (`min_alive_delta=-6`,
  `min_births_delta=-3`).
- v60 prior-100 probe:
  `output/mind/mind-v3-v60-prior100-iql-slice-report.json` changed only the
  v59 artifact's `neural_actor_prior_blend_weight` to `1.0`. It made action
  concentration worse (`eat` share `0.6507`) and still failed seed-level
  blockers on seeds `19` and `37` (`min_alive_delta=-4`,
  `min_births_delta=-1`). Higher prior blending is now explicitly ruled out as
  the next path.
- v61 risk-sharp IQL:
  `output/mind/mind-v3-v61-sharp-action-iql-train-gate.json` added
  risk-adjusted actor extraction on top of the v59 sharp action-distribution
  setup. The train gate passed but was not promotable
  (`hard_guard=0.1035`, `heuristic_delegate=0.3754`,
  `total_fallback=0.4789`, strict control target failed). The strict slice
  `output/mind/mind-v3-v61-sharp-action-iql-slice-report.json` failed with
  dominant `eat` share `0.6509`, broad means `13.0` alive and `13.3333`
  births versus linear `13.8333`/`12.3333`, and zero `carrion_only` movement.
  Per-seed blockers were seed `5` alive/birth regression, seed `19` birth
  regression, and seed `37` alive/birth regression
  (`min_alive_delta=-8`, `min_births_delta=-6`).
- v62 rollout-state top-1 actor-bias calibration:
  `output/mind/mind-v3-v62-rollout-calibrated-iql-train-gate.json` added
  opt-in `calibration_bank_top1_actor_bias_control_v1` with a `0.45`
  calibration-bank top-1 cap, `0.08` bias step, and `1.25` max bias delta. The
  full RTX train gate passed but was not promotable
  (`hard_guard=0.1035`, `heuristic_delegate=0.3751`,
  `total_fallback=0.4786`). The calibration report was the key diagnostic:
  branch-calibration actor top-1 was dominated by `stay` (`0.5852` before,
  `0.5824` after) and `move_north` (`0.4148`), not by `eat`; the only saturated
  bias adjustment was `stay=-1.25`. The strict slice
  `output/mind/mind-v3-v62-rollout-calibrated-iql-slice-report.json` therefore
  failed the same acceptance surface: dominant `eat` share `0.6507`, broad means
  `13.0` alive and `13.3333` births versus linear `13.8333`/`12.3333`, zero
  `carrion_only` movement, and seed-level blockers on seed `5` alive/birth,
  seed `19` birth, and seed `37` alive/birth (`min_alive_delta=-8`,
  `min_births_delta=-6`).
- v63 low-prior posthoc probe:
  `output/mind/mind-v3-v63-low-prior-rollout-calibrated-iql-slice-report.json`
  changed only the v62 artifact's `neural_actor_prior_blend_weight` to `0.5`.
  This falsified the low-prior scalar-blend hypothesis: dominant `eat` share
  rose to `0.7718`, broad means fell to `6.0` alive and `6.5` births, the strict
  slice produced `13` blockers, and `carrion_only` still had `0.0` alive mean.
  Together with the v60 prior-`1.0` probe, this rules out scalar prior blending
  as the next useful control axis.
- Interpretation:
  the current IQL family is producing real broad and fixture signal, but it is
  not acceptance-ready. v62 shows that the deployed `eat` concentration is not a
  simple raw-actor top-1 imbalance on the branch calibration bank, and v63 shows
  that lowering contextual-prior blend weight makes the accepted slice worse.
  The exhausted axes are aggregate action regularization, scalar prior blending,
  extraction risk adjustment, and global actor-bias calibration. The next useful
  branch is a rollout-context policy class such as sequence/flow/world-model
  control or Go-Explore-style archive replay/robustification, not another
  coefficient-only IQL variant on the same representation.

v64 rollout-context audit:

- `output/mind/mind-v3-v64-rollout-context-audit.json` added a deterministic
  pre-training audit before opening any runtime or training branch. The audit
  builds per-agent context only from previous trajectory rows: recent
  requested/resolved actions, moved/drank/ate flags, resource gain,
  energy/hydration/health deltas, no-gain eat streak, ticks since drink, ticks
  since animal-resource gain, and derivable post-carrion recovery phase. It
  uses no fixture identity, private world state, future row, heuristic action
  source, or scalar IQL retuning.
- Source data covered the locally available v51 counterfactual archive and
  v51 branch archive (`150` trajectories total). The deterministic split held
  out seeds `37,41,43` and trained on seeds `13,19,29`, producing `41,773`
  train rows and `42,983` held-out rows.
- The ecological+rollout-context lookup improved held-out accuracy
  (`0.714469` versus `0.658702`) but did not materially improve the v62 failure
  mode. True movement/drink/stay predicted as `eat` fell only from `0.040609`
  to `0.028780` (`0.011829` absolute reduction), below the `0.05` audit floor.
  The post-carrion subset fell only from `0.046532` to `0.042549`; `stay`
  post-carrion slightly worsened (`+0.000791`).
- First residual post-carrion pattern: held-out
  `counterfactual-carrion_then_water-seed-37`, tick `16`, agent `8`, true
  `drink`, predicted `eat`. The first residual movement/drink/stay pattern in
  report order was seed `37`, tick `13`, agent `11`, true `move_west`,
  predicted `eat`.
- Result: the audit is negative, so no v64 promotion candidate was trained and
  no opt-in runtime/training path was added. The next branch should not
  compensate with scalar IQL tuning; it needs a stronger policy-owned temporal
  model or archive/replay method with a reportable held-out aliasing win before
  any heavy trainer run.

v65 hydration-after-carrion intervention audit:

- `output/mind/mind-v3-v65-hydration-after-carrion-audit.json` tested the
  narrow next hypothesis from the v64 residual: maybe post-carrion true
  `drink` predicted as `eat` is mostly a simple "drink now, then eat again"
  hydration-deficit alias. The report reuses the v64 deterministic
  ecological+rollout-context lookup and the same source split: `150`
  trajectories, `41,773` train rows from seeds `13,19,29`, and `42,983`
  held-out rows from seeds `37,41,43`.
- Focus rows were held-out post-carrion states where both `drink` and `eat`
  were legal and the true label was either `drink` or `eat`. There were
  `2,372` such rows: `1,791` true `drink`, `581` true `eat`, and `160`
  true-`drink` rows predicted as `eat`.
- The best policy-visible hydration-threshold intervention was `0.9`, but it
  reduced true-`drink` predicted-as-`eat` by only `0.018426`, below the `0.05`
  materiality floor, and it changed `8` previously correct `eat` labels to
  `drink` (`0.013769` eat-label damage, above the `0.01` cap).
- Drink/eat cycle aliasing was real but not dominant. All `160` residual
  true-`drink` rows successfully drank and none died on that row, but only
  `71` (`0.44375`) had a same-agent resource-gain `eat` within the next two
  rows, below the `0.6` floor. The seed `37`, tick `16`, agent `8` residual is
  one of these cycle cases, but the broader residual set is not explained by
  that pattern.
- Result: audit failed on all three blockers, so no v65 policy/training path
  was added and no RTX run is justified. The bottleneck is not a simple
  hydration-after-carrion rule; the next diagnostic should move away from
  threshold intervention and toward outcome/action-conditional labels or exact
  branch replay for ambiguous post-carrion states.

v66 branch-action oracle audit:

- `output/mind/mind-v3-v66-branch-action-oracle-audit.json` tests exact
  branch replay from ambiguous post-carrion states instead of copying the
  logged next action. The diagnostic restores an in-process simulator snapshot
  from immediately before the ambiguous decision tick, forces only the first
  action for the target agent, then resumes with the deterministic
  `carrion_then_water` continuation policy. This is diagnostic-only branch
  replay, not a runtime policy path.
- Scope: carrion-only seeds `37,41,43`, `120` ticks,
  `1` ambiguous branch point per seed, candidate first actions
  `drink,eat,stay`, and target logged labels `drink,eat`. The report includes
  each branch point's serialized `mind_observation_v3` input and full action
  mask, so the artifact is replayable and can be used as an explicit oracle
  label source without fixture identity as a runtime input.
- Result: `3` branch points, `9` action branches, `3` oracle-changed first
  actions, `2` material oracle gains, `+5` total terminal-alive gain versus
  the logged first action, `+4` total birth gain versus logged, deterministic
  replay verified, and zero heuristic action-source count. The first material
  gain is seed `37`, tick `16`, agent `10`: logged `drink`, oracle best `eat`,
  `+4` terminal alive, `-1` births.
- Interpretation: the failure mode is not mainly missing scalar context or a
  simple hydration threshold. At least some post-carrion ambiguity is a bad
  action-label / action-aliasing problem where the locally logged action is not
  the best next action under exact branch replay. The next step is worth
  pursuing as a branch-replay oracle label archive and distillation experiment;
  do not return to scalar IQL tuning or rollout-context runtime from v64.

v67/v68 branch-action oracle labels and expanded action audit:

- `output/mind/mind-v3-v67-branch-action-oracle-labels.json` converts the v66
  replay-verified branch audit into a deterministic policy-visible label
  archive. Each label carries the serialized `mind_observation_v3` input,
  observation digest, full action mask, oracle action, logged action, and
  action-conditioned branch outcomes. The archive passes label acceptance with
  `3` labels, `2` material oracle gains, `+5` terminal-alive gain versus
  logged, and dominant oracle action share `0.333333`.
- A wider `drink,eat,stay`-only preview over strict carrion seeds
  `13,19,29,37,41,43` remained materially positive but failed the action-share
  boundary: `24` labels, `+15` terminal alive, `+13` births, zero heuristic
  action sources, but `stay` was the oracle action for `14/24` labels
  (`0.583333`). This is an explicit negative control; do not relax the
  dominant-action cap to accept it.
- `output/mind/mind-v3-v68-branch-action-oracle-audit-expanded-actions.json`
  expands the candidate first-action set to include policy-visible movement
  actions: `stay,eat,drink,move_north,move_south,move_east,move_west`. With
  replay verification enabled, the strict carrion seed bank produced `12`
  branch points, `59` action branches, `12` oracle-changed labels, `9`
  material oracle gains, `+19` terminal alive, `+14` births, deterministic
  replay verified, and zero heuristic action-source count.
- `output/mind/mind-v3-v68-branch-action-oracle-labels-expanded-actions.json`
  passes label-archive acceptance: `12` complete policy-state labels, no
  conflicting observation digests, no unsupported oracle actions, dominant
  oracle action `move_west` at `5/12` (`0.416667`), under the `0.50` cap. This
  is the current best branch-oracle supervision artifact.
- The same v68 label archive includes a leave-one-source-seed-out nearest
  policy-vector support probe. That probe is negative: `2/12` correct
  (`0.166667` accuracy), predicted-action dominant share `0.5`, and
  `materially_supports_runtime_classifier=false`. Interpretation: branch-replay
  oracle labels are valuable and policy-visible, but this tiny archive does not
  justify a direct runtime classifier or exact lookup candidate. The next useful
  step is either a larger branch-label archive or a compact outcome/world-model
  diagnostic before any RTX training.

v69 expanded branch-label direction check:

- `output/mind/mind-v3-v69-branch-action-oracle-audit-expanded-actions-preview.json`
  is a no-replay-verification preview that doubles the movement-aware branch
  budget to `4` ambiguous branch points per strict carrion seed. It is
  materially positive as branch data: `24` branch points, `122` action
  branches, `24` oracle-changed labels, `17` material oracle gains, `+36`
  terminal alive, `+26` births, `+1` target alive, zero heuristic action
  sources, and clean oracle action distribution (`move_west` dominant at
  `7/24`, share `0.291667`). Because replay verification was deliberately
  skipped, it is a preview artifact only and is not accepted as a label archive.
- `output/mind/mind-v3-v69-branch-action-oracle-labels-expanded-actions-preview.json`
  adds two deterministic support probes. The direct oracle-action classifier
  remains negative: `4/24` correct (`0.166667`). The action-conditioned
  nearest-value ranker improves but is still below the materiality floor:
  best `8/24` correct (`0.333333`) at `k=3`, with
  `materially_supports_action_value_model=false`.
- Direction update: branch replay is confirmed as the right diagnostic/data
  source, and movement actions are necessary to avoid artificial `stay`
  collapse. However, simply adding more branch labels in this narrow state
  representation is not enough to justify a runtime classifier or value ranker.
  The next useful diagnostic should test a richer, policy-visible outcome model
  for hydration/energy/death and local movement consequences, or expand branch
  labels with additional serialized temporal/context features before any RTX
  training.

v70-v76 compact world-model and option decomposition diagnostics:

- The branch-action oracle label archive now carries action-conditioned
  first-action outcomes derived from replay records: energy/hydration/health
  deltas, moved/drank/ate flags, resource gain, death flags, and movement
  deltas. These are serialized diagnostic targets only; no runtime policy path
  was added.
- A compact, policy-visible world-model probe decodes each serialized
  `mind_observation_v3` input into self vitals, current and adjacent resource
  signals, local radius summaries, navigation signals, action mask state, and
  action-specific movement/eat/drink affordances. It then runs
  leave-one-source-seed-out nearest-neighbor support checks before any training.
- Result on the replay-verified v68 12-label archive: exact terminal oracle
  ranking reached only `0.333333`. Immediate first-step dynamics were easier
  (`0.75` actual first-step accuracy), but those immediate outcomes aligned
  with terminal oracle action only `0.25`; even an upper-bound probe using the
  actual first-step outcomes reached only `0.333333` terminal accuracy.
- Larger previews were not enough. The 24-label preview reached `0.416667`
  exact terminal support, while the 36-label preview dropped to `0.277778`.
  Option-mode prediction reached `0.625` on the 24-label preview but collapsed
  to `reposition` for `23/24` rows; reposition direction itself reached
  `0.714286`, but multi-move direction support stayed at or below `0.5`.
- Result: first-step dynamics are learnable, but the terminal action choice is
  delayed and multi-modal. Do not train a runtime policy from the compact
  one-row world-model branch.

v77-v85 long-horizon branch trace and population-world-model diagnostics:

- The branch-action oracle audit now serializes deterministic target-agent and
  population horizon traces for each forced branch at deltas
  `0,1,2,3,5,8,13,21,34,55,89`. The population trace records alive agents,
  births, deaths, target alive/vitals, tick resource gain, tick action counts,
  and dominant requested-action share. These fields are replay artifacts, not
  runtime policy inputs.
- `output/mind/mind-v3-v80-branch-action-oracle-audit-long-horizon-trace.json`
  and
  `output/mind/mind-v3-v80-branch-action-oracle-labels-long-horizon-trace.json`
  showed the first temporal-credit signal: actual future target trace at
  horizon `21` and actual future population trace at horizon `55` each matched
  the terminal oracle on `6/12` labels (`0.5`). This is diagnostic-only because
  actual future traces are not available at decision time.
- Decision-time trainability did not clear the floor. On the 12-label archive,
  compact population-horizon ranking reached only `4/12` (`0.333333`), and the
  full decoded policy observation plus action mask/action one-hot also reached
  only `4/12`. This falsifies the hypothesis that a compact feature summary
  alone caused the failure.
- `output/mind/mind-v3-v83-branch-action-oracle-audit-expanded-replay-horizon-trace.json`
  replay-verified the previously previewed larger branch set: strict
  `carrion_only` seeds `13,19,29,37,41,43`, `4` branch points per seed,
  `122` action branches, `24` oracle-changed labels, `17` material gains,
  `+36` terminal alive, `+26` births, `+1` target alive, zero heuristic action
  sources, and clean oracle distribution (`move_west` dominant at `7/24`,
  share `0.291667`).
- `output/mind/mind-v3-v85-branch-action-oracle-labels-material-horizon-model.json`
  passes label acceptance on the expanded replay set, but the support probes
  remain negative. Actual future population trace alignment drops to `9/24`
  (`0.375`), actual future target trace alignment to `7/24` (`0.291667`),
  compact population-horizon ranking to `6/24` (`0.25`), and full-observation
  population-horizon ranking to `7/24` (`0.291667`). Filtering to the `17`
  material-gain labels helps only slightly: material-only full-observation
  population-horizon ranking reaches `7/17` (`0.411765`), still below the
  `0.5` support floor.
- Result: v83/v85 is a real data milestone, not a trainable runtime milestone.
  The expanded replay archive is now accepted and stronger than v69, but the
  pointwise one-row outcome/world-model path is not ready for RTX training.
  The next useful slice should harden a deterministic public sequence/history
  label contract for the target agent or an options/archive replay contract
  that predicts delayed value after repositioning.

v86-v88 public history and continuation-option blocker checks:

- `output/mind/mind-v3-v86-branch-action-oracle-audit-public-history-trace.json`
  extends the accepted branch-action oracle audit with a deterministic
  same-agent public history trace. Each branch point now serializes the last
  `8` same-agent prior trajectory rows: tick/record deltas, requested/resolved
  actions, action validity, movement/resource flags, vital deltas, and
  rollout-context counters. This is derived from public trajectory rows only;
  it adds no private `SimulationWorld` read and no fixture identity as runtime
  input.
- The v86 audit preserves the v83 branch result exactly at the aggregate level:
  `24` branch points, `122` action branches, `24` oracle-changed labels, `17`
  material gains, `+36` terminal alive, `+26` births, `+1` target alive,
  replay verified, zero heuristic action sources, and no diagnostic blockers.
  All `24` branch points carry `8` public history rows.
- `output/mind/mind-v3-v87-branch-action-oracle-labels-public-history-model.json`
  adds history-aware support probes. They are negative: all-label
  full-observation+history population-horizon ranking reaches only `7/24`
  (`0.291667`), and material-only history ranking reaches `6/17`
  (`0.352941`). This is worse than the prior material-only full-observation
  probe at `7/17` (`0.411765`), so the next blocker is not simply missing the
  last few public rows.
- `output/mind/mind-v3-v88-branch-action-oracle-option-preview-*.json` screens
  continuation-option structure using the existing branch oracle harness with
  `2` branch points per strict carrion seed and replay verification disabled
  for speed. The baseline continuation remains strongest:
  `carrion_then_water` `+19` terminal alive / `+14` births, while
  `water_first_recovery` gives `+13` / `+13`, `conserve_after_carrion` gives
  `+12` / `+17`, `water_rescue_carrion_cycle` gives `+10` / `+15`, and
  `hydration_safe_carrion_cycle` gives `+6` / `+5`.
- Result: public short history and existing scripted continuation options are
  not the next training lever. Keep the v86 public-history contract because it
  is deterministic and useful for future sequence models, but do not train from
  the v87 pointwise history probe. The next useful blocker to attack is a
  true sequence/archive objective: score multi-step branch continuations or
  branch-state archive cells directly rather than asking a one-step nearest
  model to infer delayed repositioning value.

v89 standardized progress ledger and branch-continuation archive scorer:

- `output/mind/mind-v3-v89-branch-continuation-archive-scorer.json` tests the
  next proposed blocker without training a runtime policy. Input is a compact
  policy-visible branch-state archive cell plus same-agent public history
  summary and candidate action. The target is replayed multi-horizon
  population continuation value from the branch archive, not first-action
  imitation. The predeclared gate is leave-one-source-seed-out exact-action
  ranking at `>= 0.55`, materially above the prior weak `0.35-0.42` support
  range.
- Result: negative. All-label ranking reaches only `6/24` (`0.25`) with
  horizon `55`, `k=5`, same-action archive neighbors. Material-only ranking
  reaches `6/17` (`0.352941`) with horizon `21`, `k=1`, same-action neighbors.
  The branch-continuation archive scorer gate fails and runtime training is
  blocked.
- `output/mind/mind-v3-v89-standard-progress-ledger.jsonl` and
  `output/mind/mind-v3-v89-standard-progress-ledger-report.json` backfill a
  deterministic one-row-per-version progress ledger through v89. The progress
  rule is intentionally strict: a row counts as progress only if it is a strict
  policy pass or a diagnostic with a predeclared support floor that passes.
  Accepted data archives and unfloored audit wins are still tracked, but they
  are not counted as progress by that rule. The generated report has `89`
  rows and `6` progress rows: strict policy passes at `v41`, `v42`, and `v44`,
  plus diagnostic support-floor passes at `v80`, `v81`, and `v82`. None of
  `v83-v89` clears the progress rule.
- Decision: stop this archive-scorer line for now. The negative result is
  strong evidence that the next bottleneck is not just "score branch
  continuations from compact public archive cells." The remaining credible
  direction is to enlarge or restructure the branch archive itself before
  fitting another scorer: e.g. more replay-verified branch states with diverse
  continuation trajectories, or a sequence model diagnostic that can predict
  continuation value from entire public trajectory prefixes under the same
  leave-one-seed-out floor.

v90 mode-balanced target-local objective diagnostic:

- `output/mind/mind-v3-v90-existing-branch-mode-objective-audit.json` audits
  the existing v83/v87 replay-verified archive before changing selection. It
  confirms the coverage problem: `24` labels are evenly split by seed, but
  population-first oracle modes are skewed to `reposition` (`14/24`) with only
  `1` `recover_hydration` label. Target-local and option-mode labels disagree
  sharply with the population-first labels: exact population-vs-target
  agreement is `0.083333`, and mode agreement is `0.166667`. Support remains
  negative on the existing archive: option-mode accuracy `0.375`,
  material-only option-mode accuracy `0.470588`, and reposition exact
  direction accuracy `0.375`.
- `output/mind/mind-v3-v90-balanced-branch-action-oracle-audit.json` adds an
  opt-in balanced branch-point selector. It scans all eligible post-carrion
  rows per seed instead of stopping after the first four, admits logged
  move/stay rows when cross-mode alternatives are legal, and balances selected
  rows across seed, logged option mode, energy/hydration bins, and visible
  water/carrion distance bins. Replay verification stays enabled. The preview
  accepts diagnostically with `30` branch points, `125` action branches, `26`
  changed population-first oracle actions, `+43` terminal alive, `+51` births,
  zero heuristic runtime action sources, and replay verification passing.
- `output/mind/mind-v3-v90-balanced-branch-mode-objective-audit.json` compares
  population-first, target-local, and option-mode objectives on that balanced
  archive. Coverage is broader: labels by seed are `5` each, population-first
  oracle modes are `reposition=16`, `conserve=8`, `exploit_resource=5`,
  `recover_hydration=1`, and logged move/stay rows are present in the
  logged-to-oracle matrix. The target-local objective exposes the objective
  mismatch: population-first oracle choices improve population alive/births
  on average (`+1.433333` alive, `+1.566667` births vs logged) while target
  local deltas are mostly non-positive (`target_alive_delta` mean
  `-0.133333`, target energy mean `-0.069383`, hydration mean `-0.129273`,
  health mean `-0.12709`).
- Predeclared support floors: option-mode accuracy `>=0.60`, dominant predicted
  option mode `<=0.75`, material-only option-mode accuracy `>=0.55`,
  reposition exact direction multi-move accuracy `>=0.55`, no unsupported
  oracle actions, and zero heuristic runtime action sources. The balanced
  archive clears the first three model floors (`option_mode=0.766667`,
  dominant predicted mode share `0.633333`, material-only option mode
  `0.666667`) and keeps oracle/runtime safety clean, but fails reposition
  exact direction at `10/20` (`0.5`) versus the `0.55` floor.
- Decision: v90 is a diagnostic improvement, not a training pass. Runtime
  training remains blocked. The next runtime path would only be justified after
  a follow-up diagnostic can lift multi-move reposition direction above floor
  without collapsing the option-mode head.

v91 failure-frontier reposition diagnostic:

- `output/mind/mind-v3-v91-failure-frontier-branch-action-oracle-audit.json`
  adds the opt-in `failure_frontier_mode_balanced_v1` selector. It scans all
  eligible post-carrion rows, preserves six-seed carrion fixture coverage, and
  prefers later/vital-debt/frontier rows with multi-move ambiguity and
  policy-visible water/carrion/resource context. Replay verification stays on.
  The archive itself accepts diagnostically with `48` branch points, `263`
  action branches, `45` oracle/logged disagreements, `+61` terminal alive,
  zero heuristic runtime action sources, and replay verification passing.
- `output/mind/mind-v3-v91-failure-frontier-branch-action-oracle-labels.json`
  produces `48` labels, `27` material oracle-gain labels, dominant oracle
  action share `0.333333`, and clean label-archive acceptance. Coverage is
  balanced by seed (`8` labels each), but the target-local option oracle shifts
  sharply away from reposition on the actual frontier:
  `exploit_resource=33`, `reposition=9`, `recover_hydration=4`, `conserve=2`.
  The source selector proved enough candidate coverage (`1280` eligible rows,
  `1201` eligible multi-move rows, `46` selected multi-move rows), so the low
  actual reposition label count is an objective result, not an eligibility
  shortage.
- `output/mind/mind-v3-v91-reposition-frontier-audit.json` decomposes the v90
  blocker. Best reposition-direction decoder accuracy is `5/8` (`0.625`) via
  `current_value_k5`, above the `0.60` direction floor, with errors split as
  `wrong_axis=2` and `opposite_axis=1`. That improvement is not enough:
  option-mode support collapses on the failure-frontier slice (`0.583333`,
  floor `0.70`), material-only option-mode support is just below floor
  (`0.592593`, floor `0.60`), and only `8` option-mode reposition multi-move
  labels remain against the `36` floor.
- Branch utility blocks runtime work. Option-mode predicted actions have
  positive mean population deltas versus logged (`+0.583333` terminal alive,
  `+0.125` births), but mean target-local score delta is negative
  (`-12.951134`) and target vitals regress on average (energy `-0.014044`,
  hydration `-0.020773`, health `-0.011208`). This confirms the caveat from
  v90: option-mode accuracy alone is not useful unless it improves target-local
  continuation utility near the real carrion failure frontier.
- Decision: v91 is rejected. No runtime policy or RTX training is allowed from
  this line. The current bottleneck is objective mismatch at frontier states:
  replayed population gains mostly relabel ambiguous rows as resource
  exploitation, while target-local utility for predicted actions remains
  negative.

v92 catastrophe-sensitive branch utility diagnostic:

- `output/mind/mind-v3-v92-branch-utility-risk-audit.json` reuses the v91
  failure-frontier labels and scores all `263` replayed candidate actions
  directly with a deterministic leave-one-source-seed-out utility/risk scorer.
  The feature contract is policy-visible only: decoded observation values,
  action mask, compact self/local/navigation fields when decodable, same-agent
  public history trace, and candidate action identity. Held-out seed leakage is
  `0`; no rule uses logged action as a runtime fallback, seed/branch id,
  fixture identity, or hidden simulator state as a feature.
- v92 compares the v91 option-mode baseline, mean predicted target-local
  utility, lower-confidence-bound utility, CVaR-style utility, explicit
  target-death-risk veto, vital-regression-risk veto,
  `action_family_balanced_risk_scorer_k5`, and navigation/resource baselines.
  Navigation/resource rules remain baselines only and are not acceptance
  candidates.
- The best acceptance-eligible rule is
  `action_family_balanced_risk_scorer_k5`, but it still fails four strict
  floors: mean target-local score delta is `-14.142929` (must be `> 0`),
  per-seed target-local mean is negative for seed `41` (`-127.514088`), target
  alive delta has `1` negative case (must be `0`), and dominant predicted action
  share is `0.729167` on `eat` (cap `0.50`). Mean terminal population deltas
  are positive (`+0.5625` terminal alive, `+0.104167` births), so population
  gain is still masking target-local catastrophe.
- The known catastrophic class is reproduced under leave-one-seed-out without
  hardcoding: seed `41`,
  `carrion-only-seed-41-action-branch-0-tick-113-agent-18-logged-move-south`
  is still predicted as `eat`, with target-local score delta `-1055.34151`,
  terminal alive delta `-1`, and target alive delta `-1`. The risk/veto family
  does not avoid this failure.
- Decision: v92 is rejected. v93 runtime work is not allowed. The current
  blocker is not option mode or move direction; it is branch-state continuation
  risk calibration. The offline scorer cannot yet identify target-death
  catastrophes from policy-visible support without collapsing toward `eat`.

v93 depleted-resource trap support diagnostic:

- `output/mind/mind-v3-v93-depleted-resource-trap-support-branch-action-oracle-audit.json`
  generates replay-verified support for policy-visible trap states from
  non-held-out seeds only (`1,2,3,4,6,7,8,9,10,11,12,17,23,31`). The strict
  carrion evaluation seeds (`13,19,29,37,41,43`) are excluded from support.
  The selector `depleted_resource_trap_v1` found `56` eligible trap rows and
  selected `51`, clearing the predeclared `40` support-row floor; replay
  verification is true and heuristic action-source count is `0`.
- `output/mind/mind-v3-v93-depleted-resource-trap-support-labels.json` labels
  those `51` support branch points (`267` candidate action runs). It has `39`
  oracle-changed labels, `12` material-gain labels, terminal alive gain total
  `19`, birth gain total `12`, and dominant oracle action share `0.352941`.
- `output/mind/mind-v3-v93-depleted-resource-trap-audit.json` evaluates the
  support set against the existing strict v91 frontier (`48` labels). The seed
  `41` tick `113` catastrophe is avoided for the selected diagnostic rule, but
  the best acceptance-eligible rule,
  `trap_support_action_family_balanced_k5`, still fails four floors: mean
  target-local score delta is `-20.003307` (must be `> 0`), per-seed
  target-local mean remains negative for seed `41` (`-149.153556`), target
  alive delta has `1` negative case (must be `0`), and mean birth delta is
  `-0.020833` (must be `>= 0`). Dominant predicted action share improves to
  `0.395833`, below the `0.50` cap, but that diversity is not useful enough.
- The audit also exposes the deeper issue: fixing the original depleted-eat
  trap can create a neighboring frontier catastrophe. On seed `41`, branch
  `carrion-only-seed-41-action-branch-1-tick-114-agent-18-logged-eat`, the
  trap-support rule predicts `move_east` while logged/target-local `eat` keeps
  the target alive, producing target-local score delta `-1130.74377` and target
  alive delta `-1`.
- `output/mind/mind-v3-v93-standard-progress-ledger.jsonl` and
  `output/mind/mind-v3-v93-standard-progress-ledger-report.json` update the
  one-row-per-version ledger through v93. The ledger now treats the
  version-specific v93 trap audit probe as authoritative over generic label
  probes, so v93 is recorded as `diagnostic_support_floor_fail`; progress-passed
  versions remain `[41,42,44,80,81,82,90]`.
- Decision: v93 is rejected. v94 runtime work is not allowed. The depleted
  resource trap is real and support can avoid the named seed `41` branch, but
  one-step trap/action scoring does not preserve strict frontier utility. The
  next useful line is a true sequence/world-model branch continuation scorer,
  not another one-step scorer over the v91 archive.

v94 sequence/world-model branch-continuation scorer diagnostic:

- `output/mind/mind-v3-v94-branch-sequence-continuation-scorer.json` builds a
  replay-backed sequence-continuation dataset from the v93 non-held-out support
  archive and evaluates on the v91 strict carrion frontier. Support uses `51`
  non-strict rows (`267` candidate actions) from seeds
  `1,2,3,4,6,7,9,11,12,17,23,31`; strict evaluation remains the six carrion
  seeds `13,19,29,37,41,43` with `48` rows and `263` candidate actions. Strict
  seed training leakage is `0`; replay verification is true for support and
  strict labels; heuristic action-source count and unsupported oracle action
  count are both `0`.
- The v94 scorer treats `first_action_outcome`, `target_horizon_trace`, and
  `population_horizon_trace` as replay targets, not runtime inputs. Runtime
  features are policy-visible only: decoded observation values, action mask,
  compact self/local/navigation state when decodable, same-agent public history
  trace, candidate action identity, move direction, and candidate support.
- Compared with the v93 one-step trap baseline, sequence scoring fixes the
  two seed `41` frontier catastrophes as a class. The best acceptance-eligible
  rule, `sequence_prefix_nearest_neighbor_k5`, avoids target death on both
  `carrion-only-seed-41-action-branch-0-tick-113-agent-18-logged-move-south`
  and
  `carrion-only-seed-41-action-branch-1-tick-114-agent-18-logged-eat`. It also
  reaches global mean target-local score delta `+8.359369`, target-alive
  negative count `0`, mean terminal alive delta `+0.4375`, and mean birth delta
  `+0.041667`.
- v94 still fails the predeclared strict floor because the best rule predicts
  `eat` on `30/48` strict comparisons, dominant predicted action share
  `0.625` versus cap `0.50`. Dominant predicted mode share is also `0.625`,
  below the `0.75` cap. All other strict floors for that rule clear, including
  per-seed target-local mean: seed `13` `+9.128975`, `19` `+9.436875`, `29`
  `+1.790025`, `37` `+12.871388`, `41` `+0.619016`, and `43` `+16.309938`.
- `output/mind/mind-v3-v94-standard-progress-ledger.jsonl` and
  `output/mind/mind-v3-v94-standard-progress-ledger-report.json` update the
  one-row-per-version ledger through v94. The ledger treats the v94
  `sequence_continuation_support_probe` as authoritative; v94 is recorded as
  `diagnostic_support_floor_fail`, and progress-passed versions remain
  `[41,42,44,80,81,82,90]`.
- Decision: v94 is rejected. v95 runtime work is not allowed. This is the first
  post-v93 diagnostic to show positive strict target-local utility while
  avoiding both seed `41` continuation catastrophes, but it still violates the
  anti-collapse action-share contract. The next line, if pursued, should be
  simulator-in-the-loop planning/search with an explicit diversity/action-mask
  contract, not another offline one-step or sequence scorer over the same
  archive.

v95 simulator-in-the-loop constrained planning diagnostic:

- `output/mind/mind-v3-v95-branch-constrained-planning-audit.json` uses the
  replay-verified strict v91 branch action outcomes as simulator-in-the-loop
  evidence and the v94 `sequence_prefix_nearest_neighbor_k5` decisions as the
  baseline. It does not train or emit a runtime policy. Strict evaluation
  remains `48` carrion frontier branches (`263` candidate outcomes) across
  seeds `13,19,29,37,41,43`; replay verification is true, heuristic action
  source count is `0`, and unsupported oracle/logged action counts are `0`.
- The v94 baseline has useful survival utility but violates the action cap:
  `eat` is selected `30/48` times (`0.625`, cap `0.50`), with target-alive
  negative count `0`, mean target-local score delta `+8.359369`, mean terminal
  alive delta `+0.4375`, and mean birth delta `+0.041667`.
- The minimum-cost cap repair is `greedy_constrained_planner_v1`. It replaces
  exactly six `eat` decisions, reducing `eat` to `24/48` (`0.50`) with zero
  target-alive regressions, mean target-local score delta `+8.699062`, mean
  terminal alive delta `+0.5625`, and mean birth delta `+0.0625`. The utility
  cost versus v94 is `0.0`; the target-local sum increases by `+16.30524`.
  Replacements:
  `carrion-only-seed-13-action-branch-2-tick-47-agent-3-logged-drink`
  `eat -> stay` (`+0.0` target-local delta),
  `carrion-only-seed-13-action-branch-3-tick-25-agent-8-logged-eat`
  `eat -> stay` (`+1.80794`),
  `carrion-only-seed-29-action-branch-5-tick-66-agent-3-logged-drink`
  `eat -> drink` (`+0.0015`),
  `carrion-only-seed-37-action-branch-3-tick-27-agent-9-logged-eat`
  `eat -> stay` (`+5.2009`),
  `carrion-only-seed-37-action-branch-5-tick-34-agent-4-logged-drink`
  `eat -> move_south` (`+0.8307`), and
  `carrion-only-seed-43-action-branch-5-tick-62-agent-18-logged-drink`
  `eat -> move_south` (`+8.4642`).
- All four planner variants clear the v95 diagnostic floors. The best
  diagnostic rule is `diversity_regularized_planner_v1`: dominant action share
  `0.25` (`eat` `12/48`), target-alive negative count `0`, mean target-local
  score delta `+37.156839`, mean terminal alive delta `+0.791667`, mean birth
  delta `+0.166667`, and both seed `41` tick `113` and tick `114`
  catastrophes avoided. The beam/global planner reaches slightly higher mean
  target-local score (`+37.463294`) while sitting exactly on the `0.50` action
  cap.
- `output/mind/mind-v3-v95-standard-progress-ledger.jsonl` and
  `output/mind/mind-v3-v95-standard-progress-ledger-report.json` update the
  one-row-per-version ledger through v95. The ledger treats
  `constrained_planning_support_probe` as authoritative, records v95 as
  `diagnostic_support_floor_pass`, and progress-passed versions become
  `[41,42,44,80,81,82,90,95]`.
- Decision: v95 is accepted as a diagnostic-only upper bound. It proves the
  v94 anti-collapse cap is not structurally incompatible with strict frontier
  utility when simulator-backed branch outcomes can be planned globally. v96
  distillation/runtime-feasibility work is allowed, but no v95 planner result
  is runtime-ready because it uses replayed candidate outcomes as planning
  evidence.

## Promotion Boundary

Mind v3 can replace the current baseline only after it independently sustains
alive agents and births across the extended seed matrix. Guarded fallback rates
are not promotion metrics for this track because v3 has no heuristic fallback.
Promotion should also require diversity evidence: survival and births must not
come from one brittle terrain/diet/action niche only.

## Research Direction

The next useful work is exact temporal-credit data generation, then policy
capacity. The current v36 audit means the learner does not yet have positive
`120`-horizon survival support on the target transition. Another offline
learner run on the same label distribution would mostly test extrapolation, not
learning.

Research-backed direction after v41:

1. Build an exact branch-and-explore data generator before training another
   artifact. The immediate target is not a learned world model; this simulator
   already provides deterministic dynamics. Use that advantage to branch from
   promising carrion-contact or post-contact states, explore bounded
   policy-visible macro actions, and archive terminal survivors. This follows
   the Go-Explore principle of remembering promising states, returning to them,
   then exploring from them in a deterministic simulator.
2. Keep quality diversity as the search backbone, but move the archive from
   "best whole-run controller" toward "best recovery continuations". Archive
   cells should cover time since carrion contact, energy/hydration bins,
   carrion and water reachability, movement spend, births, terminal alive, and
   dominant action share. This matches MAP-Elites/QD: keep a diverse set of
   high-performing behaviors rather than over-selecting one broad-score winner.
3. Use vectorized/process rollout workers for the branch search. OpenAI ES and
   related neuroevolution results support this for simulator-heavy work because
   candidates can be evaluated in parallel with low communication. CUDA is not
   the bottleneck until we train a neural policy or learned dynamics model.
4. Keep distillation behind a broad-plus-fixture acceptance matrix. The v41
   local slice proves the bridge can beat linear on broad seeds while retaining
   a carrion scavenging lift, but the training data still needs positive
   `carrion_only@120` terminal survivors and post-contact survivor coverage on
   held-out fixture seeds before this becomes a real promotion candidate.
5. Treat Dreamer/MuZero-style world models as the next major phase, not the
   next implementation slice. They are relevant because they learn dynamics and
   train or plan through imagined futures, but in this repo the first scalable
   model is the exact simulator. A learned world model should start only after
   exact branch rollouts have produced a validated curriculum and compact state
   contract.

Research-validation loop for subsequent Mind v3 runs:

- After every candidate run, record the exact command, output artifact paths,
  acceptance status, broad open-world deltas versus linear, fixture deltas,
  heuristic action counts, and the first seed or fixture that breaks.
- After every candidate run, do one lightweight research pass against primary
  sources or lab reports relevant to the observed failure mode. The research
  pass should not override local data. It should test whether the next proposed
  code or data change is consistent with current evidence about offline RL data
  coverage, world models, quality diversity, open-ended curricula, or
  foundation-model-assisted search.
- Prefer research-grade implementations only when they fit the repo contracts:
  deterministic replay, summary-only lightweight sweeps, explicit held-out
  gates, serialized artifacts, and no hidden heuristic action selection. Do not
  import large frameworks or training dependencies just because a paper uses
  them; first prove the local data and acceptance surface.
- Keep run artifacts in `output/mind/` and durable interpretation in this file
  so future sessions can compare results without relying on chat history.

May 13, 2026 research synthesis before v42:

- AdA/XLand-style adaptation work points toward broad task distributions,
  held-out dynamics, memory, and automated curricula. That supports validating
  v41 on the broader `5,13,19,29,37,41` seed matrix before tuning another
  residual.
- DreamerV3, Genie, Sora, and V-JEPA keep world models strategically relevant,
  but they do not remove the need for data support and exact held-out gates.
  For this simulator, exact branch rollouts are the first world model.
- VPT, RT-2, and SIMA reinforce the same data-coverage lesson: action-grounded
  agents improve when the training distribution covers the behaviors and
  contexts they must execute. The current recovery archive has only `5`
  trajectories and survivor cells only from seed `29`, so it is not enough for
  a serious torch/IQL run.
- ASAL, FunSearch, and AlphaEvolve support quality-diverse search with automated
  evaluators. The repo should expand the recovery archive across fixture seeds,
  branch points, and survivor niches before trying to learn a stronger policy.
- IQL remains relevant only after the archive contains enough supported
  successful target-horizon behavior. Its conservative policy-improvement
  design does not make a five-trajectory archive solve held-out carrion
  survival.

Post-v42 research check:

- The v42 result matches the offline-RL data-coverage warning: the candidate can
  safely improve broad behavior where the anchored linear policy already has
  support, but it does not invent positive terminal carrion survivors for
  under-covered fixture seeds. Recent offline RL work on distribution shift and
  partial data coverage keeps pointing at coverage and support constraints as
  the limiting factor, not at another local residual adjustment:
  <https://arxiv.org/abs/2310.18434>.
- The aggregate fixture scavenging lift with no terminal survivors matches the
  Go-Explore/QD diagnosis: useful stepping-stone behaviors have been found, but
  the archive must preserve more recovery continuations across seed-specific
  descriptors before distillation can generalize:
  <https://arxiv.org/abs/1901.10995>,
  <https://www.ijcai.org/proceedings/2024/773>.
- ASAL-style artificial-life search reinforces the same next step: use
  automated evaluators to illuminate a diverse simulation behavior space. For
  this repo, that means expanding the exact carrion recovery archive across
  fixture seeds and branch points before using foundation-model or torch-heavy
  machinery:
  <https://arxiv.org/abs/2412.17799>.

Post-v43 research check:

- The expanded archive result is aligned with recent offline RL coverage work:
  adding branch coverage and survivor continuations changed the data support
  directly, instead of asking a learner to extrapolate the missing
  `carrion_only@120` survivor behavior. That is consistent with partial
  coverage and distribution-shift warnings in offline RL:
  <https://openreview.net/forum?id=AfXq3x3X16>.
- Go-Explore remains the closest procedural template for this slice: archive
  promising states, return to them deterministically, explore continuations,
  then robustify or distill. The v43 result is exactly the missing Phase 1
  support that v42 lacked:
  <https://arxiv.org/abs/1901.10995>.
- Quality-diversity artificial-life work supports retaining multiple behavior
  descriptors rather than only the single best recovery trajectory. The
  seed/branch/continuation spread in v43 is therefore more valuable than the
  best survivor alone:
  <https://arxiv.org/abs/2406.04235>.
- DeepMind's 2024 open-endedness position paper reinforces the broader
  milestone order: do not collapse immediately to one narrow distill if the
  goal is autonomous ecological competence. Use the expanded archive as a
  curriculum and keep measuring diversity under held-out seeds:
  <https://arxiv.org/abs/2406.04268>.

Post-v43 expanded-distill research check:

- The v43 distill failure matches multi-task/offline data-sharing warnings:
  adding more data can hurt the target policy when the shared dataset changes
  the deployment distribution without uncertainty or relevance gating. That is
  exactly what happened here: the expanded carrion recovery archive improved
  local support, but the broad open-world policy regressed:
  <https://arxiv.org/abs/2404.19346>.
- Recent robust offline imitation work argues for selecting or weighting
  high-quality auxiliary transitions instead of directly cloning all diverse
  auxiliary data. For this repo, that means the expanded archive should feed a
  gated or relevance-weighted residual, not a single undifferentiated residual
  over every broad state:
  <https://arxiv.org/abs/2410.03626>.
- Behavior-cloning imbalance work reinforces the action-level symptom in this
  run: small residual changes shifted the broad policy toward more `drink` and
  `stay` and less movement. The next code change should expose and control
  behavior/action-balance by context, not only aggregate survivor quality:
  <https://arxiv.org/abs/2508.06319>.
- Google DeepMind's 2024 large-scale offline actor-critic result supports the
  eventual RTX/IQL direction because offline actor-critic can outperform pure
  supervised cloning on mixed-quality multi-task data. It does not support
  skipping the current acceptance failure; it supports moving beyond naive
  residual distillation once the dataset and broad acceptance surface are
  stable:
  <https://arxiv.org/abs/2402.05546>.

Post-v44 research check:

- v44 is consistent with robust offline imitation research: diverse auxiliary
  data is useful when high-quality or relevant transitions are selected or
  weighted, but unsafe when blindly cloned. The local context gate is a simple
  deterministic version of that relevance filter:
  <https://arxiv.org/abs/2410.03626>.
- The result also matches multi-task offline data-sharing work: adding related
  data can help or hurt depending on distribution shift and uncertainty. v43
  was negative transfer; v44 constrained transfer to states where the recovery
  archive is relevant:
  <https://arxiv.org/abs/2404.19346>.
- DeepMind's offline actor-critic scaling result remains the strongest argument
  for eventually moving beyond supervised residual distillation on mixed
  expert/suboptimal data. The v44 result says that future RTX/IQL should start
  from a context-aware acceptance surface, not from the ungated v43 policy:
  <https://arxiv.org/abs/2402.05546>.
- The remaining terminal-extinction failure looks less like broad negative
  transfer and more like missing state/history: the residual sees visible
  carrion context, but cannot remember post-contact recovery phase once the
  immediate carrion signal disappears. The next research-aligned slice is
  either explicit recovery-phase features/memory or a branch-state-conditioned
  value model, with stricter seed-level broad gates.

Post-v45 research check:

- v45 follows the RLiable/Google Research evaluation warning that aggregate
  point estimates can mislead in small-seed RL comparisons. The new
  `candidate_vs_linear_per_seed` payload keeps paired seed deltas visible and
  makes the promotion gate fail when one seed regresses despite positive mean
  deltas:
  <https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/>.
- The underlying NeurIPS 2021 "statistical precipice" paper argues for
  exposing variability across runs and avoiding conclusions from means alone.
  The local analogue is strict no-regression over each held-out seed before a
  recovery-distilled policy can be promoted:
  <https://arxiv.org/abs/2108.13264>.
- The result also stays aligned with robust offline imitation/data-sharing
  work: the context gate reduced broad negative transfer, but v45 shows that
  relevance filtering must be validated at the seed level, not only by fixture
  averages or aggregate open-world means.

Post-v60 research check:

- 2025-2026 offline RL work still treats behavior support and OOD value/action
  control as core failure modes. Support-constraint work explicitly frames
  offline RL errors as distribution shift between learned and behavior
  policies, while newer neighborhood constraints classify density, support, and
  sample constraints as the main families for controlling extrapolation:
  <https://arxiv.org/abs/2503.05207>,
  <https://arxiv.org/abs/2511.02567>.
- The newest flow-policy line is relevant because it attacks the same local
  symptom: complex or multi-modal behavior distributions are poorly represented
  by simple actor heads. Flow Q-Learning, Flow Actor-Critic, and the May 2026
  Flow-Anchored Noise-conditioned Q-Learning paper all point toward expressive
  policies plus conservative/behavior-regularized value learning, not toward
  relaxing held-out gates:
  <https://arxiv.org/abs/2502.02538>,
  <https://arxiv.org/abs/2602.18015>,
  <https://arxiv.org/abs/2605.01663>.
- 2026 behavior-regularized RL is also exploring implicit transport/flow-style
  policy updates. Value Gradient Flow keeps a reference distribution and
  controls the transport budget, which is conceptually closer to a rollout-state
  support regularizer than to the v60 full-prior blend:
  <https://arxiv.org/abs/2604.14265>.
- World-model work is relevant as the next major architecture branch, but it
  should stay behind exact simulator evidence. Differentiable world-model MPC
  for offline RL uses inference-time imagined rollouts to adapt the policy, and
  contextual latent world models use temporal consistency for generalization:
  <https://arxiv.org/abs/2603.22430>,
  <https://arxiv.org/abs/2603.02935>.
- Local conclusion:
  the repo is aligned with current research on evaluation rigor, behavior
  support, and conservative offline learning. The mismatch is that the current
  discrete IQL actor is still accepted or rejected mostly by deployment
  rollout diagnostics after training, while the regularizers are train-batch
  marginals. The next serious implementation should constrain or train on
  rollout-state top-1 action behavior, or move to a sequence/flow/world-model
  branch with the same strict seed-level acceptance matrix.

Major milestones from the current state:

- v37: deterministic branch-and-explore harness. It can replay or restore a
  `carrion_only` fixture to a branch point, run candidate continuations from
  there, and emit branch trajectory JSONL plus a `temporal-credit-audit` report.
  Acceptance: at least one positive `120` terminal survivor per target fixture
  seed, nonzero post-contact survivor count, and reproducible branch replay.
- v38: quality-diverse recovery archive. It expands the branch harness into a
  process-parallel search over recovery macros/controller mutations, stores
  elites by recovery descriptors, and exports a balanced survivor/failure
  training set. Acceptance: temporal-credit audit passes on the exported data,
  and the archive contains more than one survivor niche.
- v39-v41: outcome-audited policy distillation from recovery data. The current
  slice adds explicit survival/reproduction/scavenging telemetry, trains a
  deterministic anchored-neural artifact from recovery elites, and stores
  artifact-scoped residual controls so broad survival is not sacrificed for a
  fixture-specific carrion gain. The next slice should expand archive coverage
  and repeat the same acceptance matrix on held-out seeds. This is where IQL,
  behavior cloning, sequence modeling, or a compact recurrent policy becomes
  useful again.
- v42: held-out recovery-distill validation. Run the v41 residual-safe
  distillation path against broad open seeds `5,13,19,29,37,41` and
  `carrion_only` fixture seeds `29,37,41`. Acceptance questions: does the
  candidate match or beat linear on broad open survival and births, keep `0`
  heuristic runtime actions, retain fixture scavenging lift, and show which
  seed or fixture breaks first? This is a validation slice, not a tuning slice.
- v43: recovery-archive expansion. If v42 passes broad open but carrion-only
  remains extinct, expand archive coverage across more fixture seeds, branch
  points, continuation scripts, and survivor niches. Require positive
  `carrion_only@120` survivors from more than one seed before re-distilling or
  training RTX torch/IQL.
- v44: context-gated recovery residual. The expanded archive can be distilled
  without broad open-world regression only when the recovery residual is
  context-selective. Acceptance: no aggregate broad alive/birth regression
  versus linear, `0` heuristic actions, and fixture scavenging lift without
  fixture birth/scavenging regression. The next hardening step is seed-level
  broad no-regression plus a path toward terminal carrion-only survivors.
- v45: strict seed-level recovery acceptance. Aggregate broad open gains are
  insufficient; each held-out open seed must avoid alive and birth regression
  versus linear. The current context-gated candidate fails this stricter gate
  on seed `13` births.
- v46: targeted archive expansion under the strict gate. Archive expansion
  succeeded with survivor support across six carrion-only fixture seeds, but
  re-distillation still failed strict open-seed promotion on seeds `5` and `13`
  and still produced no terminal carrion-only survivors.
- v47: recovery-phase/state carryover. A policy-visible post-carrion-contact
  memory gate was added and validated, but the strict candidate still failed on
  seeds `5` and `13` and still produced no terminal carrion-only survivors.
- v48: branch-state-conditioned scoring. Add a compact value/support signal or
  recovery-phase action-value bias derived from survivor continuations. This
  artifact/scoring path is implemented and validated, but the strict candidate
  still fails on seeds `5` and `13` and no `carrion_only@120` fixture survivor
  appears.
- v49: local-resource eat guard audit. Protecting linear `eat` on current-tile
  animal resource is implemented and tested, but it fired only once in the
  broad-open run and did not change strict outcomes. The next slice must expand
  branch archive data rather than add another tiny residual guard.
- v50: learned dynamics/world-model pilot. Only after v37-v49 prove the target
  data and policy contract, train a compact dynamics/value model for short
  observation-space rollouts or MuZero/Dreamer-style planning. Acceptance is
  not promotion; it is matching exact branch decisions on held-out branch
  states and improving search throughput without inventing invalid survivors.
- v51: open-ended/autonomous curriculum. If the recovery path moves carrion,
  fold the task back into broader ecology with XLand-style dynamic task
  distributions and PBT/QD scheduling so the policy does not overfit a single
  fixture lane.
- v52: deterministic residual boundary. Recovery-window extension did not
  repair strict open-seed regressions or produce terminal carrion-only
  survivors, so the archive became ready for a value/sequence learner but not
  for promotion.
- v53-v56: first RTX/IQL boundary. IQL produced the first nonzero terminal
  `carrion_only@120` survivors and broad open gains without heuristic runtime
  actions, but failed the action-diversity cap through dominant `eat`
  concentration.
- v57-v60: strict IQL audit boundary. Archive acceptance now counts only
  trainable counterfactual survivor trajectories, labeled IQL acceptance now
  blocks per-seed alive/birth regressions, and coefficient/prior probes still
  failed dominant-action and seed-level gates. The next IQL work must train or
  constrain rollout-state top-1 action behavior instead of repeating aggregate
  marginal or prior-blend tuning.
- v61: risk-adjusted extraction boundary. Adding risk-adjusted actor extraction
  to the sharp action-distribution IQL setup passed the train gate but failed
  the strict slice worse than v59: no carrion-only movement, dominant `eat`
  share above cap, and open-seed regressions on seeds `5`, `19`, and `37`.
- v62-v63: rollout-state calibration boundary. Global actor-bias calibration
  found the calibration-bank actor was dominated by `stay`/`move_north`, while
  strict deployment still concentrated on `eat`; lowering scalar contextual-prior
  blend made the slice substantially worse. Treat the current IQL representation
  and coefficient family as exhausted for promotion. The next slice should open a
  rollout-context policy branch, not another extraction, prior-blend, or global
  action-share variant against the same acceptance surface.
- v64: rollout-context audit boundary. Deterministic previous-row context
  improved held-out action-label accuracy but reduced movement/drink/stay
  predicted-as-`eat` by only `0.011829`, below the audit floor, and left
  post-carrion `drink`/`stay` aliases. Do not train a v64 rollout-context IQL
  candidate from this audit result alone.
- v65: hydration-after-carrion audit boundary. A simple post-carrion hydration
  threshold intervention did not materially reduce true-`drink` predicted-as-
  `eat` and caused eat-label damage; drink-then-eat cycle aliasing explained
  only `0.44375` of residual true-`drink`/predicted-`eat` cases. Stop before
  training.
- v66: exact branch-action oracle positive diagnostic. Ambiguous post-carrion
  states can have a better first action than the logged action under exact
  replay; the compact held-out carrion slice produced `+5` total terminal
  alive, zero heuristic action sources, and replay-verified serialized
  observation/action-mask labels. This justifies a small branch-oracle label
  archive/distillation path before any RTX training.
- v67/v68: branch-oracle label archive boundary. The expanded movement-aware
  branch audit produced a replay-verified, action-share-clean label archive
  with `12` labels and `+19` terminal alive versus logged. However, a
  leave-one-source-seed-out nearest-vector probe reached only `0.166667`
  accuracy, so the labels are not yet sufficient for a runtime classifier.
  Continue with more branch labels or an outcome model; do not deploy a lookup
  policy from this archive.
- v69: larger branch-label preview confirms direction but not learnability.
  Doubling branch points gives stronger oracle outcome deltas (`+36` terminal
  alive, `+26` births) and clean action distribution, but classifier support is
  still `0.166667` and action-conditioned value ranking is only `0.333333`.
  Do not spend RTX on this label set as-is; test richer outcome features/model
  capacity first.
- v70-v76: compact world-model/option-mode diagnostic boundary. A
  policy-visible compact outcome representation was added over serialized
  observation inputs, action masks, local resources/navigation, first-action
  action outcomes, and terminal branch values. The replay-verified 12-label
  slice still ranked exact terminal oracle actions at only `0.333333`.
  First-step dynamics were learnable (`0.75` immediate first-step accuracy),
  but immediate outcomes aligned with terminal oracle actions only `0.25`, and
  an upper-bound terminal model with actual first-step outcomes still reached
  only `0.333333`. Larger unverified previews did not rescue the path:
  24-label exact terminal support reached `0.416667`, while the 36-label preview
  dropped to `0.277778`. Option-mode decomposition exposed a tempting but
  collapsed signal: 24-label preview mode accuracy reached `0.625` by predicting
  `reposition` for `23/24` rows, and 36-label mode accuracy fell to `0.416667`.
  Reposition direction is partially learnable, but multi-move direction support
  stayed below floor (`0.5` on the 24-label preview, `0.4` on the 36-label
  preview). Do not train a runtime policy from this branch yet. The next
  worthwhile slice is a longer-horizon branch value model or sequence/options
  archive that predicts delayed value after repositioning, not a one-step
  compact dynamics policy.
- v77-v85: long-horizon branch trace boundary. Replay artifacts now serialize
  target-agent and population horizon traces out to `89` ticks, and the larger
  `24`-label branch archive is replay-verified and label-accepted with `+36`
  terminal alive and `+26` births versus logged. Actual future traces expose a
  delayed temporal-credit signal on the 12-label archive, but neither compact
  nor full policy-visible one-row observations can predict that signal under
  leave-one-source-seed-out support on the expanded archive. Stop before
  training; the next branch needs public target-agent sequence/history or
  option/archive state, not a pointwise horizon model.
- v86-v88: public history and scripted-option boundary. Branch points now carry
  deterministic same-agent public history traces, but history-aware pointwise
  support falls to `7/24` overall and `6/17` on material labels. Existing
  counterfactual continuation scripts also do not beat the baseline
  `carrion_then_water` branch continuation on the 12-point preview. Preserve
  the history data contract, but the next real lever must score multi-step
  archive/sequence continuations directly.
- v89: standardized progress ledger and branch-continuation archive scorer
  boundary. Every v-slice is now represented by a derived structured progress
  row, and "progress" is defined as strict policy pass or predeclared support
  floor pass. The compact archive scorer does not clear the floor (`0.25`
  all-label, `0.352941` material-only, floor `0.55`), so no runtime policy was
  trained from it.
- v90: mode-balanced target-local objective boundary. Balanced archive
  selection fixed the broad option-mode support signal (`0.766667`) and avoided
  dominant-mode collapse, but exact reposition direction on multi-move rows
  reached only `0.5` against the predeclared `0.55` floor. No runtime
  hierarchical option policy was trained.
- v91: failure-frontier reposition boundary. Enlarging the archive to `48`
  frontier labels lifted best reposition direction accuracy to `0.625`, but
  actual option-mode reposition support collapsed to `8` multi-move labels,
  option-mode accuracy fell to `0.583333`, and predicted target-local branch
  utility was negative (`-12.951134`). No runtime policy was trained.
- v92: catastrophe-sensitive branch utility boundary. Candidate-action
  utility/risk scoring over the same `48` labels and `263` action branches
  still fails target-local utility (`-14.142929` best mean delta), repeats the
  seed `41` target-death catastrophe, and collapses to `eat` with dominant
  predicted action share `0.729167`. No runtime policy was trained.
- v93: depleted-resource trap support boundary. Non-held-out trap support
  generation clears the support floor (`51` selected rows from `56` eligible)
  and avoids the named seed `41` tick `113` catastrophe, but strict frontier
  utility is still negative (`-20.003307` best mean target-local delta), seed
  `41` remains negative (`-149.153556`), one target-death regression remains,
  and mean births regress (`-0.020833`). No runtime policy was trained.
- v94: sequence-continuation scorer boundary. Replay-backed sequence targets
  from the v93 support archive produce the first positive strict target-local
  frontier result (`+8.359369`) while avoiding both seed `41` tick `113` and
  tick `114` target-death catastrophes with zero target-alive regressions, but
  the best scorer collapses to `eat` on `30/48` rows (`0.625`, cap `0.50`).
  No runtime policy was trained; the next line should be simulator-in-the-loop
  planning/search rather than more offline scoring on the same archive.
- v95: simulator-in-the-loop constrained planning boundary. Replay-backed
  global planning over the strict `48` frontier branches proves the action cap
  is compatible with utility: a minimum-cost repair reduces `eat` from `30/48`
  to `24/48` with zero utility cost, and the diversity-regularized planner
  reaches dominant action share `0.25`, target-alive negative count `0`, mean
  target-local score delta `+37.156839`, terminal alive delta `+0.791667`, and
  birth delta `+0.166667`. This is diagnostic-only; v96 may test whether these
  planner labels can be distilled into a runtime-feasible artifact without
  replay outcome access.

External checks that support this direction:

- Offline RL still needs support from the data distribution. IQL avoids direct
  evaluation of out-of-dataset actions, and CQL explicitly addresses
  overestimation under distribution shift, but neither removes the need for
  successful target-horizon behavior in the dataset:
  <https://arxiv.org/abs/2110.06169>,
  <https://arxiv.org/abs/2006.04779>
- Decision Transformer supports return-conditioned sequence modeling for
  offline control, but it is also downstream of trajectory quality. It becomes
  relevant once v38 has survivor continuations, not before:
  <https://arxiv.org/abs/2106.01345>
- Go-Explore is the closest match for the current failure mode: archive
  promising states, return to them, explore from them, then robustify with
  imitation or distillation. That maps directly to deterministic fixture branch
  replay:
  <https://arxiv.org/abs/1901.10995>
- MAP-Elites explicitly argues against returning only one winner and instead
  maps diverse high-performing elites across behavior dimensions:
  <https://arxiv.org/abs/1504.04909>
- OpenAI's Evolution Strategies work frames ES as a scalable black-box
  alternative for RL-style agent search and shows why simulator-heavy candidate
  evaluation can scale across many CPU workers:
  <https://arxiv.org/abs/1703.03864>,
  <https://openai.com/index/evolution-strategies/>
- MuZero, EfficientZero, and DreamerV3 support the longer-term world-model
  direction: learn or plan through predictive models for better sample
  efficiency and farsighted control. They should not displace exact simulator
  branching until v37-v39 prove the target data and policy contract:
  <https://arxiv.org/abs/1911.08265>,
  <https://arxiv.org/abs/2111.00210>,
  <https://arxiv.org/abs/2301.04104>
- DeepMind's XLand work emphasizes open-ended task generation, iterative
  improvement, and generalization without human interaction data. SIMA 2 and
  Genie 3 point in the same long-horizon direction for embodied agents and
  generated interactive worlds:
  <https://arxiv.org/abs/2107.12808>,
  <https://arxiv.org/abs/2512.04797>,
  <https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/>

Mind v3 deliberately avoids in-tick gradient updates for now. The repo's
deterministic replay contracts are valuable; v3 should earn autonomy first
through serialized deterministic artifacts, bounded inherited controller state,
horizon-labeled outcomes, and observable ecological selection.
