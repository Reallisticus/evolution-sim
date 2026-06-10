# Evolution Sim

Minimal standalone workspace for the artificial-life simulator.

## Layout

- `docs/`: project blueprint and frozen MVP specs
- `python/evolution_sim/`: headless simulator package
- `python/tests/`: deterministic and replay tests

For a senior-developer codebase walkthrough, start with
[`docs/senior-developer-onboarding.md`](docs/senior-developer-onboarding.md).

## Run

```bash
cd evolution-sim
PYTHONPATH=python python3 -m evolution_sim.cli.run_headless --seed 7 --ticks 2000 --output output/sim-runs/seed7.json
```

## Inspect

```bash
cd evolution-sim
PYTHONPATH=python python3 -m evolution_sim.cli.inspect_run output/sim-runs/seed7.json
```

## Evaluate

Compare multiple seeds without writing replay payloads:

```bash
cd evolution-sim
npm run sim:evaluate -- --seeds 1,2,3,4,5 --ticks 120 --output output/evaluations/foundation-120.json
```

The evaluator defaults to `summary_only` mode so long seed sweeps stay focused on
shared survival, trophic, hazard, carrion, hydrology, and ecology outcomes. Use
`--mode full_replay` only for compact compatibility checks that need species
taxonomy fields.

## Trajectory Collection

Collect trainable trajectory rows without building full replay/viewer payloads:

```bash
cd evolution-sim
npm run sim:trajectory -- --seed 7 --ticks 400 --output output/trajectories/seed7.jsonl.gz
```

The trajectory stream is gzip JSONL when the output path ends in `.gz`. It writes
a header with the trajectory/observation/reward contract, one record per
decision, and a footer with summary and trajectory statistics.
Learned-policy trajectory collection is also available, but remains explicitly
opt-in and disabled by default:

```bash
npm run sim:trajectory -- \
  --seed 41 \
  --ticks 120 \
  --output output/trajectories/mind-v2-learned-seed41.jsonl.gz \
  --split-id mind-v2-learned-rollout \
  --mind-artifact output/mind/mind-v1-gate-value-deviation-artifact.json \
  --enable-mind
```

This is the first offline-to-online data path: collect experience from a learned
artifact in summary-only mode, retrain outside the simulation tick loop, then
promote only through held-out gates.

Train and evaluate the current disabled-by-default Mind v1 behavior-cloning
baseline from one or more trajectory streams:

```bash
cd evolution-sim
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --output output/mind/seed7-bc-artifact.json

npm run sim:mind:evaluate -- \
  --artifact output/mind/seed7-bc-artifact.json \
  --enable-mind \
  --seeds 8 \
  --ticks 120 \
  --output output/mind/seed7-bc-eval.json
```

Repeat `--trajectory` on `sim:mind:train` to build a seed-bank artifact. The
baseline is intentionally small and guarded by the existing observation
heuristic for survival/navigation conflicts; use evaluation gates as an honest
readiness report, not as a learned-controller quality claim.

The first neural policy artifact path is also available as an opt-in trainer:

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer neural-actor-critic-bc \
  --output output/mind/seed7-neural-artifact.json
```

This writes `guarded_neural_actor_critic_bc_v1`, a deterministic pure-Python
actor-critic MLP artifact. It is a schema/runtime milestone, not a promoted
replacement for the heuristic.

The real ML trainer is separate and opt-in. Install the Mind ML stack, then
train a PyTorch-backed artifact:

```bash
python3 -m pip install -r requirements-mind-ml.txt
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-actor-critic-bc \
  --output output/mind/seed7-torch-artifact.json
```

This writes `guarded_torch_actor_critic_bc_v1`. Training uses PyTorch/AdamW for
the actor and value heads, while runtime inference still uses the serialized
artifact weights through the deterministic simulator policy boundary. The torch
artifact also records train-set actor/value diagnostics, and the Mind gate
reports held-out neural calibration diagnostics for neural artifacts.

There is also an opt-in conservative offline-RL bridge trainer:

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-advantage-actor-critic-bc \
  --output output/mind/seed7-torch-advantage-artifact.json
```

This writes `guarded_torch_advantage_actor_critic_bc_v1`. It uses contextual
advantage-weighted actor loss blended back toward behavior cloning; it is an
experiment harness, not a promoted controller. Neural artifacts also carry
explicit score-margin and predicted-advantage thresholds for the narrow
calibrated local-`eat` acceptance path; this is gate-measured and remains
opt-in.

The first transition-based discrete offline-RL trainer is also available:

```bash
npm run sim:mind:train -- \
  --trajectory output/trajectories/seed7.jsonl.gz \
  --trainer torch-discrete-iql \
  --output output/mind/seed7-torch-iql-artifact.json
```

This writes `guarded_torch_discrete_iql_v1`. It trains Q and expectile V heads
from episode-aware trajectory transitions and extracts a masked
advantage-weighted actor. The current default-gate candidate uses
mask-renormalized neural actor scores, a `0.9` contextual-prior actor anchor,
and an IQL-only confidence-delegate margin of `0.25`. On the default held-out
gate it is the first neural/RL strict-control candidate: hard guard `0.1041`,
total fallback `0.4748`, safe deviation `0.0`, and zero per-seed alive/birth
deltas. The same candidate passes the extended strict matrix with hard guard
`0.1054`, total fallback `0.4715`, and zero per-seed alive/birth deltas. It is
not a final autonomous controller: heuristic delegation is still `0.3661`,
held-out Q/V calibration remains weak, and the artifact remains opt-in pending
promotion review. IQL artifacts currently forbid runtime value-supported
deviation metadata, so the learner must earn lower fallback through better
actor/critic calibration rather than broader heuristic bypasses.

Check the optional stack against PyPI with:

```bash
for package in torch torchrl tensordict gymnasium pettingzoo minari d3rlpy; do
  .venv/bin/python -m pip index versions "$package"
done
```

Validate the optional Mind ML stack in an environment where it is installed:

```bash
npm run sim:mind:torch:validate
```

On the configured NVIDIA trainer, require CUDA and write a dependency/device
report with:

```bash
npm run trainer -- run npm run sim:mind:torch:validate:cuda
```

This validation runs the torch-gated Mind tests and a tiny CUDA-backed training
smoke. It records dependency versions and device metadata; it is not promotion
evidence and does not replace the strict Mind gates.

Run the reproducible Mind v1 gate to collect the default multi-seed bank, train
the guarded baseline, and evaluate held-out seeds `5,13,19,29`:

```bash
cd evolution-sim
npm run sim:mind:gate
```

The default gate writes `output/mind/mind-v1-gate-report.json`, the trained
artifact at `output/mind/mind-v1-gate-artifact.json`, and training trajectories
under `output/trajectories/mind-v1-gate/`. The default train bank is
`1,2,3,4,6,7,8,9,10,11,12,17,23,31`. The report includes explicit readiness
criteria for terminal alive/birth regressions, invalid actions, guard
intervention share, and the separate confidence-delegation share. The current
Stage 3 checkpoint keeps aggregate guard intervention below `0.45`, every
role/mode below `0.50`, blocks per-seed alive or birth regressions beyond the
configured floor, and reports low-confidence learned disagreements as explicit
heuristic delegation rather than learned value.
Use `--fail-on-blockers` for CI-style failure on hard gate blockers, or
`--fail-on-review` when review warnings should also fail the command.
Use `--validation-ticks 120,240` for an opt-in longer-horizon validation matrix.
Use `npm run sim:mind:gate:extended` for the broader held-out seed matrix
(`5,13,19,29,37,41`) at 120 and 180 ticks.
See `docs/mind-v1-stage-plan.md` for the phase boundary before moving beyond
the heuristic safety floor, and `docs/mind-v2-online-neural-roadmap.md` for the
neural/offline-to-online plan.

Mind v3 is the current no-heuristic autonomous-controller track. Its durable
ledger and next direction are in `docs/mind-v3-autonomous-evolution.md`. The
project versioning policy is in `docs/versioning.md`; product SemVer is separate
from historical Mind experiment and artifact/schema identifiers. The
2026-06-10 full-repository audit is summarized in
`docs/repository-audit-2026-06-10-remediation.md`. It keeps v3 as the strategic
path, but changes the immediate work order:

- keep source/evidence durable before more experiments;
- do not spend another slice on scalar tuning, residual thresholds, actor bias,
  or tiny nearest-neighbor support probes on the same representation;
- continue from the v176 recommendation toward exact branch replay expansion,
  transition-row support, rollout-context policy capacity, or a distinct
  controller/data path;
- mint a fresh promotion-heldout seed matrix before making promotion-style
  claims from scorers trained or selected using artifacts that consumed
  `5,13,19,29,37,41` as support/provenance.

The bounded v31 pure-Python direct artifact and later residual/support-scorer
families are recorded as non-promotable diagnostics, not as active promotion
routes.

Generate the first label reports with:

```bash
npm run sim:mind:horizon-labels -- \
  --trajectory output/trajectories/mind-v3-probe.jsonl.gz \
  --output output/mind/mind-v3-horizon-labels.json

npm run sim:mind:fixture-labels -- \
  --report output/mind/mind-v3-visible-nav-v26-top8-80-120-search.json \
  --output output/mind/mind-v3-fixture-labels.json
```

Train and smoke-evaluate the first frozen deterministic v3 neural artifact with:

```bash
npm run sim:mind:v3:train-neural -- \
  --trajectory output/trajectories/mind-v3-probe.jsonl.gz \
  --horizon-labels output/mind/mind-v3-horizon-labels.json \
  --fixture-labels output/mind/mind-v3-fixture-labels.json \
  --output output/mind/mind-v3-neural-artifact.json

npm run sim:mind:v3:evaluate -- \
  --seeds 5,13 \
  --ticks 80 \
  --neural-artifact output/mind/mind-v3-neural-artifact.json \
  --compare-linear-baseline \
  --output output/mind/mind-v3-neural-80-eval.json
```

The current v3 neural runtime is a guarded residual diagnostic milestone, not
an independent promotion candidate. Frozen neural weights are evaluated through
`linear_controller_margin_guarded_neural_residual_v2`: the inherited linear
controller remains live and receives normal bounded reward updates, while the
neural residual is shadowed when the artifact expresses the known collapsed
`eat` mode or would override a nontrivial linear-anchor margin. The targeted
120-tick v26 drilldown improved neural-anchor broad survival from `16.0` to
`19.0` alive with births still `16.0`, but linear remained better at `23.5` /
`18.5` and carrion-only still failed with `0.0` terminal alive. See
`docs/mind-v3-autonomous-evolution.md` for the milestone boundary before
reducing the anchor or moving to torch/vectorized training. A follow-up
contextual fixture-bias artifact used the same policy-visible water/carrion
inputs and reproduced the same `19.0` / `16.0` broad result with the same
carrion blocker set, so the next useful step is controlled fixture trajectory
data or a stronger learner, not another hand-shaped residual bias.

`sim:mind:v3:evaluate --trajectory-output-dir ...` can now persist broad and
controlled-fixture runs as trajectory JSONL.gz files. Use this to turn
carrion-only fixture failures into `mind_horizon_labels_v1` training data
instead of relying only on aggregate fixture blocker labels.

`sim:mind:v3:carrion-autopsy` builds a deterministic post-carrion-contact
failure report from those trajectory files. The v31 autopsy shows the direct
artifact mostly dies through low-gain eat loops after contact, while the
linear/anchored paths die by post-contact movement energy depletion. The next
accepted learner slice needs counterfactual carrion-water recovery labels
before torch/IQL training.

## Foundation Gate

Run the local readiness gate before starting Mind work:

```bash
cd evolution-sim
npm run sim:gate:quick
```

The quick profile combines a short summary-only cross-seed sweep with a compact
full-replay species/taxonomy probe. The release profile is intentionally opt-in:

```bash
npm run sim:gate:release -- --output output/evaluations/foundation-release.json
```

Use the release profile at gate boundaries; it includes the long summary-only
viability sweep and the full replay speciation probe.

## Pre-Mind Readiness Plan

The current Foundation phase includes reproductive and signal cleanup before
Mind v1. The architecture plan is in
[`docs/pre-mind-reproductive-and-signal-readiness-plan.md`](docs/pre-mind-reproductive-and-signal-readiness-plan.md).

The plan keeps the policy surface numeric and opaque while preparing live
reproductive groups, staged sexed recombination, pheromone/signal fields,
trait-gated communication tokens, reserved mate/communication action slots, and
future bounded learned-state inheritance metadata. These are Foundation
contracts, not learned-controller work. Do not expose human-readable action
names, species labels, or signal meanings to the Mind.

Implemented Foundation cleanup pieces currently include the masked
`ActionContract` slots, reproductive signal contract/config scaffolding,
biology-gated reproductive readiness signal emission with deterministic
decay/diffusion, debug-only signal profile/provenance metadata in full replay,
reproductive genome mutation traits, Stage 0 reproductive-group summary/replay
registry, Stage 1 same-group facultative sexed reproduction, grouped genome
recombination helper contracts, and placeholder learned-state inheritance
metadata. Current Mind-facing replay contracts include reproductive expression
in `mind_observation_v3` and stable per-action signal outcome metadata in
`mind_action_outcome_v2`. Rare multi-offspring birth expansion is config-gated,
sexual-only, requires both parents to qualify, is bounded by parent
energy/population capacity/local destinations, emits clamp-reason metadata for
qualified attempts, and is disabled by default. The latest runtime hardening
also moved tick, trajectory, summary, and summary-finalizer work behind
dedicated runtime module boundaries, introduced a child-birth planning boundary
for reproduction, and made benchmark reports explicitly distinguish complete
runs from partial failure reports. The `mate` action remains reserved and
masked; Stage 1 is biology-gated, not policy-driven. Communication signal slots
remain opaque and disabled by default, with an opt-in trait-gated emission path
covered by tests; the default heuristic still never emits communication tokens.

## Test

```bash
cd evolution-sim
npm run sim:test
```

Prefer npm entrypoints because they set `PYTHONHASHSEED=0` and
`PYTHONPATH=python` consistently. Use raw `python3 -m unittest ...` only for
focused local probes, and include both environment variables when doing so.

## Viewer

Generate a small replay for the browser viewer:

```bash
cd evolution-sim
npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json
```

Serve the repo root:

```bash
cd evolution-sim
npm run viewer:serve
```

Open:

```text
http://127.0.0.1:4173/viewer/index.html?replay=../output/sim-runs/species-check.json
```

The viewer shows:

- terrain mix across plain, forest, wetland, rocky, and water, plus live agents colored by replay-adjudicated durable species
- current frame season, births, deaths, and live species count
- moving climate fronts and deterministic storm/drought state
- hydrology split into primary hard-water reasons and separate support counts, so shoreline, wetland support, and flooded support cannot be confused with the tile's main drinkable source
- climate-driven hazard layers for exposure and instability, with hazard counts, hazard overlays, and per-tile hazard levels
- carcass fields with visible carrion stock, freshness, mixed-source patching, deposition, consumption, and decay pressure
- trophic-role visibility so herbivore, omnivore, and carnivore occupancy can be inspected directly
- habitat-state dynamics such as bloom, flooded, and parched regions
- vegetation depletion, canopy shelter, and terrain-recovery pressure layered on top of habitat and climate, tracked only for land tiles
- active species leaderboard with current size, peak size, and lineage spread
- per-agent inspector with durable species, species status, transient ecotype, terrain context, water-access reason, adjacent-to-water support, wetland/flooded flags, refuge score, health, injury, trophic role, hazard state, carcass presence, hydration and energy modifiers, habitat state, ecology state, and genome traits relevant to niche pressure
- environmental overlays for fertility, moisture, heat, hydrology, shoreline, refuge, hazard, carcass, trophic role, habitat, and ecology that match the simulation state
- habitat overlay and habitat-pressure time-series chart
- separate hard-water and support/refuge charts so primary adjacent water, primary wetland, primary flooded, shoreline support, and canopy refuge stay semantically distinct
- separate hazard, carcass, and combat charts so damage pressure, carrion stock and flow, and predation activity can be attributed over time
- ecology overlay and vegetation-recovery time-series chart, with water kept visually separate from land ecology and habitat. The 2026-06-10 habitat-code audit finding was fixed as a dedicated replay/viewer contract change: water now serializes with the `-1` non-land sentinel instead of stable habitat.
- time-series charts for alive population, species count, births/deaths, and trait drift
- species ecology panels for terrain occupancy, shoreline support exposure, hard water access, refuge exposure, hazard exposure, trophic composition, stress, reproduction pressure, attack outcomes, and carcass use, with refuge averages labeled by denominator
- collapse and extinction event visibility tied to the replay timeline

Smoke-check the viewer:

```bash
cd evolution-sim
npm run viewer:validate
npm run viewer:smoke:malformed
npm run viewer:smoke
```

## Verification Flow

Run the regression and viewer checks for the current Foundation slice, including biotic pressure:

```bash
cd evolution-sim
npm run sim:test
npm run sim:gate:quick
npm run sim:run -- --seed 7 --ticks 300 --output output/sim-runs/species-check.json
npm run sim:inspect output/sim-runs/species-check.json
npm run viewer:validate
npm run viewer:smoke:malformed
REPLAY_PATH=../output/sim-runs/species-check.json npm run viewer:smoke
```

Manual viewer URL:

```text
http://127.0.0.1:4173/viewer/index.html?replay=../output/sim-runs/species-check.json
```
