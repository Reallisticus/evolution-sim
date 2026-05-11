# Mind v3 Quality-Diversity Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Mind v3 search from a toy single-winner smoke into a resumable, heuristic-free population search that keeps diverse behavior elites and reports holdout performance.

**Architecture:** Keep the existing `MindV3EvolutionPolicy` and bounded inherited controller metadata. Upgrade only the search/evaluation layer: richer outcome metrics from trajectory records, a small quality-diversity elite archive, incremental checkpoint/resume state, and optional holdout seed evaluation for the best searched template.

**Tech Stack:** Python 3.14, simulator `SUMMARY_ONLY` runs, JSON reports, `unittest`, npm simulator entrypoints.

---

### Task 1: Add Search Report Tests

**Files:**
- Modify: `python/tests/test_mind_v1.py`

- [ ] **Step 1: Write a failing test for archive and holdout output**

Add a test beside the existing `test_mind_v3_evolve_cli_writes_generation_report_without_heuristics`:

```python
def test_mind_v3_evolve_cli_writes_archive_and_holdout_report(self) -> None:
    from evolution_sim.cli import mind_v3_evolve

    with TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "mind-v3-evolve.json"
        with (
            patch(
                "sys.argv",
                [
                    "mind_v3_evolve",
                    "--seeds",
                    "5",
                    "--holdout-seeds",
                    "13",
                    "--ticks",
                    "12",
                    "--population-size",
                    "3",
                    "--generations",
                    "1",
                    "--output",
                    str(report_path),
                ],
            ),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            mind_v3_evolve.main()

        report = json.loads(report_path.read_text(encoding="utf-8"))

    self.assertEqual(report["archive"]["policy"], "quality_diversity_archive_v1")
    self.assertIn("survival", report["archive"]["elites"])
    self.assertIn("resource_use", report["archive"]["elites"])
    self.assertIn("score_components", report["best_candidate"])
    self.assertIn("resource_event_rate", report["best_candidate"])
    self.assertEqual(
        report["holdout_evaluation"]["aggregate"]["heuristic_action_source_count"],
        0,
    )
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest python.tests.test_mind_v1.MindV1Tests.test_mind_v3_evolve_cli_writes_archive_and_holdout_report
```

Expected: fail because `--holdout-seeds` and archive fields do not exist.

- [ ] **Step 3: Write a failing test for checkpoint resume**

Add:

```python
def test_mind_v3_evolve_cli_resumes_from_checkpoint_report(self) -> None:
    from evolution_sim.cli import mind_v3_evolve

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_path = tmp_path / "checkpoint.json"
        resumed_path = tmp_path / "resumed.json"
        first_argv = [
            "mind_v3_evolve",
            "--seeds",
            "5",
            "--ticks",
            "10",
            "--population-size",
            "2",
            "--generations",
            "1",
            "--output",
            str(checkpoint_path),
        ]
        resumed_argv = [
            "mind_v3_evolve",
            "--resume-from",
            str(checkpoint_path),
            "--generations",
            "2",
            "--output",
            str(resumed_path),
        ]
        with (
            patch("sys.argv", first_argv),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            mind_v3_evolve.main()
        with (
            patch("sys.argv", resumed_argv),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
        ):
            mind_v3_evolve.main()

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        resumed = json.loads(resumed_path.read_text(encoding="utf-8"))

    self.assertEqual(len(checkpoint["generations"]), 1)
    self.assertEqual(len(resumed["generations"]), 2)
    self.assertEqual(
        resumed["generations"][0]["best_candidate_id"],
        checkpoint["generations"][0]["best_candidate_id"],
    )
    self.assertEqual(resumed["resume"]["completed_generations"], 2)
    self.assertEqual(resumed["search"]["resumed_from"], str(checkpoint_path))
```

- [ ] **Step 4: Run the resume test and verify it fails**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest python.tests.test_mind_v1.MindV1Tests.test_mind_v3_evolve_cli_resumes_from_checkpoint_report
```

Expected: fail because `--resume-from` does not exist.

### Task 2: Implement Resumable Quality-Diversity Search

**Files:**
- Modify: `python/evolution_sim/cli/mind_v3_evolve.py`

- [ ] **Step 1: Add CLI arguments**

Add `--holdout-seeds` and `--resume-from`. `--holdout-seeds` is optional. `--resume-from` loads a previous `mind_v3_evolution_search_v1` report and continues from its stored `resume.next_candidates`.

- [ ] **Step 2: Add richer run metrics**

Extend each candidate run with:

```python
"alive_agent_ticks": len(world.trajectory_records),
"alive_agent_ticks_per_tick": _round(len(world.trajectory_records) / max(1, ticks)),
"resource_event_count": resource_event_count,
"resource_event_rate": _round(resource_event_count / max(1, len(world.trajectory_records))),
"movement_event_count": movement_event_count,
"movement_event_rate": _round(movement_event_count / max(1, len(world.trajectory_records))),
"unique_requested_actions": unique_requested_actions,
```

Compute resource events only from outcomes (`feeding.ate`, `drinking.drank`), not hard-coded policy choices.

- [ ] **Step 3: Add score components**

Use a transparent score:

```python
{
    "alive_agents": alive_agents_mean,
    "births": 12.0 * births_mean,
    "deaths": -0.25 * deaths_mean,
    "alive_agent_ticks": 0.1 * alive_agent_ticks_per_tick_mean,
    "resource_events": 3.0 * resource_event_rate,
    "movement_events": 1.0 * movement_event_rate,
    "action_diversity": 0.2 * unique_requested_actions_mean,
}
```

The score remains outcome-based and heuristic-free.

- [ ] **Step 4: Add archive elites**

Maintain a small archive with these named elites:

```python
survival -> max score, alive mean, births mean
births -> max births mean, score
resource_use -> max resource event rate, alive mean
movement -> max movement event rate, alive mean
low_death -> min deaths mean, max alive mean
balanced -> max score
```

Use archive elites as parent templates for the next generation instead of only the generation winner.

- [ ] **Step 5: Add checkpoint state**

After each generation write the report with:

```python
"resume": {
    "completed_generations": len(generations),
    "next_generation_index": next_generation_index,
    "next_candidates": candidates,
    "rng_state": encoded_rng_state,
}
```

The report should be valid JSON after every completed generation.

- [ ] **Step 6: Add holdout evaluation**

When `--holdout-seeds` is provided, evaluate `best_candidate.controller_metadata` on those seeds after search and write:

```python
"holdout_evaluation": {
    "seeds": holdout_seeds,
    "runs": runs,
    "aggregate": aggregate,
}
```

### Task 3: Run a Real Bounded Search

**Files:**
- Output artifact: `output/mind/mind-v3-quality-diversity-search-smoke.json`
- Output artifact: `output/mind/mind-v3-quality-diversity-template-eval.json`

- [ ] **Step 1: Run the upgraded search**

Run:

```bash
npm run sim:mind:v3:evolve -- \
  --seeds 5,13,19 \
  --holdout-seeds 29,37 \
  --ticks 120 \
  --population-size 12 \
  --generations 4 \
  --output output/mind/mind-v3-quality-diversity-search-smoke.json
```

- [ ] **Step 2: Validate JSON and zero heuristic counts**

Run:

```bash
jq empty output/mind/mind-v3-quality-diversity-search-smoke.json
jq '{best: .best_candidate | {id: .candidate_id, score, alive_agents_mean, births_mean, heuristic_action_source_count}, holdout: .holdout_evaluation.aggregate, archive: .archive.elites | keys}' output/mind/mind-v3-quality-diversity-search-smoke.json
```

- [ ] **Step 3: Re-evaluate the best template**

Run:

```bash
npm run sim:mind:v3:evaluate -- \
  --seeds 5,13,19,29,37 \
  --ticks 120 \
  --founder-template output/mind/mind-v3-quality-diversity-search-smoke.json \
  --output output/mind/mind-v3-quality-diversity-template-eval.json
```

### Task 4: Document Evidence

**Files:**
- Modify: `docs/mind-v3-autonomous-evolution.md`
- Modify: `docs/mind-v2-audit-and-experiment-log.md`

- [ ] **Step 1: Add research context**

Record that the implementation is aligned with:

- OpenAI Evolution Strategies for scalable black-box agent search.
- MAP-Elites for preserving diverse high-performing behaviors.
- DeepMind XLand, Genie, and SIMA for diverse environments, held-out evaluation, and self-improvement from agent experience.

- [ ] **Step 2: Add measured results**

Record the exact best, archive, holdout, and evaluation metrics from the generated JSON reports. If the run fails to beat the prior smoke, document the failure directly.

### Task 5: Validate

- [ ] **Step 1: Syntax check**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m py_compile python/evolution_sim/cli/mind_v3_evolve.py
```

- [ ] **Step 2: Focused tests**

Run the new tests plus existing v3 evolve/evaluate tests:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
python.tests.test_mind_v1.MindV1Tests.test_mind_v3_evolve_cli_writes_generation_report_without_heuristics \
python.tests.test_mind_v1.MindV1Tests.test_mind_v3_evolve_cli_writes_archive_and_holdout_report \
python.tests.test_mind_v1.MindV1Tests.test_mind_v3_evolve_cli_resumes_from_checkpoint_report \
python.tests.test_mind_v1.MindV1Tests.test_mind_v3_evaluate_cli_accepts_evolved_founder_template
```

- [ ] **Step 3: Broader checks**

Run:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest python.tests.test_mind_v1
npm run sim:test
git diff --check
```
