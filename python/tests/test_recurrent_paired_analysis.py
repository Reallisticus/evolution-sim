from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_paired_analysis import (
    RecurrentPairedAnalysisError,
    _validate_learned_distribution_summary,
    analyze_recurrent_causal_canary_pair,
)


_LIFECYCLE_FLAGS = (
    "campaign_slice_requested",
    "campaign_training_slice_consumed",
    "campaign_candidate_registered",
    "campaign_candidate_selected",
    "training_artifact_created",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "runtime_integration_authorized",
    "runtime_integrated",
    "promotion_evidence_eligible",
    "promotion_authorized",
    "promotion_attempted",
    "promoted",
    "validation_seeds_accessed",
    "validation_slice_consumed",
    "validation_run",
    "validation_authorized",
    "lockbox_seeds_accessed",
    "lockbox_slice_consumed",
    "lockbox_opened",
    "lockbox_run",
    "lockbox_authorized",
)
_TRAINING_POLICY_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ippo|policy-action-sampling-v1|2026-07-21"
)
_EVALUATION_POLICY_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-evaluation|artifact-sampling-v1"
)


def _training_policy_seed(identity: str) -> int:
    digest = hashlib.sha256(
        f"{_TRAINING_POLICY_NAMESPACE}|{identity}".encode("utf-8")
    ).digest()
    lower = 2**31
    upper = 2**63 - 1
    return lower + (int.from_bytes(digest[:8], "big") % (upper - lower + 1))


def _evaluation_policy_seed(stream_id: str, index: int) -> int:
    material = (
        f"{_EVALUATION_POLICY_NAMESPACE}|{stream_id}|replicate-{index:04d}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (2**63 - 1)


def _seal(report: dict[str, object]) -> dict[str, object]:
    report.pop("exact_digest", None)
    report["exact_digest"] = stable_payload_digest(report)
    return report


def _run(
    *,
    context: str,
    seed: int,
    sampling_seed: int | None,
    alive: int,
    births: int,
    deaths: int,
    reward: float,
    eat: int,
    stay: int,
) -> dict[str, object]:
    count = eat + stay
    survivor = alive > 0
    return {
        "context": context,
        "seed": seed,
        "policy_sampling_seed": sampling_seed,
        "horizon_ticks": 120,
        "ticks_executed": 120 if survivor else 90,
        "terminal_alive": alive,
        "births": births,
        "deaths": deaths,
        "reward_total": reward,
        "trajectory_record_count": count,
        "policy_decision_record_count": count,
        "passive_trajectory_record_count": 0,
        "requested_action_counts": {"eat": eat, "stay": stay},
        "dominant_requested_action": "eat" if eat >= stay else "stay",
        "dominant_requested_action_count": max(eat, stay),
        "dominant_requested_action_share": max(eat, stay) / count,
        "unsupported_requested_action_count": 0,
        "heuristic_action_source_count": 0,
        "eat_requested_count": eat,
        "eat_without_positive_resource_gain_count": eat // 2,
        "eat_without_positive_resource_gain_share": (eat // 2) / eat,
        "replay_digest": stable_payload_digest(
            [context, seed, sampling_seed, alive, births, deaths, reward, eat, stay]
        ),
    }


def _candidate_runs(
    *,
    context: str,
    sampling_seeds: tuple[int | None, ...],
    arm: str,
    phase: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, sampling_seed in enumerate(sampling_seeds):
        if context == "fixture:carrion_only":
            before = (0, 1 + index, 3, -4.0 + index, 6, 4)
            if arm == "gru":
                after = (2 + index, 4 + index, 1, 3.0 + index, 4, 6)
            else:
                after = (0, 2 + index, 2, -1.0 + index, 7, 3)
        else:
            before = (1, 2 + index, 2, 1.0 + index, 5, 5)
            if arm == "gru":
                after = (3, 5 + index, 1, 6.0 + index, 4, 6)
            else:
                after = (2, 3 + index, 2, 3.0 + index, 6, 4)
        values = before if phase == "before" else after
        rows.append(
            _run(
                context=context,
                seed=101,
                sampling_seed=sampling_seed,
                alive=values[0],
                births=values[1],
                deaths=values[2],
                reward=values[3],
                eat=values[4],
                stay=values[5],
            )
        )
    return rows


def _control_run(context: str) -> dict[str, object]:
    return _run(
        context=context,
        seed=101,
        sampling_seed=None,
        alive=1,
        births=2,
        deaths=2,
        reward=0.5,
        eat=5,
        stay=5,
    )


def _evaluation(
    *,
    arm: str,
    phase: str,
    mode: str,
    shared_stream_id: str,
) -> dict[str, object]:
    sampling_seeds: tuple[int | None, ...] = (
        (None,)
        if mode == "deterministic_masked_argmax"
        else tuple(
            _evaluation_policy_seed(shared_stream_id, index) for index in range(2)
        )
    )
    contexts: list[tuple[str, str | None]] = [
        ("broad_default", None),
        ("fixture:carrion_only", "carrion_only"),
    ]
    replay_checks: list[dict[str, object]] = []
    built_contexts: dict[str, dict[str, object]] = {}
    for context, _fixture in contexts:
        candidate = _candidate_runs(
            context=context,
            sampling_seeds=sampling_seeds,
            arm=arm,
            phase=phase,
        )
        control = _control_run(context)
        for run in candidate:
            replay_checks.append(
                {
                    "context": context,
                    "seed": 101,
                    "policy_sampling_seed": run["policy_sampling_seed"],
                    "digest": run["replay_digest"],
                    "passed": True,
                }
            )
        context_payload = {
            "seeds": [101],
            "candidate_sampling_seeds": [
                seed for seed in sampling_seeds if seed is not None
            ],
            "candidate_run_count_per_environment": len(sampling_seeds),
            "controls_run_once_per_environment": True,
            "policies": {
                "public_recurrent": {"runs": candidate, "aggregate": {}},
                "mind_v3_linear": {"runs": [control], "aggregate": {}},
                "masked_random": {"runs": [copy.deepcopy(control)], "aggregate": {}},
            },
            "paired_deltas": {},
        }
        built_contexts[context] = context_payload
    return {
        "schema_version": "mind_public_recurrent_evaluation_v3",
        "artifact": None,
        "candidate_provenance": {
            "mode": "in_memory_unpinned_development_canary",
            "source_pinned": False,
            "noncandidate_development_canary": True,
            "promotion_evidence_eligible_from_provenance": False,
        },
        "evaluation_contract": {
            "ticks": 120,
            "candidate_action_selection": mode,
            "candidate_sampling_seeds": [
                seed for seed in sampling_seeds if seed is not None
            ],
            "candidate_sampling_seed_count": len(sampling_seeds),
            "candidate_sampling_stream_id": shared_stream_id,
            "candidate_replay_verification": "every_run_exact_digest_repeat",
            "strict_zero_unsupported_requested_actions": True,
            "strict_zero_heuristic_candidate_actions": True,
        },
        "seed_plan": {
            "source": "caller_supplied_holdout_with_training_exclusion",
            "digest": "d" * 64,
            "broad_seeds": [101],
            "fixture_seeds": [101],
            "excluded_training_seed_count": 2,
            "train_evaluation_overlap_count": 0,
        },
        "broad": built_contexts["broad_default"],
        "fixtures": [
            {
                "fixture": "carrion_only",
                **built_contexts["fixture:carrion_only"],
            }
        ],
        "carrion_fixture_terminal_survivor_count": sum(
            1
            for run in built_contexts["fixture:carrion_only"]["policies"][
                "public_recurrent"
            ]["runs"]
            if run["terminal_alive"] > 0
        ),
        "replay_verification": {
            "all_passed": True,
            "checked_run_count": len(replay_checks),
            "checks": replay_checks,
        },
    }


def _report(*, arm: str) -> dict[str, object]:
    feed_forward = arm == "feed_forward"
    task_identity = (
        "mind_public_recurrent_ippo_training_schedule_task_v1|"
        "update=0000|world=000000|scenario=carrion_only"
    )
    task = {
        "task_id": "update-0000-world-000000-carrion_only-seed-211",
        "scenario": "carrion_only",
        "environment_seed": 211,
        "rollout_ticks": 8,
        "policy_sampling_identity": task_identity,
        "policy_sampling_seed": _training_policy_seed(task_identity),
    }
    shared_stream_id = "matched-recurrent-evaluation:" + "e" * 64
    ppo = {
        "learner_seed": 404337389,
        "learning_rate": 0.0003,
        "feed_forward_history_ablation": feed_forward,
    }
    config = {
        "updates": 1,
        "worlds_per_update": 1,
        "rollout_ticks": 8,
        "training_scenarios": ["carrion_only"],
        "evaluation_fixtures": ["carrion_only"],
        "evaluation_horizon_ticks": 120,
        "candidate_sampling_seed_count": 2,
        "evaluation_sampling_stream": {
            "id": shared_stream_id,
            "arm_independent": True,
            "contract": {
                "schema_version": (
                    "mind_v3_recurrent_matched_evaluation_sampling_stream_v1"
                ),
                "seed_plan_digest": "d" * 64,
                "fixtures": ["carrion_only"],
                "candidate_sampling_seed_count": 2,
            },
        },
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "rollout_workers": 1,
        "model": {"encoder_size": 16, "hidden_size": 16, "recurrent_layers": 1},
        "ppo": ppo,
        "feed_forward_history_ablation": feed_forward,
        "training_schedule": [[task]],
        "training_schedule_sha256": stable_payload_digest([[task]]),
    }
    prereg = {
        "schema_version": "mind_v3_recurrent_causal_canary_preregistration_v2",
        "created_before_model_initialization": True,
        "action_selection_modes_in_order": [
            "deterministic_masked_argmax",
            "replay_deterministic_masked_sampling",
        ],
        "before_after_sampling_stream_identical": True,
        "candidate_controls": ["mind_v3_linear", "masked_random"],
        "run_config": copy.deepcopy(config),
        "seed_registry_sha256": "s" * 64,
        "seed_roles": {
            "learner": {"role": "learner_development", "seed": 404337389},
            "development_selection": {
                "role": "selection",
                "seeds": [101],
                "seed_plan_digest": "d" * 64,
            },
            "validation": {
                "role": "validation",
                "seeds_materialized_by_canary": False,
                "accessed": False,
            },
            "lockbox": {
                "role": "lockbox",
                "seeds_materialized_by_canary": False,
                "accessed": False,
            },
        },
        "source_manifest_aggregate_sha256": "m" * 64,
    }
    prereg_digest = stable_payload_digest(prereg)
    prereg["exact_digest"] = prereg_digest
    prereg["synthetic_noncandidate_digest_label"] = (
        "unpinned-noncandidate-development-canary:" + prereg_digest
    )
    manifest_files = {"python/evolution_sim/example.py": "a" * 64}
    manifest = {
        "hash_algorithm": "sha256",
        "path_contract": "repository_relative_sorted_runtime_python_plus_package",
        "file_count": 1,
        "files": manifest_files,
        "aggregate_sha256": stable_payload_digest(manifest_files),
    }
    # Match the preregistration source pin after its digest has been formed.
    prereg_base = copy.deepcopy(prereg)
    prereg_base.pop("exact_digest")
    prereg_base.pop("synthetic_noncandidate_digest_label")
    prereg_base["source_manifest_aggregate_sha256"] = manifest["aggregate_sha256"]
    prereg_digest = stable_payload_digest(prereg_base)
    prereg = prereg_base
    prereg["exact_digest"] = prereg_digest
    prereg["synthetic_noncandidate_digest_label"] = (
        "unpinned-noncandidate-development-canary:" + prereg_digest
    )
    evaluations = {phase: {} for phase in ("before", "after")}
    for phase in evaluations:
        for mode in (
            "deterministic_masked_argmax",
            "replay_deterministic_masked_sampling",
        ):
            evaluations[phase][mode] = _evaluation(
                arm=arm,
                phase=phase,
                mode=mode,
                shared_stream_id=shared_stream_id,
            )
    before_state = {
        "state_sha256": "b" * 64,
        "state_tensor_count": 12,
        "state_value_count": 100,
        "parameter_count": 100,
        "trainable_parameter_count": 0,
        "dtype_value_counts": {"torch.float32": 100},
        "hash_contract": "sorted_state_dict_name_dtype_shape_and_raw_cpu_bytes_v1",
    }
    report = {
        "schema_version": "mind_v3_public_recurrent_ippo_development_causal_canary_v3",
        "policy": "public_recurrent_ippo_in_memory_causal_canary_v3",
        "development_run": True,
        "development_experiment": True,
        "noncandidate_development_canary": True,
        "source_dirty": True,
        "source_unpinned": True,
        "source_pinned": False,
        "artifact_output_refused_by_contract": True,
        "artifact_output_requested": False,
        "artifact_path": None,
        "source": {
            "git_head_observed": "f" * 40,
            "git_state_available": True,
            "source_tree_dirty": True,
            "git_status_porcelain_sha256": "c" * 64,
            "git_status_entry_count": 2,
            "source_pinned": False,
            "source_commit_explicitly_pinned": False,
            "unpinned": True,
            "noncandidate": True,
            "source_file_hash_manifest": manifest,
            "source_stable_during_run": True,
        },
        "seed_registry_sha256": "s" * 64,
        "seed_roles": copy.deepcopy(prereg["seed_roles"]),
        "configuration": config,
        "evaluation_preregistration": prereg,
        "model_states": {
            "before": before_state,
            "after": {
                **before_state,
                "state_sha256": ("1" if feed_forward else "2") * 64,
            },
            "changed": True,
            "same_before_snapshot_used_for_both_selection_modes": True,
            "same_after_model_used_for_both_selection_modes": True,
        },
        "training": {
            "contract_version": "mind_public_recurrent_ippo_experiment_v2",
            "learner_seed": 404337389,
            "device": "cpu",
            "deterministic_algorithms_enabled": True,
            "model_config": copy.deepcopy(config["model"]),
            "ppo_config": copy.deepcopy(ppo),
            "rollout_execution": {"workers_requested": 1},
            "training_scenarios": ["carrion_only"],
            "updates": [{"update_index": 0, "tasks": [copy.deepcopy(task)]}],
            "total_worlds": 1,
            "total_transitions": 10,
        },
        "evaluations": evaluations,
        "paired_outcome_deltas": {},
        "lifecycle": {flag: False for flag in _LIFECYCLE_FLAGS},
        "campaign_training_slice_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
        "non_promoted": True,
    }
    return _seal(report)


class RecurrentPairedAnalysisTests(unittest.TestCase):
    def test_module_import_does_not_require_optional_torch_dependency(self) -> None:
        probe = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError("blocked optional torch", name='torch')
        return None

sys.meta_path.insert(0, BlockTorch())
import evolution_sim.mind.recurrent_paired_analysis
"""
        subprocess.run(
            [sys.executable, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )

    def setUp(self) -> None:
        self.gru = _report(arm="gru")
        self.feed_forward = _report(arm="feed_forward")

    def test_learned_distribution_count_uses_policy_decisions_not_passive_rows(
        self,
    ) -> None:
        probabilities = {action: 0.0 for action in ACTION_NAMES}
        probabilities["stay"] = 1.0
        distribution = {
            "decision_count": 9,
            "metric_means": {
                "entropy": 0.0,
                "normalized_entropy": 0.0,
                "selected_action_probability": 1.0,
                "top_action_probability": 1.0,
                "top_two_probability_margin": 1.0,
                "eat_probability": 0.0,
            },
            "mean_action_probabilities": probabilities,
        }

        _validate_learned_distribution_summary(
            distribution,
            policy_decision_count=9,
            label="candidate-with-one-passive-terminal-row",
        )

        with self.assertRaisesRegex(
            RecurrentPairedAnalysisError,
            "policy decisions",
        ):
            _validate_learned_distribution_summary(
                distribution,
                policy_decision_count=10,
                label="candidate-with-one-passive-terminal-row",
            )

    def test_computes_matched_stream_level_difference_in_differences(self) -> None:
        analysis = analyze_recurrent_causal_canary_pair(
            self.gru,
            self.feed_forward,
        )

        self.assertEqual(
            analysis["schema_version"],
            "mind_public_recurrent_gru_feed_forward_paired_analysis_v1",
        )
        sampled = analysis["difference_in_differences"]["selection_modes"][
            "replay_deterministic_masked_sampling"
        ]
        self.assertEqual(sampled["matched_run_count"], 4)
        carrion_rows = [
            row for row in sampled["rows"] if row["context"] == "fixture:carrion_only"
        ]
        self.assertEqual(len(carrion_rows), 2)
        self.assertEqual(carrion_rows[0]["metrics"]["terminal_survival"], 1.0)
        self.assertEqual(
            {row["metrics"]["terminal_alive"] for row in carrion_rows},
            {2.0, 3.0},
        )
        self.assertEqual(carrion_rows[0]["metrics"]["births"], 2.0)
        self.assertEqual(carrion_rows[0]["metrics"]["reward_total"], 4.0)
        self.assertEqual(
            set(carrion_rows[0]["requested_action_share"]),
            set(ACTION_NAMES),
        )
        learner = analysis["difference_in_differences"]["learner_level_carrion_effect"]
        self.assertEqual(learner["learner_seed"], 404337389)
        self.assertEqual(learner["learner_replication_count_per_arm"], 1)
        self.assertFalse(learner["sampling_streams_are_learner_replications"])
        self.assertFalse(learner["inferential_claim_authorized"])
        self.assertEqual(
            learner["selection_modes"]["replay_deterministic_masked_sampling"][
                "environment_count"
            ],
            1,
        )
        self.assertEqual(
            learner["selection_modes"]["replay_deterministic_masked_sampling"][
                "policy_sampling_stream_count"
            ],
            2,
        )

    def test_accepts_canonical_json_paths_and_verifies_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gru_path = root / "gru.json"
            ff_path = root / "ff.json"
            gru_path.write_text(json.dumps(self.gru), encoding="utf-8")
            ff_path.write_text(json.dumps(self.feed_forward), encoding="utf-8")

            result = analyze_recurrent_causal_canary_pair(gru_path, ff_path)

        self.assertEqual(
            result["evidence"]["gru_report_exact_digest"],
            self.gru["exact_digest"],
        )

    def test_rejects_exact_digest_tampering(self) -> None:
        self.gru["configuration"]["rollout_ticks"] = 9
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "exact digest"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_non_ablation_configuration_drift(self) -> None:
        self.feed_forward["configuration"]["ppo"]["learning_rate"] = 0.001
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "configuration"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_same_ablation_value_in_both_arms(self) -> None:
        self.feed_forward["configuration"]["feed_forward_history_ablation"] = False
        self.feed_forward["configuration"]["ppo"]["feed_forward_history_ablation"] = (
            False
        )
        self.feed_forward["training"]["ppo_config"]["feed_forward_history_ablation"] = (
            False
        )
        self.feed_forward["evaluation_preregistration"]["run_config"][
            "feed_forward_history_ablation"
        ] = False
        self.feed_forward["evaluation_preregistration"]["run_config"]["ppo"][
            "feed_forward_history_ablation"
        ] = False
        prereg = self.feed_forward["evaluation_preregistration"]
        prereg.pop("exact_digest")
        prereg.pop("synthetic_noncandidate_digest_label")
        digest = stable_payload_digest(prereg)
        prereg["exact_digest"] = digest
        prereg["synthetic_noncandidate_digest_label"] = (
            "unpinned-noncandidate-development-canary:" + digest
        )
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "ablation"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_source_manifest_or_head_drift(self) -> None:
        self.feed_forward["source"]["git_head_observed"] = "0" * 40
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "source"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_before_model_fingerprint_drift(self) -> None:
        self.feed_forward["model_states"]["before"]["state_sha256"] = "0" * 64
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "before model"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_training_policy_sampling_seed_drift(self) -> None:
        task = self.feed_forward["configuration"]["training_schedule"][0][0]
        task["policy_sampling_seed"] += 1
        self.feed_forward["configuration"]["training_schedule_sha256"] = (
            stable_payload_digest(
                self.feed_forward["configuration"]["training_schedule"]
            )
        )
        self.feed_forward["training"]["updates"][0]["tasks"][0][
            "policy_sampling_seed"
        ] += 1
        prereg_config = self.feed_forward["evaluation_preregistration"]["run_config"]
        prereg_task = prereg_config["training_schedule"][0][0]
        prereg_task["policy_sampling_seed"] += 1
        prereg_config["training_schedule_sha256"] = stable_payload_digest(
            prereg_config["training_schedule"]
        )
        prereg = self.feed_forward["evaluation_preregistration"]
        prereg.pop("exact_digest")
        prereg.pop("synthetic_noncandidate_digest_label")
        digest = stable_payload_digest(prereg)
        prereg["exact_digest"] = digest
        prereg["synthetic_noncandidate_digest_label"] = (
            "unpinned-noncandidate-development-canary:" + digest
        )
        _seal(self.feed_forward)
        with self.assertRaisesRegex(
            RecurrentPairedAnalysisError, "policy sampling seed"
        ):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_evaluation_sampling_stream_drift(self) -> None:
        evaluation = self.feed_forward["evaluations"]["after"][
            "replay_deterministic_masked_sampling"
        ]
        evaluation["evaluation_contract"]["candidate_sampling_seeds"][1] += 1
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "sampling"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_open_lifecycle_or_seed_access(self) -> None:
        self.feed_forward["lifecycle"]["validation_run"] = True
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "lifecycle"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_replay_failure_or_run_check_mismatch(self) -> None:
        evaluation = self.feed_forward["evaluations"]["after"][
            "replay_deterministic_masked_sampling"
        ]
        evaluation["replay_verification"]["checks"][0]["passed"] = False
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "replay"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)

    def test_rejects_controls_that_drift_before_after(self) -> None:
        run = self.feed_forward["evaluations"]["after"][
            "replay_deterministic_masked_sampling"
        ]["broad"]["policies"]["mind_v3_linear"]["runs"][0]
        run["reward_total"] = 99.0
        _seal(self.feed_forward)
        with self.assertRaisesRegex(RecurrentPairedAnalysisError, "control"):
            analyze_recurrent_causal_canary_pair(self.gru, self.feed_forward)


if __name__ == "__main__":
    unittest.main()
