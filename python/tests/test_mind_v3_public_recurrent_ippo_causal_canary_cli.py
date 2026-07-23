from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.cli import (
        mind_v3_public_recurrent_ippo_causal_canary as canary,
    )
    from evolution_sim.mind.provenance import stable_payload_digest
    from evolution_sim.mind.recurrent_actor_critic import (
        PublicRecurrentActorCritic,
        RecurrentActorCriticConfig,
    )
    from evolution_sim.mind.recurrent_policy import (
        PUBLIC_RECURRENT_ARGMAX_SELECTION,
        PUBLIC_RECURRENT_SAMPLED_SELECTION,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class MindV3PublicRecurrentIPPOCausalCanaryCLITests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)

    def test_requires_development_run_and_refuses_artifact_output(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--development-run"):
            canary.main([])
        with self.assertRaisesRegex(SystemExit, "artifact output is forbidden"):
            canary.main(
                [
                    "--development-run",
                    "--artifact",
                    "forbidden-model.json",
                ]
            )
        with self.assertRaisesRegex(SystemExit, "canonical .json"):
            canary.main(
                [
                    "--development-run",
                    "--report",
                    "not-a-report.pt",
                ]
            )

    def test_parser_resolves_full_sequence_and_research_controls(self) -> None:
        args = canary.build_parser().parse_args(
            [
                "--development-run",
                "--rollout-ticks",
                "17",
                "--tbptt-steps",
                "3",
                "--full-sequence-tbptt",
                "--world-balanced-loss",
                "--target-kl",
                "0.03",
                "--feed-forward-history-ablation",
                "--selection-seed-count",
                "4",
                "--selection-seed-offset",
                "2",
                "--encoder-size",
                "24",
                "--hidden-size",
                "32",
                "--recurrent-layers",
                "2",
                "--rollout-workers",
                "4",
                "--evaluation-workers",
                "3",
            ]
        )
        model, ppo, tbptt = canary._resolved_configs(args)
        plan, roles = canary._selection_seed_plan(
            count=args.selection_seed_count,
            offset=args.selection_seed_offset,
            learner_seed=args.learner_seed,
        )

        self.assertEqual(model.encoder_size, 24)
        self.assertEqual(model.hidden_size, 32)
        self.assertEqual(model.recurrent_layers, 2)
        self.assertEqual(ppo.tbptt_steps, 17)
        self.assertTrue(ppo.world_balanced_loss)
        self.assertEqual(ppo.target_kl, 0.03)
        self.assertTrue(ppo.feed_forward_history_ablation)
        self.assertEqual(args.rollout_workers, 4)
        self.assertEqual(args.evaluation_workers, 3)
        self.assertTrue(tbptt["full_sequence_invariant_satisfied"])
        self.assertEqual(len(plan.broad_seeds), 4)
        self.assertEqual(plan.broad_seeds, plan.fixture_seeds)
        self.assertEqual(roles["development_selection"]["offset"], 2)
        self.assertFalse(roles["validation"]["accessed"])
        self.assertFalse(roles["lockbox"]["accessed"])
        self.assertEqual(
            canary.ACTION_SELECTION_MODES,
            (
                PUBLIC_RECURRENT_ARGMAX_SELECTION,
                PUBLIC_RECURRENT_SAMPLED_SELECTION,
            ),
        )

        with self.assertRaisesRegex(SystemExit, "--rollout-workers"):
            canary.main(["--development-run", "--rollout-workers", "0"])
        with self.assertRaisesRegex(SystemExit, "--evaluation-workers"):
            canary.main(["--development-run", "--evaluation-workers", "65"])

    def test_counterfactual_auxiliary_defaults_are_explicit_and_dormant(self) -> None:
        args = canary.build_parser().parse_args(["--development-run"])

        active, preregistration = canary._resolved_counterfactual_config(args)

        self.assertIsNone(active)
        self.assertFalse(preregistration["enabled"])
        self.assertEqual(preregistration["mode"], "disabled")
        resolved = preregistration["resolved_configuration"]
        self.assertEqual(resolved["collection"]["horizons"], (8, 32, 64))
        self.assertEqual(
            resolved["auxiliary"]["scalarization"]["horizon_weights"],
            [
                {"horizon_ticks": 8, "weight": 0.25},
                {"horizon_ticks": 32, "weight": 0.5},
                {"horizon_ticks": 64, "weight": 0.25},
            ],
        )
        self.assertEqual(resolved["branch_tick_candidates"], [16, 32, 48, 64])
        self.assertEqual(resolved["bundles_per_update"], 2)
        self.assertEqual(resolved["workers"], 1)
        self.assertEqual(
            resolved["auxiliary"]["behavior_kl_coefficient"],
            1.0,
        )
        self.assertEqual(resolved["step"]["learning_rate_multiplier"], 0.1)
        self.assertEqual(
            resolved["step"]["post_step_public_policy_audit"],
            {
                "mean_forward_kl_pi_old_to_pi_post_limit": 0.002,
                "max_state_forward_kl_pi_old_to_pi_post_limit": 0.01,
            },
        )

    def test_exact_and_label_shuffled_counterfactual_modes_are_fail_closed(
        self,
    ) -> None:
        exact_args = canary.build_parser().parse_args(
            ["--development-run", "--counterfactual-exact-auxiliary"]
        )
        exact, exact_preregistration = canary._resolved_counterfactual_config(
            exact_args
        )
        self.assertIsNotNone(exact)
        self.assertTrue(exact_preregistration["enabled"])
        self.assertEqual(
            exact_preregistration["mode"],
            "exact_counterfactual_labels",
        )
        self.assertTrue(exact_preregistration["exact_branch_labels_consumed"])
        self.assertFalse(
            exact_preregistration["scientific_negative_control"]["enabled"]
        )

        shuffled_args = canary.build_parser().parse_args(
            [
                "--development-run",
                "--counterfactual-label-shuffled-control",
                "--counterfactual-permutation-seed",
                "991",
            ]
        )
        shuffled, shuffled_preregistration = canary._resolved_counterfactual_config(
            shuffled_args
        )
        self.assertIsNotNone(shuffled)
        assert shuffled is not None
        self.assertEqual(
            shuffled_preregistration["mode"],
            "deterministic_valid_action_label_shuffled_control",
        )
        self.assertFalse(shuffled_preregistration["exact_branch_labels_consumed"])
        self.assertEqual(
            shuffled.auxiliary.target_permutation_seed,
            991,
        )
        self.assertEqual(
            shuffled_preregistration["scientific_negative_control"][
                "permutation_domain"
            ],
            "currently_valid_actions_only",
        )

        invalid_argument_sets = (
            (
                ["--development-run", "--counterfactual-label-shuffled-control"],
                "permutation-seed",
            ),
            (
                [
                    "--development-run",
                    "--counterfactual-exact-auxiliary",
                    "--counterfactual-permutation-seed",
                    "991",
                ],
                "only valid",
            ),
            (
                [
                    "--development-run",
                    "--counterfactual-exact-auxiliary",
                    "--feed-forward-history-ablation",
                ],
                "feed-forward-history-ablation",
            ),
            (
                [
                    "--development-run",
                    "--counterfactual-exact-auxiliary",
                    "--worlds-per-update",
                    "1",
                    "--counterfactual-bundles-per-update",
                    "2",
                ],
                "cannot exceed",
            ),
        )
        for argv, message in invalid_argument_sets:
            with self.subTest(argv=argv):
                invalid = canary.build_parser().parse_args(argv)
                with self.assertRaisesRegex(SystemExit, message):
                    canary._validate_entrypoint_args(invalid)

    def test_model_state_hash_changes_with_real_parameter_change(self) -> None:
        model = PublicRecurrentActorCritic(
            RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
            initialization_seed=103,
        )
        before = canary._model_state_fingerprint(model)
        with torch.no_grad():
            next(model.parameters()).add_(0.125)
        after = canary._model_state_fingerprint(model)

        self.assertNotEqual(before["state_sha256"], after["state_sha256"])
        self.assertEqual(before["parameter_count"], after["parameter_count"])
        self.assertEqual(len(before["state_sha256"]), 64)

    def test_matched_evaluation_sampling_stream_excludes_model_arm(self) -> None:
        recurrent_args = canary.build_parser().parse_args(
            ["--development-run", "--candidate-sampling-seed-count", "4"]
        )
        feed_forward_args = canary.build_parser().parse_args(
            [
                "--development-run",
                "--candidate-sampling-seed-count",
                "4",
                "--feed-forward-history-ablation",
            ]
        )
        recurrent_plan, _ = canary._selection_seed_plan(
            count=recurrent_args.selection_seed_count,
            offset=recurrent_args.selection_seed_offset,
            learner_seed=recurrent_args.learner_seed,
        )
        feed_forward_plan, _ = canary._selection_seed_plan(
            count=feed_forward_args.selection_seed_count,
            offset=feed_forward_args.selection_seed_offset,
            learner_seed=feed_forward_args.learner_seed,
        )
        recurrent_stream = canary._evaluation_sampling_stream(
            seed_plan=recurrent_plan,
            fixtures=("carrion_only",),
            candidate_sampling_seed_count=4,
        )
        feed_forward_stream = canary._evaluation_sampling_stream(
            seed_plan=feed_forward_plan,
            fixtures=("carrion_only",),
            candidate_sampling_seed_count=4,
        )

        self.assertEqual(recurrent_stream, feed_forward_stream)
        self.assertTrue(recurrent_stream["arm_independent"])
        self.assertIn(
            "feed_forward_history_ablation",
            recurrent_stream["contract"]["excluded_fields"],
        )

    def test_source_manifest_is_deterministic(self) -> None:
        first = canary._source_file_hash_manifest()
        second = canary._source_file_hash_manifest()

        self.assertEqual(first, second)
        self.assertGreater(first["file_count"], 300)
        self.assertEqual(len(first["aggregate_sha256"]), 64)

    def test_paired_delta_keeps_controls_fixed_and_aligns_seed(self) -> None:
        before = _evaluation_report(candidate_alive=1, candidate_reward=2.0)
        after = _evaluation_report(candidate_alive=4, candidate_reward=5.5)

        result = canary._paired_causal_deltas(before, after)

        self.assertEqual(result["run_count"], 1)
        self.assertTrue(result["controls_exactly_stable_before_after"])
        self.assertEqual(result["runs"][0]["terminal_alive_delta"], 3)
        self.assertEqual(result["runs"][0]["reward_total_delta"], 3.5)

        drifted = _evaluation_report(candidate_alive=4, candidate_reward=5.5)
        drifted["broad"]["policies"]["masked_random"]["runs"][0]["births"] = 99
        with self.assertRaisesRegex(canary.CausalCanaryError, "control drifted"):
            canary._paired_causal_deltas(before, drifted)

        stream_drifted = _evaluation_report(
            candidate_alive=4,
            candidate_reward=5.5,
        )
        stream_drifted["evaluation_contract"]["candidate_sampling_seeds"] = [12]
        with self.assertRaisesRegex(
            canary.CausalCanaryError,
            "sampling contract drifted",
        ):
            canary._paired_causal_deltas(before, stream_drifted)

    def test_active_counterfactual_contract_is_preregistered_and_passed_to_runner(
        self,
    ) -> None:
        captured: dict[str, object] = {}

        class _Runner:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                self.model = PublicRecurrentActorCritic(
                    RecurrentActorCriticConfig(encoder_size=8, hidden_size=8),
                    initialization_seed=103,
                )

            def run(self, _schedule: object) -> object:
                with torch.no_grad():
                    next(self.model.parameters()).add_(0.125)
                return object()

        def _evaluation(
            *_args: object,
            **kwargs: object,
        ) -> tuple[dict[str, object], float]:
            payload = _evaluation_report(candidate_alive=1, candidate_reward=2.0)
            payload["evaluation_contract"]["candidate_action_selection"] = kwargs[
                "selection_mode"
            ]
            return payload, 0.0

        args = canary.build_parser().parse_args(
            [
                "--development-run",
                "--updates",
                "1",
                "--worlds-per-update",
                "2",
                "--rollout-ticks",
                "2",
                "--selection-seed-count",
                "1",
                "--encoder-size",
                "8",
                "--hidden-size",
                "8",
                "--device",
                "cpu",
                "--counterfactual-exact-auxiliary",
            ]
        )
        manifest = {
            "hash_algorithm": "sha256",
            "path_contract": "test",
            "file_count": 1,
            "files": {"test.py": "a" * 64},
            "aggregate_sha256": "b" * 64,
        }
        with (
            patch.object(canary, "RecurrentExperimentRunner", _Runner),
            patch.object(canary, "_timed_evaluation", side_effect=_evaluation),
            patch.object(
                canary,
                "_source_file_hash_manifest",
                return_value=manifest,
            ),
            patch.object(
                canary,
                "_git_source_state",
                return_value={
                    "git_head_observed": "c" * 40,
                    "git_state_available": True,
                    "source_tree_dirty": True,
                    "git_status_porcelain_sha256": "d" * 64,
                    "git_status_entry_count": 1,
                },
            ),
            patch.object(
                canary,
                "recurrent_training_run_payload",
                return_value={
                    "total_worlds": 2,
                    "total_transitions": 7,
                    "counterfactual_experiment": {"enabled": True},
                },
            ),
        ):
            report = canary.run_development_causal_canary(args)

        self.assertIsNotNone(captured["counterfactual_config"])
        preregistration = report["evaluation_preregistration"]
        self.assertEqual(
            preregistration["schema_version"],
            canary.CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION,
        )
        self.assertTrue(
            preregistration["run_config"]["counterfactual_auxiliary"]["enabled"]
        )
        self.assertEqual(report["policy"], canary.CAUSAL_CANARY_POLICY_VERSION)
        self.assertTrue(report["source"]["source_stable_during_run"])
        self.assertEqual(
            report["runtime_reproducibility"]["resolved_device"],
            "cpu",
        )
        self.assertTrue(all(value is False for value in report["lifecycle"].values()))

    def test_mocked_main_writes_one_canonical_closed_report(self) -> None:
        mocked_report = _closed_report()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "canary.json"
            with patch.object(
                canary,
                "run_development_causal_canary",
                return_value=mocked_report,
            ) as run:
                result = canary.main(
                    [
                        "--development-run",
                        "--updates",
                        "2",
                        "--worlds-per-update",
                        "3",
                        "--report",
                        str(report_path),
                    ]
                )
            written = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        run.assert_called_once()
        self.assertTrue(written["source_dirty"])
        self.assertTrue(written["source_unpinned"])
        self.assertTrue(written["noncandidate_development_canary"])
        self.assertTrue(all(value is False for value in written["lifecycle"].values()))
        observed = written.pop("exact_digest")
        self.assertEqual(observed, stable_payload_digest(written))


def _evaluation_report(
    *,
    candidate_alive: int,
    candidate_reward: float,
) -> dict[str, object]:
    candidate = _run(
        terminal_alive=candidate_alive,
        births=2,
        reward_total=candidate_reward,
    )
    linear = _run(terminal_alive=2, births=3, reward_total=3.0)
    random = _run(terminal_alive=1, births=1, reward_total=1.0)
    policies = {
        "public_recurrent": {"runs": [candidate], "aggregate": {}},
        "mind_v3_linear": {"runs": [linear], "aggregate": {}},
        "masked_random": {"runs": [random], "aggregate": {}},
    }
    return {
        "evaluation_contract": {
            "candidate_action_selection": PUBLIC_RECURRENT_SAMPLED_SELECTION,
            "candidate_sampling_stream_id": "matched-test-stream",
            "candidate_sampling_seeds": [11],
            "candidate_sampling_seed_count": 1,
        },
        "broad": {
            "policies": policies,
            "paired_deltas": {
                "candidate_minus_mind_v3_linear": {},
                "candidate_minus_masked_random": {},
            },
        },
        "fixtures": [],
    }


def _run(
    *,
    terminal_alive: int,
    births: int,
    reward_total: float,
) -> dict[str, object]:
    return {
        "context": "broad",
        "seed": 101,
        "terminal_alive": terminal_alive,
        "births": births,
        "deaths": 1,
        "reward_total": reward_total,
        "dominant_requested_action_share": 0.4,
        "unsupported_requested_action_count": 0,
        "heuristic_action_source_count": 0,
    }


def _closed_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": canary.CAUSAL_CANARY_SCHEMA_VERSION,
        "policy": canary.CAUSAL_CANARY_POLICY_VERSION,
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
            "source_pinned": False,
            "unpinned": True,
            "noncandidate": True,
            "source_stable_during_run": True,
            "source_file_hash_manifest": {"aggregate_sha256": "a" * 64},
        },
        "runtime_reproducibility": {
            "schema_version": canary.CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION,
            "captured_before_model_initialization": True,
            "torch_deterministic_algorithms_enabled_after_runner": True,
        },
        "evaluation_preregistration": {
            "schema_version": canary.CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION,
            "created_before_model_initialization": True,
            "source_manifest_aggregate_sha256": "a" * 64,
            "run_config": {
                "counterfactual_auxiliary": {
                    "enabled": False,
                    "configuration_resolved_before_model_initialization": True,
                    "runtime_artifact_created": False,
                    "runtime_action_selection_changed": False,
                    "promotion_authorized": False,
                }
            },
        },
        "training": {
            "total_worlds": 6,
            "total_transitions": 42,
            "counterfactual_experiment": None,
        },
        "lifecycle": {flag: False for flag in canary._LIFECYCLE_FLAGS},
        "campaign_training_slice_consumed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
    }
    preregistration = report["evaluation_preregistration"]
    preregistration_digest = stable_payload_digest(preregistration)
    preregistration["exact_digest"] = preregistration_digest
    preregistration["synthetic_noncandidate_digest_label"] = (
        canary.UNPINNED_NONCANDIDATE_DIGEST_PREFIX + preregistration_digest
    )
    return report


if __name__ == "__main__":
    unittest.main()
