from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evolution_sim.config import (
    CombatConfig,
    DietMatchingConfig,
    ReproductionConfig,
    WorldConfig,
)
from evolution_sim.env.runtime.state import RunMode
from evolution_sim.env.world import SimulationWorld
import evolution_sim.mind.open_ecology_phase_a as phase_a
import evolution_sim.mind.open_ecology_selection as selection
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
)
from evolution_sim.mind.recurrent_actor_critic import (
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    RecurrentActorCriticConfig,
    PublicRecurrentActorCritic,
)
from evolution_sim.mind.recurrent_artifact import save_recurrent_artifact
from evolution_sim.mind.recurrent_experiment import OpenEcologyBroadWorldTreatment


REQUIRES_MIND_ML = True

_SOURCE_COMMIT = "a" * 40
_SOURCE_MANIFEST = "b" * 64
_CAMPAIGN_DIGEST = "c" * 64


def _model(cell_id: str = "A0") -> PublicRecurrentActorCritic:
    signals = OpenEcologyBroadWorldTreatment(
        initial_agents=64
    ).signals.as_signal_config()
    config = RecurrentActorCriticConfig.for_signal_config(
        signals,
        encoder_size=256,
        hidden_size=256,
        recurrent_layers=1,
        genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        critic_genome_conditioning=selection._PHASE_A_CELL_MODEL_CONTRACTS[cell_id][0],
        value_shared_trunk_gradient=selection._PHASE_A_CELL_MODEL_CONTRACTS[cell_id][1],
    )
    return PublicRecurrentActorCritic(config, initialization_seed=17).to(
        device="cpu",
        dtype=torch.float32,
    )


def _write_run_contract(
    path: Path,
    *,
    model: PublicRecurrentActorCritic,
    cell_id: str = "A0",
    learner_index: int = 0,
) -> str:
    config = model.config
    contract: dict[str, object] = {
        "schema_version": "test_open_ecology_phase_a_run_contract_v1",
        "campaign_digest": _CAMPAIGN_DIGEST,
        "source": {
            "commit": _SOURCE_COMMIT,
            "manifest_sha256": _SOURCE_MANIFEST,
        },
        "cell_id": cell_id,
        "learner_index": learner_index,
        "learner_seed": OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
            learner_index
        ],
        "model": {
            "encoder_size": config.encoder_size,
            "hidden_size": config.hidden_size,
            "recurrent_layers": config.recurrent_layers,
            "public_input_schema_version": config.public_input_schema_version,
            "public_input_size": config.public_input_size,
            "genome_conditioning_mode": config.genome_conditioning_mode,
            "critic_genome_conditioning": config.critic_genome_conditioning,
            "value_shared_trunk_gradient": config.value_shared_trunk_gradient,
        },
        "initial_full_model_sha256": selection.recurrent_model_state_sha256(
            selection.PublicRecurrentActorCritic(
                config,
                initialization_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
                    learner_index
                ],
            )
        ),
        "terminal_policy": {
            "selection_evaluation_pending": True,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
        },
    }
    contract["exact_digest"] = selection.stable_payload_digest(contract)
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(contract["exact_digest"])


def _resign(payload: dict[str, object], field: str) -> None:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    payload[field] = selection.stable_payload_digest(unsigned)


def _training_authority(
    *,
    artifact_path: Path,
    run_contract_path: Path,
    artifact_sha256: str,
    model_state_sha256: str,
    run_contract_digest: str,
    cell_id: str = "A0",
    learner_index: int = 0,
) -> dict[str, object]:
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    authority: dict[str, object] = {
        "schema_version": (
            selection.OPEN_ECOLOGY_TERMINAL_SELECTION_AUTHORITY_SCHEMA_VERSION
        ),
        "campaign_digest": _CAMPAIGN_DIGEST,
        "source_commit": _SOURCE_COMMIT,
        "source_manifest_sha256": _SOURCE_MANIFEST,
        "run_id": f"test-{cell_id}-{learner_index}",
        "cell_id": cell_id,
        "learner_index": learner_index,
        "learner_seed": OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
            learner_index
        ],
        "terminal_logical_name": "terminal.json",
        "terminal_exact_digest": "1" * 64,
        "terminal_file_sha256": "2" * 64,
        "final_prefix_commit_exact_digest": "3" * 64,
        "terminal_checkpoint_model_state_sha256": model_state_sha256,
        "initial_model_state_sha256": run_contract["initial_full_model_sha256"],
        "initial_to_terminal_model_changed": True,
        "cumulative_accepted_ppo_minibatches": 8,
        "cumulative_post_step_kl_rejected_steps": 1,
        "run_contract_logical_name": "run-contract.json",
        "run_contract_exact_digest": run_contract_digest,
        "run_contract_file_sha256": selection._file_sha256(run_contract_path),
        "artifact_logical_name": "terminal-training-artifact.json",
        "artifact_sha256": artifact_sha256,
        "artifact_file_sha256": selection._file_sha256(artifact_path),
        "selection_binding_required": True,
    }
    authority["exact_digest"] = selection.stable_payload_digest(authority)
    return authority


class OpenEcologySelectionTests(unittest.TestCase):
    def test_authoritative_source_binding_checks_commit_tree_and_manifest(self) -> None:
        root = Path(selection.__file__).resolve().parents[3]
        request = SimpleNamespace(
            source_repository_root=root,
            expected_source_commit=_SOURCE_COMMIT,
            expected_source_manifest_sha256=_SOURCE_MANIFEST,
        )
        with (
            patch.object(
                selection,
                "discover_pinned_git_executable",
                return_value=object(),
            ),
            patch.object(
                selection,
                "run_pinned_git",
                side_effect=(
                    _SOURCE_COMMIT,
                    "",
                    _SOURCE_COMMIT,
                    "",
                ),
            ) as run,
            patch.object(
                selection,
                "source_file_hash_manifest",
                return_value={"aggregate_sha256": _SOURCE_MANIFEST},
            ),
        ):
            selection._require_live_selection_source(request)
        self.assertEqual(run.call_count, 4)

    def test_authoritative_source_binding_rejects_dirty_spawn_source(self) -> None:
        root = Path(selection.__file__).resolve().parents[3]
        request = SimpleNamespace(
            source_repository_root=root,
            expected_source_commit=_SOURCE_COMMIT,
            expected_source_manifest_sha256=_SOURCE_MANIFEST,
        )
        with (
            patch.object(
                selection,
                "discover_pinned_git_executable",
                return_value=object(),
            ),
            patch.object(
                selection,
                "run_pinned_git",
                side_effect=(
                    _SOURCE_COMMIT,
                    " M python/evolution_sim/mind/open_ecology_selection.py\n",
                ),
            ),
            self.assertRaisesRegex(
                selection.OpenEcologySelectionError,
                "exact clean source",
            ),
        ):
            selection._require_live_selection_source(request)

    def test_phase_plans_use_disjoint_selection_roles_and_fixed_densities(
        self,
    ) -> None:
        phase_a = selection.OpenEcologySelectionPlan.for_phase("phase_a")
        phase_b = selection.OpenEcologySelectionPlan.for_phase("phase_b")

        self.assertEqual(phase_a.environment_seed_indices, tuple(range(32)))
        self.assertEqual(phase_b.environment_seed_indices, tuple(range(32, 64)))
        self.assertEqual(phase_a.ticks_per_execution, 512)
        self.assertEqual(phase_b.ticks_per_execution, 2_000)
        self.assertEqual(phase_a.density_cycle, (64,))
        self.assertEqual(phase_b.density_cycle, (32, 64, 128, 64))
        self.assertEqual(
            phase_a.environment_seeds,
            tuple(
                OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"][index]
                for index in range(32)
            ),
        )
        self.assertTrue(
            set(phase_a.environment_seeds).isdisjoint(phase_b.environment_seeds)
        )

    def test_fixed_policy_tapes_are_common_across_cells_learners_and_artifacts(
        self,
    ) -> None:
        base = selection._EnvironmentTask(
            phase="phase_a",
            cell_id="A0",
            learner_index=0,
            artifact_sha256="a" * 64,
            initialized_baseline_model_state_sha256="b" * 64,
            local_environment_index=0,
            environment_seed_index=0,
            environment_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"][0],
            ticks=512,
            initial_agents=64,
            stochastic_tape_identities=(
                "phase-a-selection-tape-00",
                "phase-a-selection-tape-01",
                "phase-a-selection-tape-02",
                "phase-a-selection-tape-03",
            ),
            genome_stream_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
                0
            ],
        )
        comparison = replace(
            base,
            cell_id="A3",
            learner_index=3,
            artifact_sha256="f" * 64,
            genome_stream_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
                3
            ],
        )
        base_seed = selection._policy_sampling_seed(
            base,
            tape_identity="phase-a-selection-tape-02",
        )

        self.assertEqual(
            base_seed,
            selection._policy_sampling_seed(
                comparison,
                tape_identity="phase-a-selection-tape-02",
            ),
        )
        self.assertNotEqual(
            base_seed,
            selection._policy_sampling_seed(
                replace(base, phase="phase_b"),
                tape_identity="phase-a-selection-tape-02",
            ),
        )

    def test_global_action_collapse_requires_three_lineage_wide_windows(
        self,
    ) -> None:
        collapsed = selection._action_collapse_evidence(
            [("eat", 1 + (decision % 10)) for decision in range(3_000)]
        )
        diverse = selection._action_collapse_evidence(
            [
                (
                    selection.ACTION_NAMES[decision % len(selection.ACTION_NAMES)],
                    1 + (decision % 10),
                )
                for decision in range(3_000)
            ]
        )

        self.assertIs(collapsed["collapsed"], True)
        self.assertIs(collapsed["evidence_sufficient"], True)
        self.assertIs(diverse["collapsed"], False)
        self.assertIs(diverse["evidence_sufficient"], True)

    def test_global_action_collapse_is_offset_robust_and_requires_3000_decisions(
        self,
    ) -> None:
        insufficient = selection._action_collapse_evidence(
            [("eat", 1 + (decision % 10)) for decision in range(2_999)]
        )
        offset_collapse = selection._action_collapse_evidence(
            [
                (
                    selection.ACTION_NAMES[decision % len(selection.ACTION_NAMES)],
                    1 + (decision % 10),
                )
                for decision in range(500)
            ]
            + [("eat", 1 + (decision % 10)) for decision in range(3_000)]
            + [
                (
                    selection.ACTION_NAMES[decision % len(selection.ACTION_NAMES)],
                    1 + (decision % 10),
                )
                for decision in range(500)
            ]
        )

        self.assertIs(insufficient["evidence_sufficient"], False)
        self.assertIs(insufficient["collapsed"], False)
        self.assertIs(offset_collapse["evidence_sufficient"], True)
        self.assertIs(offset_collapse["collapsed"], True)
        self.assertLessEqual(offset_collapse["first_collapsed_span_start"], 500)

    def test_selection_metrics_bootstrap_only_alive_agents_at_fixed_horizon(
        self,
    ) -> None:
        _normalized, value_rmse, advantage_variance = selection._selection_metrics(
            rewards_by_agent={1: [0.0], 2: [0.0]},
            values_by_agent={1: [0.0], 2: [0.0]},
            terminal_bootstrap_values_by_agent={1: 2.0},
        )
        alive_target = selection.OPEN_ECOLOGY_SELECTION_DISCOUNT * 2.0
        self.assertAlmostEqual(value_rmse, alive_target / (2.0**0.5), places=12)
        self.assertAlmostEqual(
            advantage_variance,
            (alive_target**2) / 4.0,
            places=12,
        )

    def test_selection_bootstrap_keeps_horizon_newborn_as_evidence_only(
        self,
    ) -> None:
        model = _model()
        policy = selection.DeterministicPublicRecurrentPolicy(
            model,
            artifact_digest="a" * 64,
            copy_to_cpu=False,
            reset_recurrent_state_each_decision=False,
            sampling_seed=None,
        )
        policy.start_world(
            world_identity="selection-horizon-newborn",
            genome_stream_seed=61,
            genome_population_mode=(selection.RecurrentGenomePopulationMode.HERITABLE),
        )
        treatment = OpenEcologyBroadWorldTreatment(initial_agents=32)
        world = SimulationWorld(
            WorldConfig(
                seed=59,
                max_ticks=1,
                width=5,
                height=5,
                initial_agents=1,
                max_agents=20,
                water_tile_ratio=0.0,
                forest_tile_ratio=0.0,
                wetland_tile_ratio=0.0,
                rocky_tile_ratio=0.0,
                base_energy_drain=0.0,
                base_hydration_drain=0.0,
                signals=treatment.signals.as_signal_config(),
                reproduction=ReproductionConfig(
                    min_age=1,
                    cooldown_ticks=1_000,
                    min_hydration_fraction=0.0,
                    energy_cost=0.0,
                    child_energy_fraction=0.3,
                ),
                diet_matching=DietMatchingConfig(
                    specialist_threshold=0.0,
                    omnivore_threshold=0.0,
                ),
                combat=CombatConfig(
                    min_attack_health_ratio=1.0,
                    min_attack_energy_ratio=1.0,
                    min_attack_hydration_ratio=1.0,
                    base_attack_damage=0.0,
                    attack_energy_cost=0.0,
                    attack_hydration_cost=0.0,
                ),
            ),
            policy=policy,
        )
        founder = world.alive_agents()[0]
        founder.age = 10
        founder.energy = founder.genome.max_energy * 1.25
        founder.hydration = founder.genome.max_hydration
        founder.health = founder.max_health
        founder.last_reproduction_tick = -10_000

        result = world.run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
        children = [
            agent
            for agent in world.alive_agents()
            if agent.parent_id == founder.agent_id
        ]
        self.assertEqual(len(children), 1)
        child = children[0]
        eligible_agent_ids = {
            int(record["agent_id"])
            for record in world.trajectory_records
            if record.get("action_source") != "passive"
        }
        authority_before = selection._selection_boundary_authority_state(
            world=world,
            policy=policy,
        )
        bootstrap = selection._terminal_alive_agent_bootstrap_values(
            world=world,
            policy=policy,
            tick=1,
            target_eligible_agent_ids=eligible_agent_ids,
        )

        self.assertEqual(
            selection._selection_boundary_authority_state(
                world=world,
                policy=policy,
            ),
            authority_before,
        )
        self.assertEqual(bootstrap["alive_agent_count"], 2)
        self.assertEqual(bootstrap["target_eligible_agent_count"], 1)
        self.assertEqual(bootstrap["zero_decision_alive_agent_count"], 1)
        self.assertEqual(
            bootstrap["zero_decision_alive_agent_ids"],
            [child.agent_id],
        )
        self.assertEqual(
            {
                row["agent_id"]
                for row in bootstrap["values"]
                if row["target_eligible"] is True
            },
            {founder.agent_id},
        )
        selection._validate_terminal_bootstrap_evidence(
            bootstrap,
            terminal_summary=result.summary,
            ticks_per_execution=1,
        )
        policy.reconcile_live_agent_ids(
            live_agent_ids=tuple(
                sorted(agent.agent_id for agent in world.alive_agents())
            )
        )
        policy.reset_world()

    def test_initialized_baseline_requires_positive_paired_median_improvement(
        self,
    ) -> None:
        positive = selection._paired_normalized_return_evidence(
            trained_normalized_returns=(0.5, 0.1, 0.2, 0.4),
            baseline_normalized_returns=(0.1, 0.0, 0.1, 0.1),
        )
        tied = selection._paired_normalized_return_evidence(
            trained_normalized_returns=(0.1, 0.2, 0.3, 0.4),
            baseline_normalized_returns=(0.1, 0.2, 0.3, 0.4),
        )

        self.assertGreater(positive["paired_median_normalized_return_improvement"], 0)
        self.assertIs(positive["positive_paired_median_improvement"], True)
        self.assertEqual(tied["paired_median_normalized_return_improvement"], 0.0)
        self.assertIs(tied["positive_paired_median_improvement"], False)

    def test_initialized_baseline_equal_weights_environment_pair_medians(
        self,
    ) -> None:
        environment_deltas = [(-100.0, -100.0, 1.0, 1.0) for _ in range(17)] + [
            (100.0, 100.0, 100.0, 100.0) for _ in range(15)
        ]
        flattened = selection._paired_normalized_return_evidence(
            trained_normalized_returns=tuple(
                value for environment in environment_deltas for value in environment
            ),
            baseline_normalized_returns=(0.0,) * (32 * 4),
        )
        hierarchical = (
            selection._aggregate_environment_paired_normalized_return_evidence(
                environment_paired_deltas=environment_deltas,
            )
        )

        self.assertIs(flattened["positive_paired_median_improvement"], True)
        self.assertEqual(hierarchical["environment_count"], 32)
        self.assertEqual(hierarchical["tapes_per_environment"], 4)
        self.assertEqual(
            hierarchical["aggregation_order"],
            (
                "paired_tape_median_within_environment_then_equal_weight_"
                "median_across_environments"
            ),
        )
        self.assertLess(
            hierarchical["paired_median_normalized_return_improvement"],
            0.0,
        )
        self.assertIs(
            hierarchical["positive_paired_median_improvement"],
            False,
        )

    def test_execution_accounting_prices_primary_and_replay_worlds(self) -> None:
        accounting = selection._selection_execution_accounting(
            [
                {
                    "execution_count": 5,
                    "exact_replay_execution_count": 0,
                    "initialized_baseline": {"execution_count": 4},
                }
                for _ in range(32)
            ],
            workers_requested=8,
            workers_used=8,
        )

        self.assertEqual(accounting["trained_policy_evaluation_count"], 160)
        self.assertEqual(accounting["initialized_baseline_evaluation_count"], 128)
        self.assertEqual(accounting["primary_evaluation_count"], 288)
        self.assertEqual(accounting["replay_evaluation_count"], 0)
        self.assertEqual(accounting["physical_world_run_count"], 288)
        self.assertEqual(
            selection._authorization_execution_accounting(accounting),
            {
                "producer_primary_world_run_count": 288,
                "authorization_reexecution_world_run_count": 288,
                "total_physical_world_run_count_through_authorization": 576,
            },
        )

    def test_artifact_binding_rejects_a_caller_authored_artifact_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal-training-artifact.json"
            run_contract_path = Path(directory) / "run-contract.json"
            model = _model()
            run_contract_digest = _write_run_contract(
                run_contract_path,
                model=model,
            )
            artifact = save_recurrent_artifact(
                path,
                model,
                training_config={"purpose": "test"},
                seed_registry_digest=OPEN_ECOLOGY_CANONICAL_SHA256,
                source_commit=_SOURCE_COMMIT,
                data_metadata={
                    "campaign_digest": _CAMPAIGN_DIGEST,
                    "run_contract_digest": run_contract_digest,
                    "environment_seed_role": "open_ecology_training",
                    "environment_seed_indices": [0],
                    "training_world_count": 1,
                    "training_scenarios": ["broad"],
                    "fixture_names": [],
                    "validation_accessed": False,
                    "lockbox_accessed": False,
                },
                run_metadata={
                    "purpose": "test_open_ecology_selection",
                    "cell_id": "A0",
                    "learner_index": 0,
                    "run_id": "test",
                    "terminal_commit_exact_digest": "e" * 64,
                    "source_manifest_sha256": _SOURCE_MANIFEST,
                    "selection_evaluation_pending": True,
                    "exact_cpu_reevaluation_required": True,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                },
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
                learner_device="cpu",
            )
            request = selection.OpenEcologySelectionRequest(
                artifact_path=path,
                run_contract_path=run_contract_path,
                expected_artifact_sha256=str(artifact["artifact_sha256"]),
                expected_source_commit=_SOURCE_COMMIT,
                expected_source_manifest_sha256=_SOURCE_MANIFEST,
                campaign_digest=_CAMPAIGN_DIGEST,
                run_contract_digest=run_contract_digest,
                phase=selection.OpenEcologySelectionPhase.PHASE_A,
                cell_id="A0",
                learner_index=0,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
            )
            primary, replay, binding = selection._load_bound_artifact(request)

            self.assertEqual(
                selection.recurrent_model_state_sha256(primary),
                selection.recurrent_model_state_sha256(replay),
            )
            self.assertEqual(
                binding["artifact_sha256"],
                artifact["artifact_sha256"],
            )
            relocated = Path(directory) / "relocated"
            relocated.mkdir()
            relocated_artifact = relocated / path.name
            relocated_contract = relocated / run_contract_path.name
            shutil.copyfile(path, relocated_artifact)
            shutil.copyfile(run_contract_path, relocated_contract)
            selection._validate_artifact_binding(
                binding,
                artifact_path=relocated_artifact,
                run_contract_path=relocated_contract,
                expected_campaign_digest=_CAMPAIGN_DIGEST,
                expected_run_contract_digest=run_contract_digest,
                expected_cell_id="A0",
                expected_learner_index=0,
                expected_learner_seed=OPEN_ECOLOGY_SEED_REGISTRY[
                    "open_ecology_learner"
                ][0],
            )
            task = selection._EnvironmentTask(
                phase="phase_a",
                cell_id="A0",
                learner_index=0,
                artifact_sha256=str(artifact["artifact_sha256"]),
                initialized_baseline_model_state_sha256=str(
                    binding["initialized_baseline_model_state_sha256"]
                ),
                local_environment_index=0,
                environment_seed_index=0,
                environment_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"][
                    0
                ],
                ticks=3,
                initial_agents=32,
                stochastic_tape_identities=(
                    "phase-a-selection-tape-00",
                    "phase-a-selection-tape-01",
                    "phase-a-selection-tape-02",
                    "phase-a-selection-tape-03",
                ),
                genome_stream_seed=OPEN_ECOLOGY_SEED_REGISTRY[
                    "open_ecology_genome_stream"
                ][0],
            )
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                sequential_environment = selection._evaluate_environment_task(
                    task,
                    primary_model=primary,
                    replay_model=None,
                    baseline_model=selection._reconstruct_initialized_baseline_model(
                        config=primary.config,
                        learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][
                            0
                        ],
                    ),
                )
            finally:
                torch.set_num_threads(previous_threads)
            with ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=selection._initialize_selection_worker,
                initargs=(request,),
            ) as executor:
                parallel_environment = executor.submit(
                    selection._evaluate_environment_task_in_worker,
                    task,
                ).result(timeout=60)
            self.assertEqual(sequential_environment, parallel_environment)
            forged = selection.OpenEcologySelectionRequest(
                artifact_path=path,
                run_contract_path=run_contract_path,
                expected_artifact_sha256="f" * 64,
                expected_source_commit=_SOURCE_COMMIT,
                expected_source_manifest_sha256=_SOURCE_MANIFEST,
                campaign_digest=_CAMPAIGN_DIGEST,
                run_contract_digest=run_contract_digest,
                phase=selection.OpenEcologySelectionPhase.PHASE_A,
                cell_id="A0",
                learner_index=0,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
            )
            with self.assertRaisesRegex(
                selection.OpenEcologySelectionError,
                "externally pinned",
            ):
                selection._load_bound_artifact(forged)

    def test_artifact_model_cannot_be_mislabeled_as_another_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "terminal-training-artifact.json"
            run_contract_path = root / "run-contract.json"
            run_contract_digest = _write_run_contract(
                run_contract_path,
                model=_model("A1"),
                cell_id="A1",
            )
            artifact = save_recurrent_artifact(
                artifact_path,
                _model("A0"),
                training_config={"purpose": "wrong-cell-test"},
                seed_registry_digest=OPEN_ECOLOGY_CANONICAL_SHA256,
                source_commit=_SOURCE_COMMIT,
                data_metadata={
                    "campaign_digest": _CAMPAIGN_DIGEST,
                    "run_contract_digest": run_contract_digest,
                    "fixture_names": [],
                    "validation_accessed": False,
                    "lockbox_accessed": False,
                },
                run_metadata={
                    "cell_id": "A1",
                    "learner_index": 0,
                    "source_manifest_sha256": _SOURCE_MANIFEST,
                    "selection_evaluation_pending": True,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                },
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
                learner_device="cpu",
            )
            request = selection.OpenEcologySelectionRequest(
                artifact_path=artifact_path,
                run_contract_path=run_contract_path,
                expected_artifact_sha256=str(artifact["artifact_sha256"]),
                expected_source_commit=_SOURCE_COMMIT,
                expected_source_manifest_sha256=_SOURCE_MANIFEST,
                campaign_digest=_CAMPAIGN_DIGEST,
                run_contract_digest=run_contract_digest,
                phase=selection.OpenEcologySelectionPhase.PHASE_A,
                cell_id="A1",
                learner_index=0,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
            )

            with self.assertRaisesRegex(
                selection.OpenEcologySelectionError,
                "preregistered tokenized",
            ):
                selection._load_bound_artifact(request)

    def test_terminal_authorization_wrapper_runs_exactly_two_worlds_per_tape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "terminal-training-artifact.json"
            run_contract_path = root / "run-contract.json"
            model = _model("A0")
            run_contract_digest = _write_run_contract(
                run_contract_path,
                model=model,
            )
            artifact = save_recurrent_artifact(
                artifact_path,
                model,
                training_config={"purpose": "two-pass-test"},
                seed_registry_digest=OPEN_ECOLOGY_CANONICAL_SHA256,
                source_commit=_SOURCE_COMMIT,
                data_metadata={
                    "campaign_digest": _CAMPAIGN_DIGEST,
                    "run_contract_digest": run_contract_digest,
                    "fixture_names": [],
                    "validation_accessed": False,
                    "lockbox_accessed": False,
                },
                run_metadata={
                    "cell_id": "A0",
                    "learner_index": 0,
                    "source_manifest_sha256": _SOURCE_MANIFEST,
                    "selection_evaluation_pending": True,
                    "runtime_integration_authorized": False,
                    "promotion_authorized": False,
                },
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
                learner_device="cpu",
            )
            request = selection.OpenEcologySelectionRequest(
                artifact_path=artifact_path,
                run_contract_path=run_contract_path,
                expected_artifact_sha256=str(artifact["artifact_sha256"]),
                expected_source_commit=_SOURCE_COMMIT,
                expected_source_manifest_sha256=_SOURCE_MANIFEST,
                campaign_digest=_CAMPAIGN_DIGEST,
                run_contract_digest=run_contract_digest,
                phase=selection.OpenEcologySelectionPhase.PHASE_A,
                cell_id="A0",
                learner_index=0,
                learner_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_learner"][0],
                training_authority=_training_authority(
                    artifact_path=artifact_path,
                    run_contract_path=run_contract_path,
                    artifact_sha256=str(artifact["artifact_sha256"]),
                    model_state_sha256=selection.recurrent_model_state_sha256(model),
                    run_contract_digest=run_contract_digest,
                ),
            )
            original_run_world = selection._run_selection_world
            with (
                patch.object(
                    selection,
                    "OPEN_ECOLOGY_SELECTION_ENVIRONMENT_COUNT",
                    1,
                ),
                patch.object(
                    selection,
                    "OPEN_ECOLOGY_SELECTION_CAUSAL_STATE_COUNT",
                    64,
                ),
                patch.object(selection, "_PHASE_A_SELECTION_TICKS", 3),
                patch.object(
                    selection,
                    "_run_selection_world",
                    wraps=original_run_world,
                ) as run_world,
            ):
                provisional = selection.evaluate_open_ecology_selection_artifact(
                    request
                )
                with patch.object(
                    phase_a,
                    "build_verified_phase_a_selection_request",
                    return_value=request,
                ) as build_request:
                    learner = phase_a.authorize_phase_a_terminal_selection_report(
                        {},
                        provisional,
                        cell_id="A0",
                        learner_index=0,
                        terminal_path=root / "missing-terminal.json",
                    )

            self.assertIs(provisional["authority"]["authoritative"], False)
            build_request.assert_called_once()
            self.assertEqual(
                provisional["execution"]["primary_evaluation_count"],
                9,
            )
            self.assertEqual(
                provisional["execution"]["replay_evaluation_count"],
                0,
            )
            self.assertEqual(
                learner["evaluation"][
                    "total_physical_world_run_count_through_authorization"
                ],
                18,
            )
            self.assertIs(
                learner["gates"]["exact_same_contract_replay"],
                True,
            )
            self.assertEqual(run_world.call_count, 18)

    def test_mapping_only_request_has_no_public_learner_authority_path(self) -> None:
        self.assertFalse(
            hasattr(
                selection,
                "phase_a_learner_evidence_from_selection_report",
            )
        )
        self.assertFalse(
            hasattr(
                selection,
                "verify_open_ecology_selection_report_by_reexecution",
            )
        )

    def test_real_argmax_states_reproduce_and_support_frozen_interventions(
        self,
    ) -> None:
        model = _model()
        task = selection._EnvironmentTask(
            phase="phase_a",
            cell_id="A0",
            learner_index=0,
            artifact_sha256="a" * 64,
            initialized_baseline_model_state_sha256=(
                selection.recurrent_model_state_sha256(model)
            ),
            local_environment_index=0,
            environment_seed_index=0,
            environment_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"][0],
            ticks=3,
            initial_agents=32,
            stochastic_tape_identities=(
                "phase-a-selection-tape-00",
                "phase-a-selection-tape-01",
                "phase-a-selection-tape-02",
                "phase-a-selection-tape-03",
            ),
            genome_stream_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_genome_stream"][
                0
            ],
        )

        environment = selection._evaluate_environment_task(
            task,
            primary_model=model,
            replay_model=None,
            baseline_model=selection.frozen_cpu_model_copy(model),
        )
        authoritative_reexecution = selection._evaluate_environment_task(
            task,
            primary_model=selection.frozen_cpu_model_copy(model),
            replay_model=None,
            baseline_model=selection.frozen_cpu_model_copy(model),
        )
        causal = environment["causal_genome"]

        self.assertEqual(environment["execution_count"], 5)
        self.assertEqual(environment["exact_replay_execution_count"], 0)
        self.assertEqual(environment, authoritative_reexecution)
        self.assertEqual(causal["state_count"], 64)
        self.assertIs(causal["complete"], True)
        self.assertEqual(len(causal["state_rows"]), 64)
        self.assertTrue(
            all(
                row["genome_sha256"] != row["donor_genome_sha256"]
                for row in causal["state_rows"]
            )
        )

        forged = copy.deepcopy(environment)
        forged_execution = forged["executions"][0]
        forged_execution["action_source_counts"] = {
            "fabricated_nonheuristic_source": forged_execution[
                "trajectory_record_count"
            ]
        }
        forged_execution["heuristic_action_source_count"] = 0
        forged_execution["gates"]["no_heuristic_action_source"] = True
        run_unsigned = dict(forged_execution)
        run_unsigned.pop("run_evidence_sha256")
        run_unsigned.pop("exact_replay")
        run_unsigned.pop("exact_digest")
        forged_execution["run_evidence_sha256"] = selection.stable_payload_digest(
            run_unsigned
        )
        _resign(forged_execution, "exact_digest")
        forged["stochastic_tape_aggregate"] = selection._aggregate_stochastic_runs(
            forged["executions"][:4]
        )
        _resign(forged, "exact_digest")

        with self.assertRaisesRegex(
            selection.OpenEcologySelectionError,
            "action sources",
        ):
            selection._validate_environment_evidence(forged)

        forged_target = copy.deepcopy(environment)
        target_execution = forged_target["executions"][0]
        target_execution["metrics"]["target_contract"] = "forged_stale_bootstrap_target"
        target_run_unsigned = dict(target_execution)
        target_run_unsigned.pop("run_evidence_sha256")
        target_run_unsigned.pop("exact_replay")
        target_run_unsigned.pop("exact_digest")
        target_execution["run_evidence_sha256"] = selection.stable_payload_digest(
            target_run_unsigned
        )
        _resign(target_execution, "exact_digest")
        _resign(forged_target, "exact_digest")
        with self.assertRaisesRegex(
            selection.OpenEcologySelectionError,
            "target contract",
        ):
            selection._validate_environment_evidence(forged_target)

        forged_provenance = copy.deepcopy(environment)
        provenance_execution = forged_provenance["executions"][0]
        provenance_execution["genome_population_provenance"][
            "genome_population_mode"
        ] = "zero_all"
        _resign(
            provenance_execution["genome_population_provenance"],
            "provenance_sha256",
        )
        provenance_run_unsigned = dict(provenance_execution)
        provenance_run_unsigned.pop("run_evidence_sha256")
        provenance_run_unsigned.pop("exact_replay")
        provenance_run_unsigned.pop("exact_digest")
        provenance_execution["run_evidence_sha256"] = selection.stable_payload_digest(
            provenance_run_unsigned
        )
        _resign(provenance_execution, "exact_digest")
        forged_provenance["stochastic_tape_aggregate"] = (
            selection._aggregate_stochastic_runs(forged_provenance["executions"][:4])
        )
        _resign(forged_provenance, "exact_digest")
        with self.assertRaisesRegex(
            selection.OpenEcologySelectionError,
            "genome provenance differs",
        ):
            selection._validate_environment_evidence(forged_provenance)

    def test_capture_adapter_is_behaviorally_inert_across_all_four_cells(
        self,
    ) -> None:
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for learner_index, cell_id in enumerate(
                selection._PHASE_A_CELL_MODEL_CONTRACTS
            ):
                model = _model(cell_id)
                model_sha = selection.recurrent_model_state_sha256(model)
                task = selection._EnvironmentTask(
                    phase="capture_noninterference",
                    cell_id=cell_id,
                    learner_index=learner_index,
                    artifact_sha256=model_sha,
                    initialized_baseline_model_state_sha256=model_sha,
                    local_environment_index=learner_index,
                    environment_seed_index=learner_index,
                    environment_seed=OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"][
                        learner_index
                    ],
                    ticks=3,
                    initial_agents=32,
                    stochastic_tape_identities=(),
                    genome_stream_seed=OPEN_ECOLOGY_SEED_REGISTRY[
                        "open_ecology_genome_stream"
                    ][learner_index],
                )
                capture_model = selection.frozen_cpu_model_copy(model)
                ordinary_model = selection.frozen_cpu_model_copy(model)
                captured, states = selection._run_selection_world(
                    model=capture_model,
                    task=task,
                    tape_identity="argmax",
                    sampling_seed=None,
                    capture_causal_states=True,
                )
                ordinary, ordinary_states = selection._run_selection_world(
                    model=ordinary_model,
                    task=task,
                    tape_identity="argmax",
                    sampling_seed=None,
                    capture_causal_states=False,
                )
                self.assertEqual(len(states), 64)
                self.assertEqual(ordinary_states, ())
                self.assertEqual(captured, ordinary)
                self.assertEqual(
                    selection.recurrent_model_state_sha256(capture_model),
                    model_sha,
                )
                self.assertEqual(
                    selection.recurrent_model_state_sha256(ordinary_model),
                    model_sha,
                )
        finally:
            torch.set_num_threads(previous_threads)

    def test_capture_proof_uses_dedicated_full_cell_density_matrix(self) -> None:
        cases = selection._capture_noninterference_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(
            {(case["cell_id"], case["initial_agents"]) for case in cases},
            {
                (cell_id, density)
                for cell_id in selection._PHASE_A_CELL_MODEL_CONTRACTS
                for density in (32, 64, 128)
            },
        )
        self.assertTrue(
            all(case["environment_seed_role"] == "open_ecology_proof" for case in cases)
        )
        self.assertTrue(
            {int(case["environment_seed"]) for case in cases}.isdisjoint(
                OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"]
            )
        )

    def test_structural_capture_mapping_requires_full_reexecution_for_authority(
        self,
    ) -> None:
        rows: list[dict[str, object]] = []
        for index, expected in enumerate(selection._capture_noninterference_cases()):
            row: dict[str, object] = {
                **expected,
                "initial_model_state_sha256": "1" * 64,
                "capture_final_model_state_sha256": "1" * 64,
                "ordinary_final_model_state_sha256": "1" * 64,
                "captured_state_count": 64,
                "capture_full_behavior_sha256": "2" * 64,
                "ordinary_full_behavior_sha256": "2" * 64,
                "capture_run_evidence_sha256": "3" * 64,
                "ordinary_run_evidence_sha256": "3" * 64,
                "full_run_evidence_equal": True,
                "model_state_unchanged": True,
                "births": 1 if index == 0 else 0,
                "deaths": 1 if index == 0 else 0,
                "passed": True,
            }
            row["exact_digest"] = selection.stable_payload_digest(row)
            rows.append(row)
        proof: dict[str, object] = {
            "schema_version": (
                selection.OPEN_ECOLOGY_CAPTURE_NONINTERFERENCE_SCHEMA_VERSION
            ),
            "source": {
                "commit": _SOURCE_COMMIT,
                "manifest_sha256": _SOURCE_MANIFEST,
            },
            "contract": {
                "adapter": "read_only_protected_state_capture_v1",
                "ordinary_policy": selection.PUBLIC_RECURRENT_POLICY_ID,
                "device": "cpu",
                "parameter_dtype": "torch.float32",
                "torch_num_threads": 1,
                "cell_order": list(selection._PHASE_A_CELL_MODEL_CONTRACTS),
                "case_matrix": "four_cells_by_three_densities_cartesian_v1",
                "density_levels": [32, 64, 128],
                "environment_seed_role": "open_ecology_proof",
                "environment_seed_indices": list(range(12)),
                "scientific_selection_seed_accessed": False,
                "case_count": 12,
                "ticks_per_case": 128,
                "comparison": (
                    "full_summary_trajectory_decision_diagnostics_rng_and_"
                    "genome_provenance"
                ),
            },
            "cases": rows,
            "dynamics": {
                "births_observed": True,
                "deaths_observed": True,
                "total_births": 1,
                "total_deaths": 1,
            },
            "passed": True,
        }
        proof["exact_digest"] = selection.stable_payload_digest(proof)
        selection.validate_open_ecology_capture_noninterference_proof(
            proof,
            expected_source_commit=_SOURCE_COMMIT,
            expected_source_manifest_sha256=_SOURCE_MANIFEST,
        )
        with (
            patch.object(
                selection,
                "build_open_ecology_capture_noninterference_proof",
                return_value={**proof, "exact_digest": "f" * 64},
            ),
            patch.object(phase_a, "validate_open_ecology_phase_a_preregistration"),
            patch.object(phase_a, "_require_live_source"),
            self.assertRaisesRegex(
                selection.OpenEcologySelectionError,
                "changed under exact reexecution",
            ),
        ):
            selection.verify_open_ecology_capture_noninterference_proof_by_reexecution(
                proof,
                preregistration={
                    "source": {
                        "commit": _SOURCE_COMMIT,
                        "manifest_sha256": _SOURCE_MANIFEST,
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
