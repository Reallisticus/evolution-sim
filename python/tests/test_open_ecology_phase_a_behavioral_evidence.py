from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.signals import SignalFieldState
from evolution_sim.mind import open_ecology_phase_a_readiness as readiness
from evolution_sim.mind import recurrent_experiment
from evolution_sim.mind import recurrent_genome_population
from evolution_sim.mind.open_ecology_phase_a_behavioral_evidence import (
    _critic_gradient_facts,
    _cross_surface_facts,
    _d4_input_rows,
    _d4_phase_a_collection_inputs,
    _d4_probe_spec,
    _d4_proof_seed_contract,
    _fixed_batch_facts,
    _run_critic_gradient_probe,
    _run_cross_surface_probe,
    _run_fixed_batch_probe,
    _run_d4_equivalence,
    _run_runtime_genome_probe,
    _run_viewer_validator_process,
    _runtime_genome_facts,
    _viewer_surface_probe,
    produce_cross_surface_and_self_echo_report,
    produce_critic_gradient_and_density_schedule_report,
    produce_fixed_batch_equivalence_and_speed_report,
    produce_runtime_genome_and_action_source_report,
    verify_cross_surface_and_self_echo_report,
    verify_critic_gradient_and_density_schedule_report,
    verify_fixed_batch_equivalence_and_speed_report,
    verify_runtime_genome_and_action_source_report,
)
from evolution_sim.mind.open_ecology_seed_registry import OPEN_ECOLOGY_SEED_REGISTRY
from evolution_sim.mind.recurrent_actor_critic import (
    RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
    RECURRENT_NUMERIC_KERNEL_VERSION,
)
from evolution_sim.mind.recurrent_genome import RecurrentControllerGenome
from evolution_sim.mind.recurrent_rollout import (
    OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
    RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS,
    PreviousPublicFeedback,
    RecurrentCoreOutput,
    RecurrentOnPolicyCollector,
    TorchRecurrentPolicyCore,
)
from python.tests.test_open_ecology_phase_a import _campaign


REQUIRES_MIND_ML = True

_D4_BUCKET_MATRIX_ACTIVE_ROWS = (
    1,
    2,
    3,
    5,
    9,
    17,
    33,
    64,
    65,
    129,
    257,
    319,
    320,
)
_D4_BUCKET_MATRIX_EXECUTION_ROWS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    64,
    128,
    256,
    320,
    320,
    320,
)
_D4_BUCKET_MATRIX_CASES = [
    {
        "active_rows": active_rows,
        "execution_rows": execution_rows,
        "observed_recurrent_input_shape": [1, execution_rows, 256],
    }
    for active_rows, execution_rows in zip(
        _D4_BUCKET_MATRIX_ACTIVE_ROWS,
        _D4_BUCKET_MATRIX_EXECUTION_ROWS,
        strict=True,
    )
]
_D4_BUCKET_MATRIX_COMPARISON_COUNT = sum(_D4_BUCKET_MATRIX_ACTIVE_ROWS)


class OpenEcologyPhaseABehavioralEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.preregistration = _campaign()

    def test_d1_real_surfaces_receiver_and_self_echo_repeat_exactly(self) -> None:
        first = _run_cross_surface_probe()
        second = _run_cross_surface_probe()

        self.assertEqual(first, second)
        facts = _cross_surface_facts(first)
        readiness._SEMANTIC_VALIDATORS["cross_surface_and_self_echo"](
            facts,
            self.preregistration,
            None,
        )
        self.assertEqual(set(first["surface_token_count"].values()), {4})
        self.assertEqual(
            set(first["surface_profiles_per_token"].values()),
            {2},
        )
        self.assertEqual(
            first["surface_action_order"]["viewer"],
            first["surface_action_order"]["replay"],
        )
        self.assertEqual(
            facts["emitter_own_signal_self_projection_nonzero_count"],
            0,
        )
        self.assertEqual(
            facts["emitter_own_signal_local_patch_projection_nonzero_count"],
            0,
        )
        self.assertGreater(
            facts["adjacent_receiver_self_projection_positive_count"],
            0,
        )
        self.assertGreater(
            facts["adjacent_receiver_local_patch_projection_positive_count"],
            0,
        )
        self.assertEqual(facts["self_echo_case_count"], 8)
        self.assertEqual(facts["signal_emission_success_count"], 8)
        self.assertEqual(facts["global_signal_field_positive_count"], 8)
        self.assertEqual(facts["global_signal_emission_event_count"], 8)
        self.assertEqual(facts["signal_action_identity_mismatch_count"], 0)

    def test_d1_total_receiver_projection_blackout_cannot_pass(self) -> None:
        def blackout(
            state: SignalFieldState,
            name: str,
            *,
            x: int,
            y: int,
            receiver_agent_id: int,
        ) -> float:
            del state, name, x, y, receiver_agent_id
            return 0.0

        with patch.object(
            SignalFieldState,
            "field_value_for_receiver",
            new=blackout,
        ):
            observed = _run_cross_surface_probe()
        facts = _cross_surface_facts(observed)
        self.assertEqual(
            facts["adjacent_receiver_self_projection_positive_count"],
            0,
        )
        self.assertEqual(
            facts["adjacent_receiver_local_patch_projection_positive_count"],
            0,
        )
        with self.assertRaises(Exception):
            readiness._SEMANTIC_VALIDATORS["cross_surface_and_self_echo"](
                facts,
                self.preregistration,
                None,
            )

    def test_d1_every_token_profile_action_is_exercised(self) -> None:
        from evolution_sim.env.runtime import signals as runtime_signals

        original = runtime_signals.emit_communication_signal_action

        def break_nonfirst_actions(
            agent: object,
            action: str,
            *,
            context: object,
        ) -> dict[str, object]:
            if action != "signal_0_profile_0":
                raise RuntimeError("non-first signal path is broken")
            return original(agent, action, context=context)

        with (
            patch.object(
                runtime_signals,
                "emit_communication_signal_action",
                new=break_nonfirst_actions,
            ),
            self.assertRaisesRegex(RuntimeError, "non-first signal"),
        ):
            _run_cross_surface_probe()

    def test_d1_broken_stdin_hanging_validator_is_killed_promptly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hostile_node = Path(temporary) / "hostile-node"
            child_pid_path = Path(temporary) / "child.pid"
            child_body = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "while True: time.sleep(1)\n"
            )
            hostile_node.write_text(
                f"#!{sys.executable}\n"
                "import os\n"
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "child=subprocess.Popen("
                f"[sys.executable,'-c',{child_body!r}],"
                "stdin=subprocess.DEVNULL)\n"
                f"Path({str(child_pid_path)!r}).write_text("
                "str(child.pid),encoding='ascii')\n"
                "os.close(0)\n"
                "time.sleep(60)\n",
                encoding="utf-8",
            )
            hostile_node.chmod(0o500)
            started = time.monotonic()
            with (
                patch(
                    "evolution_sim.mind."
                    "open_ecology_phase_a_behavioral_evidence.shutil.which",
                    return_value=str(hostile_node),
                ),
                self.assertRaisesRegex(RuntimeError, "closed stdin"),
            ):
                _viewer_surface_probe({"padding": "x" * (900 * 1024)})
            self.assertLess(time.monotonic() - started, 5.0)
            _assert_process_gone(self, child_pid_path)

    def test_d1_nonreader_stdin_stall_is_bounded_and_closes_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            child_body = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "while True: time.sleep(1)\n"
            )
            parent_body = (
                "from pathlib import Path\n"
                "import signal\n"
                "import subprocess\n"
                "import sys\n"
                "import time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "child=subprocess.Popen("
                f"[sys.executable,'-c',{child_body!r}],"
                "stdin=subprocess.DEVNULL)\n"
                f"Path({str(child_pid_path)!r}).write_text("
                "str(child.pid),encoding='ascii')\n"
                "while True: time.sleep(1)\n"
            )
            process = _start_hostile_validator(parent_body)
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    _run_viewer_validator_process(
                        process,
                        input_bytes=b"x" * (900 * 1024),
                        maximum_output_bytes=256 * 1024,
                        timeout_seconds=0.5,
                    )
                self.assertLess(time.monotonic() - started, 3.0)
                self.assertIsNotNone(process.returncode)
                _assert_process_gone(self, child_pid_path)
            finally:
                _force_process_group_closed(process)

    def test_d1_successful_leader_exit_still_closes_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            child_body = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "while True: time.sleep(1)\n"
            )
            parent_body = (
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n"
                "child=subprocess.Popen("
                f"[sys.executable,'-c',{child_body!r}],"
                "stdin=subprocess.DEVNULL)\n"
                f"Path({str(child_pid_path)!r}).write_text("
                "str(child.pid),encoding='ascii')\n"
                "payload=sys.stdin.buffer.read()\n"
                "sys.stdout.buffer.write(b'accepted:'+payload)\n"
            )
            process = _start_hostile_validator(parent_body)
            try:
                returncode, stdout, stderr = _run_viewer_validator_process(
                    process,
                    input_bytes=b"replay",
                    maximum_output_bytes=256 * 1024,
                    timeout_seconds=2.0,
                )
                self.assertEqual(returncode, 0)
                self.assertEqual(stdout, b"accepted:replay")
                self.assertEqual(stderr, b"")
                self.assertIsNotNone(process.returncode)
                _assert_process_gone(self, child_pid_path)
            finally:
                _force_process_group_closed(process)

    def test_d1_zero_exit_before_complete_input_is_rejected_and_closes_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            child_pid_path = Path(temporary) / "child.pid"
            child_body = (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                "while True: time.sleep(1)\n"
            )
            parent_body = (
                "from pathlib import Path\n"
                "import subprocess\n"
                "import sys\n"
                "child=subprocess.Popen("
                f"[sys.executable,'-c',{child_body!r}])\n"
                f"Path({str(child_pid_path)!r}).write_text("
                "str(child.pid),encoding='ascii')\n"
                "sys.stdout.buffer.write(b'premature-success')\n"
            )
            process = _start_hostile_validator(parent_body)
            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "before consuming the complete replay",
                ):
                    _run_viewer_validator_process(
                        process,
                        input_bytes=b"x" * (900 * 1024),
                        maximum_output_bytes=256 * 1024,
                        timeout_seconds=2.0,
                    )
                self.assertIsNotNone(process.returncode)
                _assert_process_gone(self, child_pid_path)
            finally:
                _force_process_group_closed(process)

    def test_d2_actual_world_hooks_runtime_and_snapshot_replay_repeat(self) -> None:
        first = _run_runtime_genome_probe()
        second = _run_runtime_genome_probe()

        self.assertEqual(first, second)
        facts = _runtime_genome_facts(first)
        readiness._SEMANTIC_VALIDATORS["runtime_genome_and_action_source"](
            facts,
            self.preregistration,
            None,
        )
        self.assertEqual(facts["heuristic_action_source_count"], 0)
        self.assertEqual(facts["genome_digest_replay_mismatch_count"], 0)
        self.assertEqual(facts["birth_state_reset_count"], 2)
        self.assertEqual(facts["parent_pre_birth_nonzero_state_count"], 2)
        self.assertEqual(facts["exact_child_derivation_mismatch_count"], 0)
        self.assertEqual(facts["inheritance_bound_violation_count"], 0)
        self.assertEqual(facts["missing_genome_rejection_count"], 2)
        self.assertGreaterEqual(facts["decision_count"], 2)

    def test_d2_broken_founder_world_hook_fails_probe(self) -> None:
        with (
            patch.object(
                SimulationWorld,
                "_founder_mind_metadata",
                return_value={},
            ),
            self.assertRaises(Exception),
        ):
            _run_runtime_genome_probe()

    def test_d2_broken_child_world_hook_fails_probe(self) -> None:
        with (
            patch.object(
                SimulationWorld,
                "_mind_inheritance_for_child",
                return_value={},
            ),
            self.assertRaises(Exception),
        ):
            _run_runtime_genome_probe()

    def test_d2_copying_live_parent_state_into_child_is_rejected(self) -> None:
        def copy_parent_state(
            collector: RecurrentOnPolicyCollector,
            *,
            child_agent_id: int,
            primary_parent_id: int,
            secondary_parent_id: int | None,
        ) -> dict[str, object]:
            if primary_parent_id in collector._hidden_by_agent:
                collector._hidden_by_agent[child_agent_id] = collector._hidden_by_agent[
                    primary_parent_id
                ]
            if primary_parent_id in collector._feedback_by_agent:
                collector._feedback_by_agent[child_agent_id] = (
                    collector._feedback_by_agent[primary_parent_id]
                )
            return collector._require_genome_population_manager().child_metadata(
                child_agent_id=child_agent_id,
                primary_parent_id=primary_parent_id,
                secondary_parent_id=secondary_parent_id,
            )

        with patch.object(
            RecurrentOnPolicyCollector,
            "child_metadata",
            new=copy_parent_state,
        ):
            facts = _runtime_genome_facts(_run_runtime_genome_probe())
        self.assertLess(facts["birth_state_reset_count"], 2)
        with self.assertRaises(Exception):
            readiness._SEMANTIC_VALIDATORS["runtime_genome_and_action_source"](
                facts,
                self.preregistration,
                None,
            )

    def test_d2_in_bounds_but_wrong_actual_child_is_rejected(self) -> None:
        original = SimulationWorld._mind_inheritance_for_child

        def tamper_live_child(
            world: SimulationWorld,
            primary_parent: object,
            secondary_parent: object,
            child_agent_id: int,
        ) -> dict[str, object]:
            metadata = original(
                world,
                primary_parent,
                secondary_parent,
                child_agent_id,
            )
            manager = world.policy._genome_population_manager
            if manager is not None and manager.mode.value == "heritable":
                parent_genome = manager.genome_for_agent(primary_parent.agent_id)
                child_genome = manager.genome_for_agent(child_agent_id)
                values = list(child_genome.values)
                parent_value = float(parent_genome.values[0])
                values[0] = round(
                    parent_value + (0.04 if parent_value <= 0.96 else -0.04),
                    8,
                )
                manager._bindings[child_agent_id] = (
                    recurrent_genome_population._bind_genome(
                        RecurrentControllerGenome(values=tuple(values))
                    )
                )
                manager._invalidate_state_sha256()
            return metadata

        with patch.object(
            SimulationWorld,
            "_mind_inheritance_for_child",
            new=tamper_live_child,
        ):
            facts = _runtime_genome_facts(_run_runtime_genome_probe())
        self.assertGreater(facts["exact_child_derivation_mismatch_count"], 0)
        self.assertEqual(facts["inheritance_bound_violation_count"], 0)
        with self.assertRaises(Exception):
            readiness._SEMANTIC_VALIDATORS["runtime_genome_and_action_source"](
                facts,
                self.preregistration,
                None,
            )

    def test_d3_all_four_gradient_routes_density_and_torch_state(self) -> None:
        import torch

        rng_before = torch.get_rng_state().clone()
        threads_before = torch.get_num_threads()
        first = _run_critic_gradient_probe(self.preregistration)
        second = _run_critic_gradient_probe(self.preregistration)

        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads_before)
        self.assertEqual(
            [row["cell_id"] for row in first["cell_gradient_observations"]],
            ["A0", "A1", "A2", "A3"],
        )
        facts = _critic_gradient_facts(first)
        readiness._SEMANTIC_VALIDATORS["critic_gradient_and_density_schedule"](
            facts,
            self.preregistration,
            None,
        )
        self.assertGreater(facts["shared_value_trunk_gradient_l2"], 0.0)
        self.assertEqual(facts["stop_gradient_value_trunk_gradient_l2"], 0.0)
        self.assertEqual(
            facts["observed_cell_contracts"],
            facts["cell_contracts"],
        )
        self.assertGreater(
            facts["critic_conditioned_genome_sensitivity_min_abs_delta"],
            0.0,
        )
        self.assertEqual(
            facts["critic_unconditioned_genome_sensitivity_max_abs_delta"],
            0.0,
        )
        self.assertGreater(
            facts["critic_conditioned_parameter_gradient_l2"],
            0.0,
        )

    def test_d3_rejecting_a_canonical_cell_fails_probe(self) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        original = behavioral.PublicRecurrentActorCritic

        def reject_a0_a1(config: object, *args: object, **kwargs: object) -> object:
            if getattr(config, "critic_genome_conditioning", None) == "none":
                raise RuntimeError("A0/A1 constructor rejected")
            return original(config, *args, **kwargs)

        with (
            patch.object(
                behavioral,
                "PublicRecurrentActorCritic",
                new=reject_a0_a1,
            ),
            self.assertRaisesRegex(RuntimeError, "A0/A1"),
        ):
            _run_critic_gradient_probe(self.preregistration)

    def test_d3_density_generator_drift_is_observed_and_rejected(self) -> None:
        original = recurrent_experiment._open_ecology_density_cycle

        def drift(phase: object) -> tuple[int, ...]:
            if phase == "phase_a":
                return (63,)
            return original(phase)

        with patch.object(
            recurrent_experiment,
            "_open_ecology_density_cycle",
            new=drift,
        ):
            observed = _run_critic_gradient_probe(self.preregistration)
        facts = _critic_gradient_facts(observed)
        self.assertEqual(facts["phase_a_density_schedule"], [63])
        with self.assertRaises(Exception):
            readiness._SEMANTIC_VALIDATORS["critic_gradient_and_density_schedule"](
                facts,
                self.preregistration,
                None,
            )

    def test_d3_silently_ignored_film_conditioning_is_rejected(self) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        original = behavioral.PublicRecurrentActorCritic

        def ignore_critic_conditioning(
            config: object,
            *args: object,
            **kwargs: object,
        ) -> object:
            return original(
                replace(config, critic_genome_conditioning="none"),
                *args,
                **kwargs,
            )

        with patch.object(
            behavioral,
            "PublicRecurrentActorCritic",
            new=ignore_critic_conditioning,
        ):
            facts = _critic_gradient_facts(
                _run_critic_gradient_probe(self.preregistration)
            )
        self.assertNotEqual(
            facts["observed_cell_contracts"],
            facts["cell_contracts"],
        )
        self.assertEqual(
            facts["critic_conditioned_genome_sensitivity_min_abs_delta"],
            0.0,
        )
        with self.assertRaises(Exception):
            readiness._SEMANTIC_VALIDATORS["critic_gradient_and_density_schedule"](
                facts,
                self.preregistration,
                None,
            )

    def test_small_d4_uses_capacity_320_numeric_equivalence_and_restores_torch(
        self,
    ) -> None:
        import torch

        capacities: list[int] = []
        original = TorchRecurrentPolicyCore.forward_fixed_batch

        def record_capacity(
            core: TorchRecurrentPolicyCore,
            *args: object,
            **kwargs: object,
        ) -> object:
            capacities.append(int(kwargs["batch_capacity"]))
            return original(core, *args, **kwargs)

        rng_before = torch.get_rng_state().clone()
        threads_before = torch.get_num_threads()
        with patch.object(
            TorchRecurrentPolicyCore,
            "forward_fixed_batch",
            new=record_capacity,
        ):
            observed = _run_fixed_batch_probe(
                device="cpu",
                shape={"worlds": 1, "rollout_ticks": 2, "initial_agents": 64},
                repeat_count=2,
            )
        facts = _fixed_batch_facts(observed)
        forged_kernel = copy.deepcopy(observed)
        forged_kernel["numeric_kernel"] = "unsealed_dense_kernel"
        with self.assertRaisesRegex(
            ValueError,
            "recurrent model or numeric-kernel contract drifted",
        ):
            _fixed_batch_facts(forged_kernel)
        missing_identity = copy.deepcopy(observed)
        del missing_identity["model_contract_version"]
        with self.assertRaisesRegex(ValueError, "raw observation root schema drifted"):
            _fixed_batch_facts(missing_identity)
        surplus_identity = copy.deepcopy(observed)
        surplus_identity["legacy_model_contract"] = "v4"
        with self.assertRaisesRegex(ValueError, "raw observation root schema drifted"):
            _fixed_batch_facts(surplus_identity)

        self.assertEqual(set(capacities), {OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY})
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(torch.get_num_threads(), threads_before)
        self.assertEqual(facts["batch_capacity"], 320)
        self.assertEqual(
            facts["fixed_batch_execution_buckets"],
            list(RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS),
        )
        self.assertEqual(facts["device"], "cpu")
        self.assertEqual(
            facts["model_contract_version"],
            RECURRENT_ACTOR_CRITIC_CONTRACT_VERSION,
        )
        self.assertEqual(
            facts["numeric_kernel"],
            RECURRENT_NUMERIC_KERNEL_VERSION,
        )
        self.assertIsNone(facts["deterministic_cuda_contract"])
        self.assertEqual(
            facts["proof_seed_contract"]["engineering_seed_role"],
            "open_ecology_proof",
        )
        self.assertEqual(
            facts["proof_seed_contract"]["environment_seeds"],
            list(OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"][:1]),
        )
        self.assertEqual(
            facts["proof_seed_contract"]["scientific_training_seed_access_count"],
            0,
        )
        self.assertEqual(
            facts["proof_seed_contract"]["scientific_selection_seed_access_count"],
            0,
        )
        self.assertEqual(
            facts["timing_order"],
            [["scalar", "batched"], ["batched", "scalar"]],
        )
        self.assertEqual(facts["scalar_timing_sample_count"], 2)
        self.assertEqual(facts["batched_timing_sample_count"], 2)
        self.assertEqual(len(observed["scalar"]["samples"]), 2)
        self.assertEqual(len(observed["batched"]["samples"]), 2)
        self.assertEqual(
            [sample["repeat_index"] for sample in observed["scalar"]["samples"]],
            [0, 1],
        )
        self.assertEqual(
            [
                sample["timing_order_index"]
                for sample in observed["batched"]["samples"]
            ],
            [1, 0],
        )
        self.assertEqual(
            observed["batched"]["samples"][0]["collector_path"],
            observed["batched_collector_path"],
        )
        self.assertEqual(
            len(set(facts["scalar_repeat_collector_path_sha256"])),
            1,
        )
        self.assertEqual(
            len(set(facts["batched_repeat_collector_path_sha256"])),
            1,
        )
        self.assertEqual(facts["action_mismatch_count"], 0)
        self.assertEqual(facts["numeric_comparison_count"], 128)
        self.assertEqual(facts["reference_numeric_comparison_count"], 128)
        self.assertEqual(facts["collector_device"], "cpu")
        self.assertEqual(facts["rollout_workers"], 1)
        self.assertGreater(facts["collector_transition_count"], 0)
        self.assertEqual(facts["collector_semantic_mismatch_count"], 0)
        self.assertEqual(
            facts["collector_paired_transition_count"],
            facts["collector_transition_count"],
        )
        self.assertEqual(
            facts["collector_numeric_transition_comparison_count"],
            facts["collector_transition_count"],
        )
        self.assertEqual(
            facts["collector_hidden_component_comparison_count"],
            facts["collector_transition_count"] * 256,
        )
        bootstrap_value_comparison_count = facts[
            "collector_bootstrap_value_comparison_count"
        ]
        self.assertGreater(bootstrap_value_comparison_count, 0)
        self.assertLessEqual(
            bootstrap_value_comparison_count,
            facts["collector_transition_count"],
        )
        for mode in ("scalar", "batched"):
            self.assertEqual(
                {
                    sample["collector_path"]["bootstrap_value_count"]
                    for sample in observed[mode]["samples"]
                },
                {bootstrap_value_comparison_count},
            )
        self.assertEqual(facts["collector_identity_mismatch_count"], 0)
        self.assertEqual(facts["collector_hidden_shape_mismatch_count"], 0)
        self.assertEqual(facts["collector_bootstrap_none_mismatch_count"], 0)
        for field in (
            "collector_max_input_hidden_tolerance_ratio",
            "collector_max_logprob_tolerance_ratio",
            "collector_max_entropy_tolerance_ratio",
            "collector_max_value_tolerance_ratio",
            "collector_max_bootstrap_value_tolerance_ratio",
        ):
            self.assertGreaterEqual(facts[field], 0.0)
            self.assertLessEqual(facts[field], 1.0)
        self.assertEqual(facts["scalar_fixed_batch_step_count"], 0)
        self.assertEqual(
            facts["batched_fixed_batch_step_count"],
            facts["collector_transition_count"],
        )
        execution_bucket_histogram = (
            facts["batched_fixed_batch_execution_bucket_histogram"]
        )
        self.assertEqual(
            set(execution_bucket_histogram),
            {str(bucket) for bucket in RECURRENT_FIXED_BATCH_EXECUTION_BUCKETS},
        )
        self.assertEqual(
            sum(execution_bucket_histogram.values()),
            facts["batched_fixed_batch_call_count"],
        )
        self.assertEqual(
            sum(
                int(bucket) * call_count
                for bucket, call_count in execution_bucket_histogram.items()
            ),
            facts["batched_fixed_batch_execution_row_slots"],
        )
        self.assertGreater(facts["batched_fixed_batch_call_count"], 0)
        self.assertGreater(facts["batched_fixed_batch_active_row_slots"], 0)
        self.assertLessEqual(
            facts["batched_fixed_batch_active_row_slots"],
            facts["batched_fixed_batch_execution_row_slots"],
        )
        self.assertAlmostEqual(
            facts["batched_fixed_batch_slot_utilization"],
            facts["batched_fixed_batch_active_row_slots"]
            / facts["batched_fixed_batch_execution_row_slots"],
        )
        self.assertEqual(
            facts["batched_fixed_batch_topology_mismatch_count"],
            0,
        )
        bucket_matrix = facts["bounded_bucket_numeric_matrix"]
        self.assertEqual(
            bucket_matrix["cases"],
            _D4_BUCKET_MATRIX_CASES,
        )
        self.assertEqual(bucket_matrix["genome_conditioning_mode"], "actor_film_v1")
        self.assertEqual(
            bucket_matrix["genome_conditioned_row_count"],
            _D4_BUCKET_MATRIX_COMPARISON_COUNT,
        )
        self.assertEqual(
            bucket_matrix["comparison_count"],
            _D4_BUCKET_MATRIX_COMPARISON_COUNT,
        )
        self.assertEqual(
            bucket_matrix["reference_comparison_count"],
            _D4_BUCKET_MATRIX_COMPARISON_COUNT,
        )
        self.assertEqual(bucket_matrix["action_mismatch_count"], 0)
        self.assertEqual(
            bucket_matrix["scalar_reference_action_mismatch_count"],
            0,
        )
        self.assertEqual(
            bucket_matrix["batched_reference_action_mismatch_count"],
            0,
        )
        self.assertGreater(bucket_matrix["distinct_action_mask_count"], 1)
        self.assertGreater(bucket_matrix["distinct_feedback_vector_count"], 1)
        self.assertEqual(
            bucket_matrix["nonzero_feedback_row_count"],
            _D4_BUCKET_MATRIX_COMPARISON_COUNT,
        )
        self.assertEqual(len(bucket_matrix["action_mask_input_sha256"]), 64)
        self.assertEqual(len(bucket_matrix["feedback_input_sha256"]), 64)
        self.assertEqual(
            len(
                {
                    bucket_matrix["scalar_semantic_sha256"],
                    bucket_matrix["batched_semantic_sha256"],
                    bucket_matrix["reference_semantic_sha256"],
                }
            ),
            1,
        )
        self.assertEqual(facts["scalar_reference_action_mismatch_count"], 0)
        self.assertEqual(facts["batched_reference_action_mismatch_count"], 0)
        self.assertLessEqual(facts["max_logit_tolerance_ratio"], 1.0)
        self.assertLessEqual(facts["max_value_tolerance_ratio"], 1.0)
        self.assertLessEqual(facts["max_hidden_tolerance_ratio"], 1.0)
        self.assertEqual(
            set(facts["scalar_repeat_semantic_sha256"]),
            {facts["scalar_semantic_sha256"]},
        )
        self.assertEqual(
            set(facts["batched_repeat_semantic_sha256"]),
            {facts["scalar_semantic_sha256"]},
        )
        self.assertEqual(
            facts["reference_semantic_sha256"],
            facts["core_scalar_semantic_sha256"],
        )
        self.assertEqual(
            facts["core_batched_semantic_sha256"],
            facts["core_scalar_semantic_sha256"],
        )
        self.assertEqual(
            facts["ordered_merge_semantic_sha256"],
            facts["scalar_semantic_sha256"],
        )
        self.assertGreater(facts["scalar_median_elapsed_ns"], 0)
        self.assertGreater(facts["batched_median_elapsed_ns"], 0)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"]["ordered_merge_semantic_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ValueError,
            "production collector equivalence failed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        del forged["collector_equivalence"][
            "hidden_component_comparison_count"
        ]
        with self.assertRaisesRegex(
            ValueError,
            "production collector observations are malformed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"][
            "numeric_transition_comparison_count"
        ] = True
        with self.assertRaisesRegex(
            ValueError,
            "production collector numerical evidence is malformed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"]["unexpected_numeric_claim"] = 0
        with self.assertRaisesRegex(
            ValueError,
            "production collector observations are malformed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"]["max_value_tolerance_ratio"] = 1.01
        with self.assertRaisesRegex(
            ValueError,
            "production collector equivalence failed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"]["max_abs_value_error"] = float("nan")
        with self.assertRaisesRegex(
            ValueError,
            "production collector numerical evidence is malformed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"]["identity_mismatch_count"] = 1
        with self.assertRaisesRegex(
            ValueError,
            "production collector equivalence failed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["collector_equivalence"]["bootstrap_value_comparison_count"] = 0
        forged["collector_equivalence"]["max_abs_bootstrap_value_error"] = 0.0
        forged["collector_equivalence"][
            "max_bootstrap_value_tolerance_ratio"
        ] = 0.0
        for mode in ("scalar", "batched"):
            for sample in forged[mode]["samples"]:
                sample["collector_path"]["bootstrap_value_count"] = 0
        forged["scalar_collector_path"]["bootstrap_value_count"] = 0
        forged["batched_collector_path"]["bootstrap_value_count"] = 0
        with self.assertRaisesRegex(
            ValueError,
            "production collector equivalence failed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["batched_collector_path"][
            "fixed_batch_topology_mismatch_count"
        ] = 1
        with self.assertRaisesRegex(
            ValueError,
            "timed sample topology or bucket selection failed",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["batched_collector_path"][
            "fixed_batch_execution_bucket_histogram"
        ]["3"] = 1
        with self.assertRaisesRegex(
            ValueError,
            "execution bucket provenance drifted",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        forged["batched"]["samples"][1]["elapsed_ns"] = (
            forged["batched"]["samples"][0]["elapsed_ns"]
        )
        with self.assertRaisesRegex(
            ValueError,
            "timed collector sample binding drifted",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        padded_256_case = next(
            case
            for case in forged["equivalence"]["bucket_matrix"]["cases"]
            if case["active_rows"] == 129
        )
        padded_256_case["execution_rows"] = 320
        with self.assertRaisesRegex(
            ValueError,
            "bounded-bucket numerical provenance drifted",
        ):
            _fixed_batch_facts(forged)

        forged = copy.deepcopy(observed)
        del forged["equivalence"]["bucket_matrix"]
        with self.assertRaisesRegex(
            ValueError,
            "bounded-bucket numerical matrix is malformed",
        ):
            _fixed_batch_facts(forged)

    def test_d4_collector_numeric_corruption_cannot_hide_in_semantic_digest(
        self,
    ) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        model, tasks, _proof_seed_contract = _d4_phase_a_collection_inputs(
            shape={"worlds": 1, "rollout_ticks": 2, "initial_agents": 64}
        )
        scalar, _scalar_path, _scalar_elapsed = behavioral._run_d4_collector_mode(
            mode="scalar",
            model=model,
            tasks=tasks,
            rollout_workers=1,
        )
        batched, _batched_path, _batched_elapsed = (
            behavioral._run_d4_collector_mode(
                mode="batched",
                model=model,
                tasks=tasks,
                rollout_workers=1,
            )
        )
        corrupted_steps = list(batched.steps)
        none_index = next(
            index
            for index, step in enumerate(corrupted_steps)
            if step.bootstrap_value is None
        )
        bootstrap_index = next(
            index
            for index, step in enumerate(corrupted_steps)
            if index != none_index and step.bootstrap_value is not None
        )
        numeric_index = next(
            index
            for index in range(len(corrupted_steps))
            if index not in {none_index, bootstrap_index}
        )
        shape_index = next(
            index
            for index in range(len(corrupted_steps))
            if index not in {none_index, bootstrap_index, numeric_index}
        )
        none_step = corrupted_steps[none_index]
        corrupted_steps[none_index] = replace(none_step, bootstrap_value=0.0)
        bootstrap_step = corrupted_steps[bootstrap_index]
        assert bootstrap_step.bootstrap_value is not None
        corrupted_steps[bootstrap_index] = replace(
            bootstrap_step,
            bootstrap_value=bootstrap_step.bootstrap_value + 1.0,
        )
        numeric_step = corrupted_steps[numeric_index]
        corrupted_steps[numeric_index] = replace(
            numeric_step,
            hidden=tuple(value + 1.0 for value in numeric_step.hidden),
            logprob=numeric_step.logprob + 1.0,
            entropy=numeric_step.entropy + 1.0,
            value=numeric_step.value + 1.0,
        )
        shape_step = corrupted_steps[shape_index]
        corrupted_steps[shape_index] = replace(
            shape_step,
            hidden=shape_step.hidden[:-1],
        )

        class _BufferView:
            def __init__(self, steps: object) -> None:
                self.steps = tuple(steps)  # type: ignore[arg-type]

        equivalence = behavioral._d4_collector_equivalence(
            scalar,
            _BufferView(corrupted_steps),  # type: ignore[arg-type]
        )

        self.assertEqual(equivalence["semantic_mismatch_count"], 0)
        self.assertEqual(equivalence["identity_mismatch_count"], 0)
        self.assertEqual(equivalence["bootstrap_none_mismatch_count"], 1)
        self.assertEqual(equivalence["hidden_shape_mismatch_count"], 1)
        self.assertEqual(
            equivalence["numeric_transition_comparison_count"],
            len(scalar.steps) - 2,
        )
        self.assertEqual(
            equivalence["hidden_component_comparison_count"],
            (len(scalar.steps) - 2) * 256,
        )
        for field in (
            "max_input_hidden_tolerance_ratio",
            "max_logprob_tolerance_ratio",
            "max_entropy_tolerance_ratio",
            "max_value_tolerance_ratio",
            "max_bootstrap_value_tolerance_ratio",
        ):
            self.assertGreater(equivalence[field], 1.0)

    def test_d4_heterogeneous_mask_and_feedback_row_wiring_is_observable(
        self,
    ) -> None:
        proof_seed_contract = _d4_proof_seed_contract(
            shape={"worlds": 1, "rollout_ticks": 1, "initial_agents": 64}
        )
        _observations, masks, feedback, _genomes = _d4_input_rows(
            initial_agents=64,
            public_input_size=256,
            proof_seed_contract=proof_seed_contract,
        )
        self.assertGreater(len(set(masks)), 1)
        self.assertGreater(
            len({row.vector() for row in feedback}),
            1,
        )
        self.assertTrue(all(any(row.vector()) for row in feedback))

        original = TorchRecurrentPolicyCore.forward_fixed_batch

        def permute_mask_rows(
            core: TorchRecurrentPolicyCore,
            *args: object,
            **kwargs: object,
        ) -> object:
            mutable = list(args)
            mask_rows = tuple(mutable[1])  # type: ignore[arg-type]
            if len(mask_rows) > 1:
                mutable[1] = (*mask_rows[1:], mask_rows[0])
            return original(core, *mutable, **kwargs)

        def permute_feedback_rows(
            core: TorchRecurrentPolicyCore,
            *args: object,
            **kwargs: object,
        ) -> object:
            mutable = list(args)
            feedback_rows = tuple(mutable[2])  # type: ignore[arg-type]
            if len(feedback_rows) > 1:
                mutable[2] = (*feedback_rows[1:], feedback_rows[0])
            return original(core, *mutable, **kwargs)

        for field, corrupted in (
            ("action mask", permute_mask_rows),
            ("previous feedback", permute_feedback_rows),
        ):
            with (
                self.subTest(field=field),
                patch.object(
                    TorchRecurrentPolicyCore,
                    "forward_fixed_batch",
                    new=corrupted,
                ),
            ):
                equivalence = _run_d4_equivalence(
                    device="cpu",
                    shape={
                        "worlds": 1,
                        "rollout_ticks": 1,
                        "initial_agents": 4,
                    },
                )
            matrix = equivalence["bucket_matrix"]
            self.assertNotEqual(
                matrix["batched_semantic_sha256"],
                matrix["scalar_semantic_sha256"],
            )
            self.assertTrue(
                matrix["action_mismatch_count"] > 0
                or matrix["max_logit_tolerance_ratio"] > 1.0
                or matrix["max_value_tolerance_ratio"] > 1.0
                or matrix["max_hidden_tolerance_ratio"] > 1.0
            )

    def test_d4_accumulated_sub_tolerance_hidden_drift_is_detected(
        self,
    ) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        original = TorchRecurrentPolicyCore.forward_fixed_batch
        per_tick_hidden_perturbation = 5e-7

        def perturb_next_hidden(
            core: TorchRecurrentPolicyCore,
            *args: object,
            **kwargs: object,
        ) -> object:
            outputs = original(core, *args, **kwargs)
            return tuple(
                replace(
                    output,
                    next_hidden=tuple(
                        float(value) + per_tick_hidden_perturbation
                        for value in output.next_hidden
                    ),
                )
                for output in outputs
            )

        with (
            patch.object(
                behavioral,
                "_run_d4_bucket_matrix",
                return_value={},
            ),
            patch.object(
                TorchRecurrentPolicyCore,
                "forward_fixed_batch",
                new=perturb_next_hidden,
            ),
        ):
            one_tick = _run_d4_equivalence(
                device="cpu",
                shape={
                    "worlds": 1,
                    "rollout_ticks": 1,
                    "initial_agents": 1,
                },
            )
            accumulated = _run_d4_equivalence(
                device="cpu",
                shape={
                    "worlds": 1,
                    "rollout_ticks": 128,
                    "initial_agents": 1,
                },
            )

        self.assertLess(one_tick["max_hidden_tolerance_ratio"], 1.0)
        self.assertGreater(accumulated["max_hidden_tolerance_ratio"], 1.0)
        self.assertGreater(
            accumulated["max_batched_reference_hidden_tolerance_ratio"],
            1.0,
        )

    def test_d4_later_timed_batched_fallback_cannot_influence_median(
        self,
    ) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        original = behavioral._run_d4_collector_mode
        batched_call_count = 0

        def fallback_on_second_batched_sample(
            *,
            mode: str,
            model: object,
            tasks: object,
            rollout_workers: int,
        ) -> object:
            nonlocal batched_call_count
            actual_mode = mode
            if mode == "batched":
                batched_call_count += 1
                if batched_call_count == 2:
                    actual_mode = "scalar"
            return original(
                mode=actual_mode,
                model=model,
                tasks=tasks,
                rollout_workers=rollout_workers,
            )

        with patch.object(
            behavioral,
            "_run_d4_collector_mode",
            new=fallback_on_second_batched_sample,
        ):
            observed = _run_fixed_batch_probe(
                device="cpu",
                shape={"worlds": 1, "rollout_ticks": 1, "initial_agents": 64},
                repeat_count=2,
            )

        self.assertEqual(
            len(
                {
                    *observed["scalar"]["semantic_sha256"],
                    *observed["batched"]["semantic_sha256"],
                }
            ),
            1,
        )
        self.assertGreater(
            observed["batched"]["samples"][0]["collector_path"][
                "fixed_batch_step_count"
            ],
            0,
        )
        self.assertEqual(
            observed["batched"]["samples"][1]["collector_path"][
                "fixed_batch_step_count"
            ],
            0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "batched timed sample topology or bucket selection failed",
        ):
            _fixed_batch_facts(observed)

    def test_d4_collection_inputs_are_proof_only_with_canonical_identities(
        self,
    ) -> None:
        from evolution_sim.mind import open_ecology_phase_a

        shape = {"worlds": 2, "rollout_ticks": 2, "initial_agents": 64}
        with patch.object(
            open_ecology_phase_a,
            "build_phase_a_run_components",
            side_effect=AssertionError("scientific training schedule accessed"),
        ):
            _model, tasks, proof = _d4_phase_a_collection_inputs(shape=shape)

        proof_seeds = OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_proof"]
        self.assertEqual([task.seed_role for task in tasks], ["open_ecology_proof"] * 2)
        self.assertEqual(
            [task.environment_seed for task in tasks],
            list(proof_seeds[:2]),
        )
        self.assertEqual(
            [task.task_id for task in tasks],
            proof["task_ids"],
        )
        self.assertEqual(
            [task.policy_sampling_identity for task in tasks],
            proof["policy_sampling_identities"],
        )
        self.assertEqual(
            [task.policy_sampling_seed for task in tasks],
            proof["policy_sampling_seeds"],
        )
        self.assertEqual(proof["scientific_training_seed_access_count"], 0)
        self.assertEqual(proof["scientific_selection_seed_access_count"], 0)
        scientific_seeds = {
            *OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_train"],
            *OPEN_ECOLOGY_SEED_REGISTRY["open_ecology_selection"],
        }
        self.assertTrue(scientific_seeds.isdisjoint(proof["environment_seeds"]))

    def test_d4_producer_and_verifier_reject_live_determinism_drift(self) -> None:
        import torch
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        deterministic_before = torch.are_deterministic_algorithms_enabled()
        warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
        cudnn_benchmark_before = torch.backends.cudnn.benchmark
        cudnn_deterministic_before = torch.backends.cudnn.deterministic
        cudnn_tf32_before = torch.backends.cudnn.allow_tf32
        matmul_tf32_before = torch.backends.cuda.matmul.allow_tf32
        precision_before = torch.get_float32_matmul_precision()
        dtype_before = torch.get_default_dtype()
        workspace_was_set = "CUBLAS_WORKSPACE_CONFIG" in os.environ
        workspace_before = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

        def drift_after_probe(**_kwargs: object) -> dict[str, object]:
            torch.use_deterministic_algorithms(False)
            return {}

        def drift_while_loading(*_args: object, **_kwargs: object) -> object:
            torch.use_deterministic_algorithms(False)
            return {}, {}

        try:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "d4"
                with (
                    patch.object(
                        readiness,
                        "_require_live_source_twice",
                        return_value=None,
                    ),
                    patch.object(
                        behavioral,
                        "_run_fixed_batch_probe",
                        new=drift_after_probe,
                    ),
                    self.assertRaisesRegex(Exception, "contract drifted"),
                ):
                    produce_fixed_batch_equivalence_and_speed_report(
                        self.preregistration,
                        output_directory=destination,
                    )
                self.assertFalse(destination.exists())

            with (
                patch.object(
                    behavioral,
                    "_load_verified_raw_observations",
                    new=drift_while_loading,
                ),
                self.assertRaisesRegex(Exception, "contract drifted"),
            ):
                verify_fixed_batch_equivalence_and_speed_report(
                    Path("unused-report.json"),
                    self.preregistration,
                    object(),
                )
        finally:
            torch.use_deterministic_algorithms(
                deterministic_before,
                warn_only=warn_only_before,
            )
            torch.backends.cudnn.benchmark = cudnn_benchmark_before
            torch.backends.cudnn.deterministic = cudnn_deterministic_before
            torch.backends.cudnn.allow_tf32 = cudnn_tf32_before
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32_before
            torch.set_float32_matmul_precision(precision_before)
            torch.set_default_dtype(dtype_before)
            if workspace_was_set:
                assert workspace_before is not None
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace_before
            else:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    def test_d4_verifier_rejects_pre_identity_probe_spec_transplant(self) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        deterministic_contract = behavioral._sealed_d4_deterministic_cuda_contract()
        legacy_spec = _d4_probe_spec(
            device="cuda",
            rollout_workers=int(
                self.preregistration["runtime_contract"]["rollout_workers"]
            ),
            deterministic_cuda_contract=deterministic_contract,
        )
        del legacy_spec["model_contract_version"]
        del legacy_spec["numeric_kernel"]
        with (
            patch.object(
                behavioral,
                "_configure_and_verify_d4_deterministic_cuda_contract",
                return_value=deterministic_contract,
            ),
            patch.object(
                behavioral,
                "_verify_current_d4_deterministic_cuda_contract",
                return_value=deterministic_contract,
            ),
            patch.object(
                behavioral,
                "_load_verified_raw_observations",
                return_value=(legacy_spec, {"device": "cuda"}),
            ),
            self.assertRaisesRegex(Exception, "D4 probe specification drifted"),
        ):
            verify_fixed_batch_equivalence_and_speed_report(
                Path("unused-report.json"),
                self.preregistration,
                object(),
            )

    def test_d4_non_argmax_batch_corruption_is_detected_numerically(self) -> None:
        original = TorchRecurrentPolicyCore.forward_fixed_batch

        def corrupt_logits(
            core: TorchRecurrentPolicyCore,
            *args: object,
            **kwargs: object,
        ) -> object:
            outputs = original(core, *args, **kwargs)
            return tuple(
                replace(
                    output,
                    logits=tuple(float(value) + 0.01 for value in output.logits),
                )
                for output in outputs
            )

        with patch.object(
            TorchRecurrentPolicyCore,
            "forward_fixed_batch",
            new=corrupt_logits,
        ):
            facts = _fixed_batch_facts(
                _run_fixed_batch_probe(
                    device="cpu",
                    shape={
                        "worlds": 1,
                        "rollout_ticks": 2,
                        "initial_agents": 64,
                    },
                    repeat_count=2,
                )
            )
        self.assertEqual(facts["action_mismatch_count"], 0)
        self.assertEqual(
            facts["scalar_semantic_sha256"],
            facts["batched_semantic_sha256"],
        )
        self.assertGreater(facts["max_logit_tolerance_ratio"], 1.0)

    def test_d4_hardcoded_core_cannot_hide_behind_learned_source_label(self) -> None:
        def constant_output(hidden: object) -> RecurrentCoreOutput:
            logits = [-1_000_000.0] * len(ACTION_NAMES)
            logits[ACTION_NAMES.index("stay")] = 1.0
            return RecurrentCoreOutput(
                logits=tuple(logits),
                value=0.0,
                next_hidden=tuple(0.25 for _value in hidden),
            )

        def hardcoded_scalar(
            core: TorchRecurrentPolicyCore,
            observation: object,
            current_action_mask: object,
            previous_feedback: PreviousPublicFeedback,
            hidden: object,
            *,
            genome_values: object = None,
        ) -> RecurrentCoreOutput:
            del (
                core,
                observation,
                current_action_mask,
                previous_feedback,
                genome_values,
            )
            return constant_output(hidden)

        def hardcoded_batch(
            core: TorchRecurrentPolicyCore,
            observations: object,
            current_action_masks: object,
            previous_feedback: object,
            hidden: object,
            *,
            batch_capacity: int,
            execution_batch_rows: int | None = None,
            genome_values: object = None,
        ) -> tuple[RecurrentCoreOutput, ...]:
            del (
                core,
                current_action_masks,
                previous_feedback,
                batch_capacity,
                execution_batch_rows,
                genome_values,
            )
            return tuple(constant_output(row) for row in hidden[: len(observations)])

        with (
            patch.object(
                TorchRecurrentPolicyCore,
                "forward_step",
                new=hardcoded_scalar,
            ),
            patch.object(
                TorchRecurrentPolicyCore,
                "forward_fixed_batch",
                new=hardcoded_batch,
            ),
        ):
            d2_facts = _runtime_genome_facts(_run_runtime_genome_probe())
            readiness._SEMANTIC_VALIDATORS["runtime_genome_and_action_source"](
                d2_facts,
                self.preregistration,
                None,
            )
            with self.assertRaisesRegex(
                ValueError,
                "bounded-bucket physical recurrent shape drifted",
            ):
                _fixed_batch_facts(
                    _run_fixed_batch_probe(
                        device="cpu",
                        shape={
                            "worlds": 1,
                            "rollout_ticks": 2,
                            "initial_agents": 64,
                        },
                        repeat_count=2,
                    )
                )
        self.assertEqual(d2_facts["heuristic_action_source_count"], 0)

    def test_d4_authority_rejects_cpu_and_malformed_timing_samples(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(Exception, "requires the CUDA runtime"),
        ):
            produce_fixed_batch_equivalence_and_speed_report(
                self.preregistration,
                output_directory=Path(temporary) / "d4",
                device="cpu",
            )

        observed = _run_fixed_batch_probe(
            device="cpu",
            shape={"worlds": 1, "rollout_ticks": 2, "initial_agents": 64},
            repeat_count=2,
        )
        observed["scalar"]["elapsed_ns"].pop()
        with self.assertRaisesRegex(ValueError, "timing sample"):
            _fixed_batch_facts(observed)

    def test_source_before_probe_and_source_after_before_publish(self) -> None:
        from evolution_sim.mind import (
            open_ecology_phase_a_behavioral_evidence as behavioral,
        )

        events: list[str] = []
        original_probe = behavioral._run_cross_surface_probe
        original_rename = readiness._rename_path_no_replace

        def source_check(
            _preregistration: object,
            *,
            before: bool,
        ) -> None:
            events.append("source_before" if before else "source_after")

        def probe() -> dict[str, object]:
            events.append("probe")
            return original_probe()

        def rename(source: Path, destination: Path) -> None:
            events.append("rename")
            original_rename(source, destination)

        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary).resolve() / "d1"
            with (
                patch.object(
                    readiness,
                    "_require_live_source_twice",
                    new=source_check,
                ),
                patch.object(
                    behavioral,
                    "_run_cross_surface_probe",
                    new=probe,
                ),
                patch.object(
                    readiness,
                    "_rename_path_no_replace",
                    new=rename,
                ),
            ):
                produce_cross_surface_and_self_echo_report(
                    self.preregistration,
                    output_directory=bundle,
                )
        self.assertEqual(
            events,
            ["source_before", "probe", "source_after", "rename"],
        )

    def test_source_after_failure_retains_unpublished_pending_bundle(self) -> None:
        def source_check(
            _preregistration: object,
            *,
            before: bool,
        ) -> None:
            if not before:
                raise readiness._contract().OpenEcologyPhaseAError("source changed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bundle = root / "d1"
            with (
                patch.object(
                    readiness,
                    "_require_live_source_twice",
                    new=source_check,
                ),
                self.assertRaisesRegex(Exception, "source changed"),
            ):
                produce_cross_surface_and_self_echo_report(
                    self.preregistration,
                    output_directory=bundle,
                )
            self.assertFalse(bundle.exists())
            pending = list(root.glob(".d1.pending-*"))
            self.assertEqual(len(pending), 1)
            self.assertTrue((pending[0] / "report.json").is_file())

    def test_d2_d3_bundle_verifiers_reexecute_real_probes(self) -> None:
        cases = (
            (
                "d2",
                produce_runtime_genome_and_action_source_report,
                verify_runtime_genome_and_action_source_report,
            ),
            (
                "d3",
                produce_critic_gradient_and_density_schedule_report,
                verify_critic_gradient_and_density_schedule_report,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.object(
                readiness,
                "_require_live_source_twice",
                return_value=None,
            ):
                for name, producer, verifier in cases:
                    with self.subTest(name=name):
                        bundle = root / name
                        report = producer(
                            self.preregistration,
                            output_directory=bundle,
                        )
                        authority = verifier(
                            bundle / "report.json",
                            self.preregistration,
                            None,
                        )
                        self.assertEqual(authority["facts"], report["facts"])

    def test_d1_bundle_verifier_reexecutes_and_rejects_raw_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bundle = root / "d1"
            with patch.object(
                readiness,
                "_require_live_source_twice",
                return_value=None,
            ):
                report = produce_cross_surface_and_self_echo_report(
                    self.preregistration,
                    output_directory=bundle,
                )
                authority = verify_cross_surface_and_self_echo_report(
                    bundle / "report.json",
                    self.preregistration,
                    None,
                )
            self.assertEqual(authority["facts"], report["facts"])

            observations = bundle / "raw" / "observations.json"
            payload = json.loads(observations.read_text(encoding="ascii"))
            payload["self_echo_cases"][0]["emission_event_count"] = 0
            observations.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                Exception,
                "reference or bytes drifted",
            ):
                verify_cross_surface_and_self_echo_report(
                    bundle / "report.json",
                    self.preregistration,
                    None,
                )


def _start_hostile_validator(code: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )


def _force_process_group_closed(process: subprocess.Popen[bytes]) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2.0)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _assert_process_gone(
    case: unittest.TestCase,
    child_pid_path: Path,
) -> None:
    case.assertTrue(child_pid_path.is_file(), "validator descendant did not start")
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 3.0
    state = ""
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ("ps", "-p", str(child_pid), "-o", "stat="),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        state = completed.stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    case.assertTrue(
        not state or state.startswith("Z"),
        f"D1 viewer validator descendant survived cleanup: {state}",
    )


if __name__ == "__main__":
    unittest.main()
