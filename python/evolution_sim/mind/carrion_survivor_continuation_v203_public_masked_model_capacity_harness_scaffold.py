from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import fields
import hashlib
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract as v202,
)
from evolution_sim.mind.candidate_campaign import write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest


M3_CARRION_SURVIVOR_CONTINUATION_V203_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_SCAFFOLD_SCHEMA_VERSION = "m3_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold_report_v1"
M3_CARRION_SURVIVOR_CONTINUATION_V203_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_SCAFFOLD_POLICY = "diagnostics_only_m3_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold_v1"

DEFAULT_V202_REPORT_PATH = v202.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v203-carrion-survivor-continuation-public-masked-model-capacity-harness-scaffold.json"
)

EXPECTED_V202_REPORT_EXACT_DIGEST = (
    "5776fd258e7a6ae5c2b650c88f868c74cbfa0fc13c76a0f29fc23e6ab96a1a8a"
)
EXPECTED_V202_CLASSIFICATION = "m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_ready_for_scaffold_no_training"
EXPECTED_V202_ROUTE = v202.HARNESS_SCAFFOLD_ROUTE
TRAINING_SLICES_CONSUMED = 3
TRAINING_SLICE_BUDGET = 10

LINEAGE_REPAIR_ROUTE = "v204_v202_scaffold_lineage_repair_no_training"
SCAFFOLD_REPAIR_ROUTE = (
    "v204_public_recurrent_ippo_scaffold_contract_repair_no_training"
)
FUTURE_RECURRENT_IPPO_TRAINING_ROUTE = (
    "v204_public_masked_recurrent_ippo_training_slice_4_opt_in"
)


def run_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold(
    *,
    v202_report_path: str | Path = DEFAULT_V202_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v202_report_exact_digest: str = EXPECTED_V202_REPORT_EXACT_DIGEST,
) -> dict[str, object]:
    """Audit the executable recurrent-IPPO scaffold without running training."""

    v202_report, load_error = _load_optional_json_report(v202_report_path)
    source_validation = validate_v203_source_pin(
        v202_report=v202_report,
        load_error=load_error,
        expected_v202_report_exact_digest=expected_v202_report_exact_digest,
    )
    surface_audit = public_recurrent_ippo_scaffold_surface_audit()
    route_decision = route_decision_for_v203(
        source_validation=source_validation,
        scaffold_surface_audit=surface_audit,
    )
    classification = classification_for_v203(
        source_validation=source_validation,
        scaffold_surface_audit=surface_audit,
    )
    report: dict[str, object] = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V203_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_SCAFFOLD_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V203_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_SCAFFOLD_POLICY
        ),
        "selected_input_route": EXPECTED_V202_ROUTE,
        "inputs": {
            "v202_report": str(v202_report_path),
            "expected_v202_report_exact_digest": (expected_v202_report_exact_digest),
        },
        "historical_evidence_checkpoint": historical_evidence_checkpoint(),
        "source_pin_validation": source_validation,
        "public_recurrent_ippo_scaffold_surface_audit": surface_audit,
        "controls_and_future_training_plan": controls_and_future_training_plan(),
        "budget_state": {
            "training_slices_consumed": TRAINING_SLICES_CONSUMED,
            "training_slice_budget": TRAINING_SLICE_BUDGET,
            "campaign_budget": (f"{TRAINING_SLICES_CONSUMED}/{TRAINING_SLICE_BUDGET}"),
            "slice_4_available_but_not_started": True,
            "slice_4_training_authorized_by_v203": False,
        },
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "public_masked_recurrent_actor_critic_scaffold",
                "parameter_shared_recurrent_ippo_scaffold",
                "executable_contract_probes",
                "no_training",
                "no_slice_4_consumption",
                "no_training_artifact",
                "no_runtime_artifact",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_promotion",
            ],
        },
        **lifecycle_flags(),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v203_source_pin(
    *,
    v202_report: Mapping[str, object],
    load_error: str | None,
    expected_v202_report_exact_digest: str,
) -> dict[str, object]:
    exact = exact_digest_validation_report(v202_report)
    route = _mapping(v202_report.get("route_decision"))
    classification = _mapping(v202_report.get("classification"))
    budget = _mapping(v202_report.get("budget_state"))
    harness = _mapping(v202_report.get("public_masked_model_capacity_harness_contract"))
    checks = {
        "v202_report_loaded": load_error is None,
        "v202_schema_matches": v202_report.get("schema_version")
        == v202.M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_SCHEMA_VERSION,
        "v202_policy_matches": v202_report.get("policy")
        == v202.M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_POLICY,
        "v202_exact_digest_self_valid": exact.get("passed") is True,
        "v202_exact_digest_matches_expected": v202_report.get("exact_digest")
        == expected_v202_report_exact_digest,
        "v202_classification_matches": classification.get("primary")
        == EXPECTED_V202_CLASSIFICATION,
        "v202_selected_route_matches": route.get("selected_route")
        == EXPECTED_V202_ROUTE,
        "v202_recommended_route_matches": route.get("recommended_next_route")
        == EXPECTED_V202_ROUTE,
        "v202_source_pins_passed": _mapping(
            v202_report.get("source_pin_validation")
        ).get("passed")
        is True,
        "v202_required_facts_passed": _mapping(
            v202_report.get("v201_fact_assessment")
        ).get("passed")
        is True,
        "v202_public_contract_complete": harness.get("contract_complete_for_scaffold")
        is True,
        "v202_public_contract_forbids_training": _mapping(
            harness.get("lifecycle_limits")
        ).get("training_allowed")
        is False,
        "v202_budget_is_3_of_10": budget.get("training_slices_consumed") == 3
        and budget.get("training_slice_budget") == 10,
        "v202_lifecycle_closed": _v202_lifecycle_closed(v202_report),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v203_v202_source_pin_validation_v1",
        "passed": not failures,
        "load_error": load_error,
        "expected_v202_report_exact_digest": expected_v202_report_exact_digest,
        "observed_v202_report_exact_digest": v202_report.get("exact_digest"),
        "computed_v202_report_exact_digest": exact.get("computed_exact_digest"),
        "expected_v202_classification": EXPECTED_V202_CLASSIFICATION,
        "expected_v202_route": EXPECTED_V202_ROUTE,
        "checks": checks,
        "failures": failures,
    }


def public_recurrent_ippo_scaffold_surface_audit() -> dict[str, object]:
    """Run deterministic, in-memory probes over the exported scaffold surfaces.

    This is executable contract evidence, not a replacement for an explicit
    contract export from every module. It creates no dataset, policy file,
    training artifact, runtime artifact, or simulator-side integration.
    """

    try:
        import torch

        from evolution_sim.config.schema import WorldConfig
        from evolution_sim.env.runtime.action_contract import ACTION_NAMES
        from evolution_sim.env.runtime.state import RunMode
        from evolution_sim.env.runtime.trajectory import REWARD_COMPONENT_BOUNDS
        from evolution_sim.env.world import SimulationWorld
        from evolution_sim.mind.recurrent_actor_critic import (
            ACTION_COUNT,
            LEARNED_ENCODER_INPUT_SIZE,
            PREVIOUS_PUBLIC_FEEDBACK_SIZE,
            PUBLIC_INPUT_SIZE,
            ActionMaskError,
            PreviousPublicFeedbackInput,
            PublicRecurrentActorCritic,
            RecurrentActorCriticConfig,
            previous_public_feedback_tensor,
            recurrent_actor_critic_contract,
            strict_action_mask_tensor,
        )
        from evolution_sim.mind.recurrent_artifact import (
            RECURRENT_ARTIFACT_SCHEMA_VERSION,
            RecurrentArtifactError,
            build_recurrent_artifact,
            model_from_recurrent_artifact,
            validate_recurrent_artifact,
        )
        from evolution_sim.mind.recurrent_policy import (
            DeterministicPublicRecurrentPolicy,
        )
        from evolution_sim.mind.recurrent_ppo import (
            PPOTrainingSequence,
            RecurrentPPOConfig,
            recurrent_ppo_contract,
        )
        from evolution_sim.mind.recurrent_rollout import (
            RECURRENT_LEARNED_INPUT_VECTOR_SIZE,
            RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE,
            RECURRENT_ROLLOUT_ACTION_SOURCE,
            PreviousPublicFeedback,
            RecurrentOnPolicyCollector,
            RecurrentRolloutBuffer,
            RecurrentRolloutStep,
            TorchRecurrentPolicyCore,
        )
        from evolution_sim.mind.recurrent_seed_registry import (
            CANONICAL_SEED_REGISTRY_SHA256,
            GENERATED_SEED_REGISTRY_SHA256,
            RECURRENT_SEED_REGISTRY,
            canonical_seed_registry_json,
            recurrent_seed_registry_contract,
            validate_recurrent_seed_registry,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "policy": "m3_carrion_survivor_continuation_v203_public_recurrent_ippo_scaffold_surface_audit_v1",
            "evidence_mode": "exported_contracts_plus_executable_no_training_probes",
            "passed": False,
            "checks": {"optional_mind_ml_surfaces_imported": False},
            "failures": ["optional_mind_ml_surfaces_imported"],
            "import_error": f"{type(exc).__name__}: {exc}",
            "training_ran": False,
        }

    torch.set_num_threads(1)
    model_config = RecurrentActorCriticConfig(
        encoder_size=8,
        hidden_size=8,
        recurrent_layers=1,
    )
    actor_contract = recurrent_actor_critic_contract(model_config)
    ppo_contract = recurrent_ppo_contract(
        RecurrentPPOConfig(
            update_epochs=1,
            sequence_minibatch_size=1,
            tbptt_steps=4,
            learner_seed=17,
        )
    )
    seed_contract = recurrent_seed_registry_contract()
    validate_recurrent_seed_registry(RECURRENT_SEED_REGISTRY)

    model = PublicRecurrentActorCritic(
        model_config,
        initialization_seed=203,
    ).to(device="cpu", dtype=torch.float32)
    model.eval()
    observation = torch.zeros(PUBLIC_INPUT_SIZE, dtype=torch.float32)
    action_mask_mapping = {
        action: index == 7 for index, action in enumerate(ACTION_NAMES)
    }
    action_mask = strict_action_mask_tensor(action_mask_mapping)
    previous_feedback = previous_public_feedback_tensor(
        PreviousPublicFeedbackInput.zero()
    )
    initial_state = model.initial_state(1)
    with torch.no_grad():
        first = model.act(
            observation,
            action_mask,
            previous_feedback,
            recurrent_state=initial_state,
            deterministic=True,
        )
        second = model.act(
            observation,
            action_mask,
            previous_feedback,
            recurrent_state=first.next_state,
            deterministic=True,
        )
    hard_mask_selected_only_legal_action = int(first.actions.item()) == 7
    hard_mask_invalid_logits_are_negative_infinity = bool(
        torch.isneginf(first.masked_logits[0, ~action_mask]).all().item()
    )
    empty_mask_failed_closed = False
    try:
        strict_action_mask_tensor({action: False for action in ACTION_NAMES})
    except ActionMaskError:
        empty_mask_failed_closed = True
    recurrent_state_updated = not bool(
        torch.equal(initial_state, first.next_state)
    ) and not bool(torch.equal(first.next_state, second.next_state))

    terminated_gae_ok, truncated_gae_ok = _probe_gae_contract(
        action_names=tuple(ACTION_NAMES),
        reward_component_names=tuple(REWARD_COMPONENT_BOUNDS),
        recurrent_rollout_step_type=RecurrentRolloutStep,
        recurrent_rollout_buffer_type=RecurrentRolloutBuffer,
        previous_feedback_type=PreviousPublicFeedback,
    )
    collector = RecurrentOnPolicyCollector(TorchRecurrentPolicyCore(model))
    collector.start_world(
        world_id="v203-finalized-alignment-probe",
        seed=203,
        rollout_ticks=1,
    )
    SimulationWorld(
        WorldConfig(seed=203, max_ticks=2),
        policy=collector,
    ).run(mode=RunMode.SUMMARY_ONLY, record_trajectory=True)
    collector.finish_world()
    collected_steps = collector.buffer.steps
    finalized_alignment_probe_passed = bool(collected_steps) and all(
        step.tick == 0
        and step.action_mask[step.action_index]
        and step.requested_action == ACTION_NAMES[step.action_index]
        and step.world_id == "v203-finalized-alignment-probe"
        and (step.terminated or step.truncated)
        for step in collected_steps
    )

    artifact = build_recurrent_artifact(
        model,
        training_config={
            "probe": "v203_in_memory_scaffold_only",
            "training_ran": False,
        },
        seed_registry_digest=CANONICAL_SEED_REGISTRY_SHA256,
        source_commit="0000000000000000000000000000000000000000",
        data_metadata={"dataset_created": False, "rows": 0},
        run_metadata={"persisted": False, "purpose": "contract_probe"},
        learner_seed=203,
        learner_device="cpu_contract_probe",
    )
    validate_recurrent_artifact(artifact)
    loaded_model = model_from_recurrent_artifact(artifact)
    with torch.no_grad():
        loaded = loaded_model.act(
            observation,
            action_mask,
            previous_feedback,
            recurrent_state=loaded_model.initial_state(1),
            deterministic=True,
        )
    artifact_roundtrip_replay_equal = (
        torch.equal(first.actions, loaded.actions)
        and torch.equal(first.raw_logits, loaded.raw_logits)
        and torch.equal(first.values, loaded.values)
        and torch.equal(first.next_state, loaded.next_state)
    )
    tensor_records = artifact.get("tensors")
    tensor_records = tensor_records if isinstance(tensor_records, list) else []
    serialization = _mapping(artifact.get("serialization"))
    artifact_digests_present = (
        isinstance(artifact.get("artifact_sha256"), str)
        and len(str(artifact.get("artifact_sha256"))) == 64
        and isinstance(serialization.get("whole_model_sha256"), str)
        and len(str(serialization.get("whole_model_sha256"))) == 64
        and bool(tensor_records)
        and all(
            isinstance(record, Mapping)
            and isinstance(record.get("sha256"), str)
            and len(str(record.get("sha256"))) == 64
            for record in tensor_records
        )
    )
    tampered = copy.deepcopy(artifact)
    tampered_records = tampered.get("tensors")
    assert isinstance(tampered_records, list) and tampered_records
    assert isinstance(tampered_records[0], dict)
    tampered_records[0]["sha256"] = "0" * 64
    artifact_tamper_failed_closed = False
    try:
        validate_recurrent_artifact(tampered)
    except RecurrentArtifactError:
        artifact_tamper_failed_closed = True

    rollout_fields = {field.name for field in fields(RecurrentRolloutStep)}
    finalized_alignment_fields = {
        "observation",
        "previous_feedback",
        "action_mask",
        "hidden",
        "action_index",
        "requested_action",
        "logprob",
        "value",
        "reward",
        "reward_components",
        "resolved_action",
        "resolution_action_mask",
        "resolution_action_valid",
        "terminated",
        "truncated",
        "bootstrap_value",
    }
    ppo_sequence_fields = {field.name for field in fields(PPOTrainingSequence)}
    required_ppo_fields = {
        "world_id",
        "observations",
        "action_masks",
        "previous_feedback",
        "recurrent_states",
        "actions",
        "old_log_probs",
        "old_values",
        "advantages",
        "return_targets",
        "episode_starts",
    }
    seed_payload = {
        role: tuple(seeds) for role, seeds in RECURRENT_SEED_REGISTRY.items()
    }
    flattened_seeds = [seed for seeds in seed_payload.values() for seed in seeds]
    legacy = set(seed_payload["diagnostic_red_team"])
    fresh = {
        seed
        for role, seeds in seed_payload.items()
        if role != "diagnostic_red_team"
        for seed in seeds
    }

    checks = {
        "optional_mind_ml_surfaces_imported": True,
        "public_ecological_input_is_541": PUBLIC_INPUT_SIZE == 541,
        "stable_action_space_is_20": ACTION_COUNT == 20 and len(ACTION_NAMES) == 20,
        "previous_public_feedback_is_43": PREVIOUS_PUBLIC_FEEDBACK_SIZE == 43
        and RECURRENT_PUBLIC_FEEDBACK_VECTOR_SIZE == 43,
        "learned_input_parity_is_604": LEARNED_ENCODER_INPUT_SIZE == 604
        and RECURRENT_LEARNED_INPUT_VECTOR_SIZE == 604
        and _mapping(actor_contract.get("architecture")).get(
            "learned_encoder_input_size"
        )
        == 604,
        "learned_inputs_include_current_mask_and_prior_finalized_feedback": _mapping(
            actor_contract.get("architecture")
        ).get("learned_encoder_inputs")
        == [
            "current_ecological_observation_541",
            "current_action_mask_20",
            "previous_requested_action_one_hot_20",
            "previous_resolved_action_one_hot_20",
            "previous_resolution_valid_1",
            "previous_moved_1",
            "previous_public_reward_total_normalized_1",
        ],
        "hard_mask_selected_only_legal_action": (hard_mask_selected_only_legal_action),
        "hard_mask_invalid_logits_are_negative_infinity": (
            hard_mask_invalid_logits_are_negative_infinity
        ),
        "empty_action_mask_fails_closed": empty_mask_failed_closed,
        "recurrent_state_shape_and_update_work": tuple(initial_state.shape) == (1, 1, 8)
        and recurrent_state_updated,
        "per_agent_state_lifecycle_declared": _mapping(
            actor_contract.get("recurrent_state")
        )
        == {
            "scope": "one_hidden_state_per_agent",
            "birth": "zero_state",
            "episode_or_world_reset": "zero_state",
            "death": "discard_state",
            "birth_or_world_reset_previous_feedback": "all_zero",
        },
        "finalized_transition_alignment_fields_present": (
            finalized_alignment_fields <= rollout_fields
        ),
        "collector_alignment_boundary_is_executable": all(
            callable(getattr(RecurrentOnPolicyCollector, method, None))
            for method in (
                "start_world",
                "decide",
                "observe_transition",
                "finish_world",
            )
        ),
        "real_world_finalized_transition_alignment_probe_passed": (
            finalized_alignment_probe_passed
        ),
        "terminated_gae_zero_bootstrap_probe_passed": terminated_gae_ok,
        "truncated_gae_frozen_bootstrap_probe_passed": truncated_gae_ok,
        "ppo_ordered_sequence_contract_complete": required_ppo_fields
        == ppo_sequence_fields,
        "ppo_uses_stored_observation_masks_and_frozen_statistics": _mapping(
            ppo_contract.get("likelihood_contract")
        )
        == {
            "action_masks": "stored_observation_time_masks_only",
            "old_log_probs": "frozen_behavior_policy_values",
            "old_values": "frozen_behavior_policy_values",
            "resolution_masks_used_for_likelihood": False,
        },
        "ppo_primary_is_parameter_shared_recurrent_ippo": ppo_contract.get("algorithm")
        == "parameter_shared_recurrent_ippo_clipped_ppo",
        "ppo_feed_forward_history_ablation_exists": _mapping(
            ppo_contract.get("history")
        ).get("feed_forward_ablation")
        == "reset_hidden_before_every_decision",
        "ppo_forbids_hardcoded_actions_and_private_features": ppo_contract.get(
            "hardcoded_action_selection"
        )
        is False
        and ppo_contract.get("private_world_features") is False,
        "artifact_schema_is_json_tensor_digest_bound": artifact.get("schema_version")
        == RECURRENT_ARTIFACT_SCHEMA_VERSION
        and serialization.get("format") == "json_tensor_artifact_v1",
        "artifact_per_tensor_model_and_payload_digests_present": (
            artifact_digests_present
        ),
        "artifact_roundtrip_replay_is_exact": artifact_roundtrip_replay_equal,
        "artifact_tamper_fails_closed": artifact_tamper_failed_closed,
        "fresh_seed_registry_digest_matches": GENERATED_SEED_REGISTRY_SHA256
        == CANONICAL_SEED_REGISTRY_SHA256
        and stable_payload_digest(seed_contract.get("seeds"))
        == CANONICAL_SEED_REGISTRY_SHA256,
        "fresh_seed_roles_are_globally_disjoint": len(flattened_seeds)
        == len(set(flattened_seeds)),
        "legacy_diagnostics_are_excluded_from_fresh_roles": not (legacy & fresh),
        "selection_validation_and_lockbox_roles_exist": all(
            role in seed_payload for role in ("selection", "validation", "lockbox")
        ),
        "deterministic_policy_adapter_is_present_and_opt_in": all(
            callable(getattr(DeterministicPublicRecurrentPolicy, method, None))
            for method in ("decide", "observe_transition", "reset_world")
        )
        and RECURRENT_ROLLOUT_ACTION_SOURCE == "learned_recurrent_on_policy",
        "seed_registry_canonical_bytes_match_digest": hashlib.sha256(
            canonical_seed_registry_json()
        ).hexdigest()
        == CANONICAL_SEED_REGISTRY_SHA256,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v203_public_recurrent_ippo_scaffold_surface_audit_v1",
        "evidence_mode": "exported_contracts_plus_executable_no_training_probes",
        "audited_surfaces": [
            "recurrent_actor_critic",
            "recurrent_rollout",
            "recurrent_artifact",
            "recurrent_seed_registry",
            "recurrent_policy",
            "recurrent_ppo",
        ],
        "contract_summary": {
            "public_input_size": PUBLIC_INPUT_SIZE,
            "current_action_mask_size": ACTION_COUNT,
            "previous_public_feedback_size": PREVIOUS_PUBLIC_FEEDBACK_SIZE,
            "learned_encoder_input_size": LEARNED_ENCODER_INPUT_SIZE,
            "architecture": _mapping(actor_contract.get("architecture")),
            "selection": _mapping(actor_contract.get("selection")),
            "recurrent_state": _mapping(actor_contract.get("recurrent_state")),
            "rollout_alignment_fields": sorted(finalized_alignment_fields),
            "ppo_algorithm": ppo_contract.get("algorithm"),
            "ppo_gae": _mapping(ppo_contract.get("gae")),
            "ppo_sequence_batching": _mapping(ppo_contract.get("sequence_batching")),
            "artifact_schema_version": RECURRENT_ARTIFACT_SCHEMA_VERSION,
            "seed_registry_sha256": CANONICAL_SEED_REGISTRY_SHA256,
            "seed_role_counts": _mapping(seed_contract.get("role_counts")),
            "runtime_integrated": False,
        },
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "training_ran": False,
        "optimizer_update_ran": False,
        "simulator_contract_probe_ran": True,
        "simulator_training_rollout_ran": False,
        "simulator_contract_probe_world_count": 1,
        "artifact_persisted": False,
        "dataset_created_or_mutated": False,
    }


def _probe_gae_contract(
    *,
    action_names: tuple[str, ...],
    reward_component_names: tuple[str, ...],
    recurrent_rollout_step_type: type,
    recurrent_rollout_buffer_type: type,
    previous_feedback_type: type,
) -> tuple[bool, bool]:
    mask = tuple(index == 0 for index in range(len(action_names)))
    reward_components = {name: 0.0 for name in reward_component_names}
    reward_components["resource_acquisition"] = 1.0
    common = {
        "world_seed": 203,
        "tick": 0,
        "agent_id": 1,
        "decision_index": 0,
        "observation": (0.0,) * 541,
        "previous_feedback": previous_feedback_type.zero(),
        "action_mask": mask,
        "hidden": (0.0,) * 8,
        "action_index": 0,
        "requested_action": action_names[0],
        "logprob": 0.0,
        "entropy": 0.0,
        "value": 0.5,
        "reward": 1.0,
        "reward_components": reward_components,
        "resolved_action_index": 0,
        "resolved_action": action_names[0],
        "resolution_action_mask": mask,
        "action_valid": True,
        "resolution_action_valid": True,
        "moved": False,
        "outcome": {},
    }
    terminated_buffer = recurrent_rollout_buffer_type()
    terminated_buffer.register_world("v203-terminated-probe")
    terminated_buffer.append(
        recurrent_rollout_step_type(
            world_id="v203-terminated-probe",
            terminated=True,
            **common,
        )
    )
    terminated_row = terminated_buffer.compute_gae(
        gamma=0.9,
        gae_lambda=0.8,
    )[0]
    terminated_ok = (
        abs(terminated_row.advantage - 0.5) <= 1.0e-12
        and abs(terminated_row.return_target - 1.0) <= 1.0e-12
    )

    truncated_buffer = recurrent_rollout_buffer_type()
    truncated_buffer.register_world("v203-truncated-probe")
    truncated_buffer.append(
        recurrent_rollout_step_type(
            world_id="v203-truncated-probe",
            truncated=True,
            bootstrap_value=2.0,
            **common,
        )
    )
    truncated_row = truncated_buffer.compute_gae(
        gamma=0.9,
        gae_lambda=0.8,
    )[0]
    truncated_ok = (
        abs(truncated_row.advantage - 2.3) <= 1.0e-12
        and abs(truncated_row.return_target - 2.8) <= 1.0e-12
    )
    return terminated_ok, truncated_ok


def controls_and_future_training_plan() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v203_controls_and_future_training_plan_v1",
        "candidate": {
            "algorithm": "parameter_shared_recurrent_ippo_clipped_ppo",
            "network": "public_604_input_layernorm_encoder_gru_masked_actor_critic",
            "action_authority": "learned_logits_constrained_by_current_public_action_mask",
            "scripted_or_hardcoded_learner_actions": False,
            "heuristic_fallback_in_candidate": False,
            "private_world_or_fixture_identity_inputs": False,
        },
        "required_controls": [
            {
                "name": "masked_random",
                "purpose": "environment and reward-floor control",
                "candidate_or_action_authority": False,
            },
            {
                "name": "current_linear_mind_v3",
                "purpose": "strict per-seed alive and birth baseline",
                "candidate_or_action_authority": False,
            },
            {
                "name": "untrained_masked_recurrent_network",
                "purpose": "initialization and action-distribution control",
                "candidate_or_action_authority": False,
            },
            {
                "name": "feed_forward_history_reset_ppo_ablation",
                "purpose": "measure the causal value of recurrent memory",
                "candidate_or_action_authority": False,
            },
            {
                "name": "public_recurrent_ippo",
                "purpose": "primary learned candidate",
                "candidate_or_action_authority": True,
            },
        ],
        "future_scale_plan": {
            "optimization_worlds": (
                "at least 4096 policy-induced worlds across the fresh curriculum "
                "and train roles before interpreting a scale result"
            ),
            "training_seed_roles": ["curriculum", "train"],
            "configuration_selection_role": "selection",
            "post_selection_role": "validation",
            "promotion_role": "lockbox_once_after_candidate_and_config_freeze",
            "legacy_diagnostic_role": "red_team_only_not_promotion_evidence",
            "learner_replicates": (
                "learner_development then frozen learner_confirmation"
            ),
            "rollout_policy": (
                "freeze behavior parameters per rollout batch, collect exact "
                "policy-induced public transitions, then update outside simulation"
            ),
            "optimization": (
                "ordered recurrent TBPTT PPO with clipped policy/value objectives, "
                "entropy regularization, deterministic minibatches, and gradient clipping"
            ),
        },
        "future_acceptance_matrix": {
            "broad_and_controlled_reported_separately": True,
            "per_seed_alive_birth_no_regression": True,
            "carrion_only_120_terminal_survivor_positive": True,
            "dominant_requested_action_share_cap": 0.5,
            "heuristic_action_source_count": 0,
            "no_aggregate_only_pass": True,
            "deterministic_replay_from_frozen_artifact": True,
        },
        "future_stop_rules": [
            "stop on non-finite optimizer, model, gradient, or rollout statistics",
            "stop on action collapse above the dominant-action cap",
            "stop on strict per-seed alive or birth regression",
            "stop if recurrent candidate does not beat the feed-forward history ablation",
            "do not open lockbox before model, hyperparameters, and artifact are frozen",
            "do not integrate runtime behavior or promote from training success alone",
        ],
        "training_authorized_by_v203": False,
        "future_explicit_route_required": FUTURE_RECURRENT_IPPO_TRAINING_ROUTE,
    }


def route_decision_for_v203(
    *,
    source_validation: Mapping[str, object],
    scaffold_surface_audit: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        route = LINEAGE_REPAIR_ROUTE
    elif scaffold_surface_audit.get("passed") is not True:
        route = SCAFFOLD_REPAIR_ROUTE
    else:
        route = FUTURE_RECURRENT_IPPO_TRAINING_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v203_route_decision_v1",
        "selected_route": route,
        "recommended_next_route": route,
        "exactly_one_next_route_recommended": True,
        "v202_source_pin_valid": source_validation.get("passed") is True,
        "public_recurrent_ippo_scaffold_passed": scaffold_surface_audit.get("passed")
        is True,
        "future_route_requires_separate_explicit_opt_in": True,
        "training_allowed_by_v203": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
        "rationale": _route_rationale(route),
    }


def classification_for_v203(
    *,
    source_validation: Mapping[str, object],
    scaffold_surface_audit: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v203_"
    if source_validation.get("passed") is not True:
        return prefix + "v202_lineage_invalid_routes_to_repair_no_training"
    if scaffold_surface_audit.get("passed") is not True:
        return (
            prefix
            + "public_recurrent_ippo_scaffold_incomplete_routes_to_repair_no_training"
        )
    return (
        prefix
        + "public_masked_model_capacity_harness_scaffold_ready_for_future_explicit_recurrent_ippo_training_no_training"
    )


def lifecycle_flags() -> dict[str, object]:
    return {
        "scaffold_created": True,
        "training_started": False,
        "training_ran": False,
        "optimizer_update_ran": False,
        "training_slice_4_consumed": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_integration_changed": False,
        "runtime_action_selection_changed": False,
        "default_policy_changed": False,
        "support_generated": False,
        "support_expanded": False,
        "dataset_created": False,
        "dataset_mutated": False,
        "gate_relaxed": False,
        "promotion_authorized": False,
        "non_promoted": True,
    }


def historical_evidence_checkpoint() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v203_historical_evidence_checkpoint_v1",
        "v53_v63": "scalar IQL tuning and extraction-only families are closed",
        "v154_v176": "tiny archive and nearest-neighbor scoring are saturated",
        "v180_v186_v195": "three training slices failed and are not promotion evidence",
        "v196_v201": (
            "low-specificity table coverage caused eat collapse and cannot satisfy "
            "high-specific current-valid action demands"
        ),
        "v202": (
            "authorized only a public observation plus current-mask capacity scaffold"
        ),
        "scaffold_hypothesis": (
            "policy-induced recurrent IPPO can learn interaction-level temporal "
            "credit from public finalized transitions without table imputation, "
            "heuristic action authority, seed identity, or private state"
        ),
        "first_falsification": (
            "the recurrent candidate must outperform the same PPO architecture "
            "with hidden state reset each decision before any promotion claim"
        ),
        "anti_loop_stop_rules": [
            "do not rerun scalar IQL or table-bias tuning",
            "do not graft lower-specificity action values",
            "do not hardcode carrion actions or fixture-specific branches",
            "do not use seed, split, fixture, support count, or provenance as model input",
            "do not consume slice 4 from this no-training report",
        ],
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def _v202_lifecycle_closed(report: Mapping[str, object]) -> bool:
    false_keys = (
        "training_started",
        "training_slice_4_consumed",
        "training_artifact_created",
        "runtime_artifact_created",
        "runtime_integration_changed",
        "runtime_action_selection_changed",
        "default_policy_changed",
        "support_generated",
        "support_expanded",
        "dataset_mutated",
        "gate_relaxed",
        "promotion_authorized",
    )
    return all(report.get(key) is False for key in false_keys) and (
        report.get("non_promoted") is True
    )


def _route_rationale(route: str) -> str:
    if route == FUTURE_RECURRENT_IPPO_TRAINING_ROUTE:
        return (
            "The pinned v202 lineage and executable public recurrent-IPPO scaffold "
            "both pass. A future slice-4 training run is worth testing only through "
            "a separate explicit opt-in route; v203 itself trains nothing."
        )
    if route == SCAFFOLD_REPAIR_ROUTE:
        return (
            "At least one public-input, mask, state, rollout, GAE, artifact, seed, "
            "policy, or PPO scaffold invariant failed and must be repaired without training."
        )
    return (
        "The pinned v202 report, digest, route, or closed lifecycle did not validate."
    )


def _load_optional_json_report(
    path: str | Path,
) -> tuple[dict[str, object], str | None]:
    try:
        return load_json_report(path), None
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
