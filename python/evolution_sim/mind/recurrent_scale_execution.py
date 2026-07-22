from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import RecurrentActorCriticConfig
from evolution_sim.mind.recurrent_artifact import (
    FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
    load_frozen_recurrent_policy_artifact,
    load_recurrent_training_crash_checkpoint,
    save_frozen_recurrent_policy_artifact,
    save_recurrent_training_crash_checkpoint,
)
from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED,
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS,
    RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED,
    CounterfactualHorizonScalarization,
    RecurrentCounterfactualAuxiliaryConfig,
    RecurrentCounterfactualAuxiliaryStepConfig,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RecurrentCounterfactualCollectionConfig,
)
from evolution_sim.mind.recurrent_counterfactual_comparison import (
    BASE_ARM,
    EXACT_ARM,
    SHUFFLED_ARM,
)
from evolution_sim.mind.recurrent_evaluation import (
    UNPINNED_NONCANDIDATE_DIGEST_PREFIX,
    evaluate_frozen_recurrent_policy_artifact,
    evaluate_recurrent_model,
    validate_recurrent_evaluation_report,
)
from evolution_sim.mind.recurrent_experiment import (
    RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
    RecurrentCounterfactualExperimentConfig,
    RecurrentExperimentRunner,
    RecurrentTrainingUpdateResult,
    build_recurrent_training_schedule,
    configure_recurrent_training_determinism,
)
from evolution_sim.mind.recurrent_policy import (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
)
from evolution_sim.mind.recurrent_ppo import RecurrentPPOConfig
from evolution_sim.mind.recurrent_runtime_provenance import (
    assert_recurrent_scale_runtime_provenance_match,
    build_recurrent_scale_runtime_provenance,
    validate_recurrent_scale_runtime_provenance,
)
from evolution_sim.mind.recurrent_scale_campaign import (
    RECURRENT_SCALE_ARM_REPORT_SCHEMA_VERSION,
    RECURRENT_SCALE_ARMS,
    RECURRENT_SCALE_CAMPAIGN_ANALYSIS_SCHEMA_VERSION,
    RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT,
    RECURRENT_SCALE_ROLLOUT_TICKS,
    RECURRENT_SCALE_SELECTION_SEED_COUNT,
    RECURRENT_SCALE_TOTAL_TRAINING_WORLDS,
    RECURRENT_SCALE_UPDATE_COUNT,
    RECURRENT_SCALE_WORLDS_PER_UPDATE,
    RecurrentScaleCampaignError,
    load_strict_json,
    recurrent_scale_arm_run_id,
    recurrent_scale_selection_seed_plan,
    source_file_hash_manifest,
    validate_recurrent_scale_campaign_preregistration,
    write_atomic_json,
)
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_SEED_REGISTRY,
)


RECURRENT_SCALE_UPDATE_RECORD_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_scale_update_record_v1"
)
RECURRENT_SCALE_FULL_WORLD_VERIFICATION_RUNNER = (
    "evolution_sim.mind.recurrent_evaluation"
)
_ACTION_SELECTION_MODES = (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_scale_run_components(
    preregistration: Mapping[str, object],
    *,
    learner_seed: int,
    arm: str,
    counterfactual_workers: int,
) -> tuple[
    RecurrentActorCriticConfig,
    RecurrentPPOConfig,
    RecurrentCounterfactualExperimentConfig | None,
    tuple[tuple[object, ...], ...],
]:
    """Resolve executable objects only from the immutable preregistration."""

    validate_recurrent_scale_campaign_preregistration(preregistration)
    recurrent_scale_arm_run_id(learner_seed=learner_seed, arm=arm)
    if (
        isinstance(counterfactual_workers, bool)
        or not isinstance(counterfactual_workers, int)
        or counterfactual_workers <= 0
    ):
        raise RecurrentScaleCampaignError(
            "counterfactual_workers must be a positive integer"
        )
    training = _mapping(preregistration.get("training"), field="training")
    model_payload = _mapping(training.get("model"), field="training.model")
    ppo_payload = _mapping(training.get("ppo"), field="training.ppo")
    model_config = RecurrentActorCriticConfig(
        encoder_size=_positive_int(
            model_payload.get("encoder_size"), field="model.encoder_size"
        ),
        hidden_size=_positive_int(
            model_payload.get("hidden_size"), field="model.hidden_size"
        ),
        recurrent_layers=_positive_int(
            model_payload.get("recurrent_layers"), field="model.recurrent_layers"
        ),
    )
    ppo_config = RecurrentPPOConfig(
        learning_rate=_float(ppo_payload.get("learning_rate"), field="learning_rate"),
        adam_epsilon=_float(ppo_payload.get("adam_epsilon"), field="adam_epsilon"),
        gamma=_float(ppo_payload.get("gamma"), field="gamma"),
        gae_lambda=_float(ppo_payload.get("gae_lambda"), field="gae_lambda"),
        policy_clip_range=_float(
            ppo_payload.get("policy_clip_range"), field="policy_clip_range"
        ),
        value_clip_range=_float(
            ppo_payload.get("value_clip_range"), field="value_clip_range"
        ),
        value_loss_coefficient=_float(
            ppo_payload.get("value_loss_coefficient"),
            field="value_loss_coefficient",
        ),
        entropy_coefficient=_float(
            ppo_payload.get("entropy_coefficient"), field="entropy_coefficient"
        ),
        update_epochs=_positive_int(
            ppo_payload.get("update_epochs"), field="update_epochs"
        ),
        sequence_minibatch_size=_positive_int(
            ppo_payload.get("sequence_minibatch_size"),
            field="sequence_minibatch_size",
        ),
        tbptt_steps=_positive_int(ppo_payload.get("tbptt_steps"), field="tbptt_steps"),
        burn_in_steps=_nonnegative_int(
            ppo_payload.get("burn_in_steps"), field="burn_in_steps"
        ),
        max_gradient_norm=_float(
            ppo_payload.get("max_gradient_norm"), field="max_gradient_norm"
        ),
        normalize_advantages=_exact_bool(
            ppo_payload.get("normalize_advantages"), field="normalize_advantages"
        ),
        advantage_epsilon=_float(
            ppo_payload.get("advantage_epsilon"), field="advantage_epsilon"
        ),
        target_kl=_float(ppo_payload.get("target_kl"), field="target_kl"),
        learner_seed=learner_seed,
        feed_forward_history_ablation=_exact_bool(
            ppo_payload.get("feed_forward_history_ablation"),
            field="feed_forward_history_ablation",
        ),
        world_balanced_loss=_exact_bool(
            ppo_payload.get("world_balanced_loss"), field="world_balanced_loss"
        ),
    )
    counterfactual_config = _counterfactual_config(
        preregistration,
        learner_seed=learner_seed,
        arm=arm,
        workers=counterfactual_workers,
    )
    schedule = build_recurrent_training_schedule(
        update_count=_positive_int(
            training.get("updates_per_arm_learner"),
            field="updates_per_arm_learner",
        ),
        worlds_per_update=_positive_int(
            training.get("worlds_per_update"), field="worlds_per_update"
        ),
        rollout_ticks=_positive_int(
            training.get("rollout_ticks"), field="rollout_ticks"
        ),
        scenarios=_text_sequence(training.get("scenarios"), field="scenarios"),
        seed_registry_contract=RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
        scale_learner_seed=learner_seed,
    )
    return model_config, ppo_config, counterfactual_config, schedule


def _scale_run_contract(
    *,
    preregistration: Mapping[str, object],
    learner_seed: int,
    arm: str,
    model_config: RecurrentActorCriticConfig,
    ppo_config: RecurrentPPOConfig,
    counterfactual_config: RecurrentCounterfactualExperimentConfig | None,
    schedule: Sequence[Sequence[object]],
) -> dict[str, object]:
    contract: dict[str, object] = {
        "preregistration_digest": preregistration["exact_digest"],
        "run_id": recurrent_scale_arm_run_id(
            learner_seed=learner_seed,
            arm=arm,
        ),
        "learner_seed": learner_seed,
        "arm": arm,
        "model": asdict(model_config),
        "ppo": asdict(ppo_config),
        "counterfactual": (
            None
            if counterfactual_config is None
            else {
                "collection": asdict(counterfactual_config.collection),
                "auxiliary": counterfactual_config.auxiliary.as_contract(),
                "step": counterfactual_config.step.as_contract(),
                "bundles_per_update": counterfactual_config.bundles_per_update,
                "branch_tick_candidates": list(
                    counterfactual_config.branch_tick_candidates
                ),
            }
        ),
        "schedule_sha256": stable_payload_digest(
            [[asdict(task) for task in tasks] for tasks in schedule]
        ),
    }
    normalized = json.loads(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    if not isinstance(normalized, dict):
        raise RecurrentScaleCampaignError(
            "scale run contract must normalize to a JSON object"
        )
    return normalized


def run_recurrent_scale_arm(
    preregistration: Mapping[str, object],
    *,
    expected_preregistration_digest: str,
    learner_seed: int,
    arm: str,
    output_root: str | Path,
    device: torch.device | str,
    rollout_workers: int,
    counterfactual_workers: int,
    evaluation_workers: int,
    runtime_provenance_path: str | Path,
    resume: bool,
) -> dict[str, object]:
    """Run one learner/arm cell with atomic updates and exact CPU replay."""

    validate_recurrent_scale_campaign_preregistration(preregistration)
    preregistration_digest = _sha256(
        preregistration.get("exact_digest"), field="preregistration.exact_digest"
    )
    if preregistration_digest != _sha256(
        expected_preregistration_digest,
        field="expected_preregistration_digest",
    ):
        raise RecurrentScaleCampaignError("preregistration digest pin mismatched")
    run_id = recurrent_scale_arm_run_id(learner_seed=learner_seed, arm=arm)
    source = _mapping(preregistration.get("source"), field="source")
    source_commit = _text(source.get("commit"), field="source.commit")
    source_manifest_sha256 = _sha256(
        source.get("manifest_sha256"), field="source.manifest_sha256"
    )
    _require_exact_clean_source(
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
    )
    for field, value in (
        ("rollout_workers", rollout_workers),
        ("counterfactual_workers", counterfactual_workers),
        ("evaluation_workers", evaluation_workers),
    ):
        _positive_int(value, field=field)
    if type(resume) is not bool:
        raise RecurrentScaleCampaignError("resume must be an exact boolean")
    runtime_path = Path(runtime_provenance_path)
    if not runtime_path.is_file():
        raise RecurrentScaleCampaignError("runtime provenance file is missing")
    runtime_provenance = load_strict_json(runtime_path)
    validate_recurrent_scale_runtime_provenance(runtime_provenance)
    configure_recurrent_training_determinism(
        learner_seed=learner_seed,
        device=device,
    )
    observed_runtime_provenance = build_recurrent_scale_runtime_provenance(
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
        preregistration_digest=preregistration_digest,
        repository_clean=True,
        device=device,
        rollout_workers=rollout_workers,
        counterfactual_workers=counterfactual_workers,
        evaluation_workers=evaluation_workers,
    )
    required_device_type = _text(
        _mapping(
            preregistration.get("execution"),
            field="preregistration.execution",
        ).get("training_device_type"),
        field="preregistration.execution.training_device_type",
    )
    if (
        _mapping(
            observed_runtime_provenance.get("device"),
            field="observed_runtime_provenance.device",
        ).get("type")
        != required_device_type
    ):
        raise RecurrentScaleCampaignError(
            "scale campaign runtime device differs from preregistration"
        )
    assert_recurrent_scale_runtime_provenance_match(
        runtime_provenance,
        observed_runtime_provenance,
    )
    runtime_provenance_digest = _sha256(
        runtime_provenance.get("exact_digest"),
        field="runtime_provenance.exact_digest",
    )
    runtime_provenance_reference = _file_reference(runtime_path)
    run_directory = Path(output_root) / run_id
    update_directory = run_directory / "updates"
    evidence_directory = run_directory / "evidence"
    evaluation_directory = run_directory / "evaluations"
    artifact_path = run_directory / "frozen-policy.json"
    checkpoint_path = run_directory / "training-checkpoint.json"
    report_path = run_directory / "report.json"
    run_directory.mkdir(parents=True, exist_ok=True)
    if report_path.is_file():
        if not resume:
            raise RecurrentScaleCampaignError(
                "completed report exists but resume was not explicitly enabled"
            )
        existing = load_strict_json(report_path)
        validate_recurrent_scale_arm_report(existing, preregistration=preregistration)
        if existing.get("runtime_provenance") != {
            "payload": runtime_provenance,
            "file": runtime_provenance_reference,
        }:
            raise RecurrentScaleCampaignError(
                "completed report runtime provenance drifted"
            )
        _verify_completed_report_evidence(
            existing,
            run_directory=run_directory,
            runtime_provenance_path=runtime_path,
            preregistration=preregistration,
        )
        return existing

    model_config, ppo_config, counterfactual_config, schedule = (
        build_scale_run_components(
            preregistration,
            learner_seed=learner_seed,
            arm=arm,
            counterfactual_workers=counterfactual_workers,
        )
    )
    run_contract = _scale_run_contract(
        preregistration=preregistration,
        learner_seed=learner_seed,
        arm=arm,
        model_config=model_config,
        ppo_config=ppo_config,
        counterfactual_config=counterfactual_config,
        schedule=schedule,
    )
    runner = RecurrentExperimentRunner(
        learner_seed=learner_seed,
        device=device,
        model_config=model_config,
        ppo_config=ppo_config,
        rollout_workers=rollout_workers,
        counterfactual_config=counterfactual_config,
    )
    completed_updates = 0
    elapsed_seconds_before_resume = 0.0
    if checkpoint_path.is_file():
        if not resume:
            raise RecurrentScaleCampaignError(
                "checkpoint exists but resume was not explicitly enabled"
            )
        loaded = load_recurrent_training_crash_checkpoint(checkpoint_path)
        checkpoint = loaded.checkpoint
        checkpoint_source = _mapping(
            checkpoint.get("source"), field="checkpoint.source"
        )
        checkpoint_config = _mapping(
            checkpoint.get("configuration"), field="checkpoint.configuration"
        )
        progress = _mapping(checkpoint.get("progress"), field="checkpoint.progress")
        if (
            checkpoint_source.get("source_commit") != source_commit
            or checkpoint_source.get("source_manifest_sha256") != source_manifest_sha256
            or checkpoint_source.get("seed_registry_digest")
            != SCALE_DEVELOPMENT_CANONICAL_SHA256
            or checkpoint_config.get("training_config") != run_contract
            or progress.get("run_id") != run_id
            or progress.get("learner_seed") != learner_seed
        ):
            raise RecurrentScaleCampaignError(
                "crash checkpoint does not match the exact scale run contract"
            )
        completed_updates = _nonnegative_int(
            progress.get("completed_updates"), field="completed_updates"
        )
        if completed_updates > len(schedule):
            raise RecurrentScaleCampaignError(
                "checkpoint progress exceeds the preregistered schedule"
            )
        loaded_rng_state = _mapping(
            loaded.rng_state,
            field="checkpoint.rng_state",
        )
        if (
            loaded_rng_state.get("scale_runtime_provenance_digest")
            != runtime_provenance_digest
        ):
            raise RecurrentScaleCampaignError(
                "checkpoint runtime provenance digest drifted"
            )
        runner.restore_training_checkpoint_state(
            model_state=loaded.model.state_dict(),
            optimizer_state=loaded.optimizer_state,
            rng_state=loaded.rng_state,
            completed_updates=completed_updates,
        )
        elapsed_seconds_before_resume = _float(
            loaded_rng_state.get("scale_campaign_elapsed_seconds"),
            field="checkpoint.scale_campaign_elapsed_seconds",
        )
        if elapsed_seconds_before_resume < 0.0:
            raise RecurrentScaleCampaignError(
                "checkpoint scale campaign elapsed time cannot be negative"
            )
    started = time.perf_counter()
    for update_index in range(completed_updates, len(schedule)):
        update = runner.train_update(schedule[update_index])
        raw_collection_evidence = _write_collection_evidence(
            update,
            evidence_directory=evidence_directory,
        )
        update_payload = _scale_update_payload(
            update,
            run_id=run_id,
            raw_collection_evidence=raw_collection_evidence,
        )
        update_payload["exact_digest"] = stable_payload_digest(update_payload)
        write_atomic_json(
            update_directory / f"update-{update_index:04d}.json",
            update_payload,
        )
        checkpoint_state = runner.export_training_checkpoint_state()
        checkpoint_rng_state = dict(
            _mapping(
                checkpoint_state["rng_state"],
                field="runner checkpoint rng_state",
            )
        )
        checkpoint_rng_state["scale_campaign_elapsed_seconds"] = round(
            elapsed_seconds_before_resume + time.perf_counter() - started,
            6,
        )
        checkpoint_rng_state["scale_runtime_provenance_digest"] = (
            runtime_provenance_digest
        )
        save_recurrent_training_crash_checkpoint(
            checkpoint_path,
            runner.model,
            optimizer_state=checkpoint_state["optimizer_state"],
            rng_state=checkpoint_rng_state,
            optimizer_type="torch.optim.Adam",
            training_config=run_contract,
            seed_registry_digest=SCALE_DEVELOPMENT_CANONICAL_SHA256,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
            learner_seed=learner_seed,
            completed_updates=runner.completed_update_count,
            run_id=run_id,
        )

    update_records = tuple(
        _load_update_record(update_directory / f"update-{index:04d}.json", index=index)
        for index in range(len(schedule))
    )
    total_worlds = sum(
        _positive_int(
            _mapping(record.get("rollout"), field="update.rollout").get("world_count"),
            field="update.rollout.world_count",
        )
        for record in update_records
    )
    expected_worlds = RECURRENT_SCALE_UPDATE_COUNT * RECURRENT_SCALE_WORLDS_PER_UPDATE
    if total_worlds != expected_worlds:
        raise RecurrentScaleCampaignError(
            "arm/learner training-world count drifted from preregistration"
        )
    total_transitions = sum(
        _positive_int(
            _mapping(record.get("rollout"), field="update.rollout").get(
                "transition_count"
            ),
            field="update.rollout.transition_count",
        )
        for record in update_records
    )
    seed_provenance = _scale_training_seed_provenance(schedule)
    treatment_delivery = _treatment_delivery_summary(update_records, arm=arm)
    sampling_stream_id = _scale_sampling_stream_id(
        preregistration_digest,
        learner_seed=learner_seed,
    )
    seed_plan = recurrent_scale_selection_seed_plan()
    in_memory_evaluations: dict[str, dict[str, object]] = {}
    for mode in _ACTION_SELECTION_MODES:
        report = evaluate_recurrent_model(
            runner.model,
            synthetic_noncandidate_digest_label=(
                UNPINNED_NONCANDIDATE_DIGEST_PREFIX
                + stable_payload_digest(
                    {
                        "policy": "scale_full_world_manifest_source_v1",
                        "preregistration_digest": preregistration_digest,
                        "run_id": run_id,
                        "selection_mode": mode,
                    }
                )
            ),
            seed_plan=seed_plan,
            fixture_names=("carrion_only",),
            candidate_action_selection=mode,
            candidate_sampling_seed_count=(
                1
                if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
                else RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
            ),
            candidate_sampling_stream_id=sampling_stream_id,
            evaluation_workers=evaluation_workers,
        )
        in_memory_evaluations[mode] = report
        write_atomic_json(evaluation_directory / f"in-memory-{mode}.json", report)

    full_world_manifest = _full_world_replay_manifest(in_memory_evaluations)
    artifact = save_frozen_recurrent_policy_artifact(
        artifact_path,
        runner.model,
        training_config=asdict(ppo_config),
        experiment_config=run_contract,
        seed_registry_digest=SCALE_DEVELOPMENT_CANONICAL_SHA256,
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
        data_metadata={
            **seed_provenance,
            "policy_induced": True,
            "worlds": total_worlds,
            "agent_transitions": total_transitions,
            "runtime_provenance_digest": runtime_provenance_digest,
            "dependency_freeze_sha256": _sha256(
                _mapping(
                    runtime_provenance.get("dependency_freeze"),
                    field="runtime_provenance.dependency_freeze",
                ).get("packages_sha256"),
                field="runtime_provenance.dependency_freeze.packages_sha256",
            ),
            "scenarios": list(
                _mapping(preregistration.get("training"), field="training").get(
                    "scenarios", []
                )
            ),
        },
        run_metadata={
            "purpose": "preregistered_scale_development_arm",
            "preregistration_digest": preregistration_digest,
            "run_id": run_id,
            "arm": arm,
            "runtime_provenance_digest": runtime_provenance_digest,
            "campaign_slice_consumed": True,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
        },
        full_world_replay_manifest=full_world_manifest,
        learner_seed=learner_seed,
        learner_device=str(torch.device(device)),
    )
    artifact_evaluations: dict[str, dict[str, object]] = {}
    evaluation_evidence: dict[str, object] = {}
    for mode in _ACTION_SELECTION_MODES:
        report = evaluate_frozen_recurrent_policy_artifact(
            artifact_path,
            seed_plan=seed_plan,
            expected_source_commit=source_commit,
            expected_source_manifest_sha256=source_manifest_sha256,
            expected_seed_registry_digest=SCALE_DEVELOPMENT_CANONICAL_SHA256,
            fixture_names=("carrion_only",),
            candidate_action_selection=mode,
            candidate_sampling_seed_count=(
                1
                if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
                else RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
            ),
            candidate_sampling_stream_id=sampling_stream_id,
            evaluation_workers=evaluation_workers,
        )
        artifact_evaluations[mode] = report
        artifact_evaluation_path = evaluation_directory / f"artifact-cpu-{mode}.json"
        write_atomic_json(artifact_evaluation_path, report)
        memory_summary = recurrent_candidate_outcome_summary(
            in_memory_evaluations[mode]
        )
        artifact_summary = recurrent_candidate_outcome_summary(report)
        if memory_summary != artifact_summary:
            raise RecurrentScaleCampaignError(
                "frozen CPU artifact outcomes differ from the in-memory model"
            )
        evaluation_evidence[mode] = {
            "in_memory_report": _file_reference(
                evaluation_directory / f"in-memory-{mode}.json"
            ),
            "artifact_cpu_report": _file_reference(artifact_evaluation_path),
            "exact_cpu_outcome_replay_match": True,
            "candidate_outcome_summary": artifact_summary,
        }

    report: dict[str, object] = {
        "schema_version": RECURRENT_SCALE_ARM_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "preregistration_digest": preregistration_digest,
        "source": {
            "commit": source_commit,
            "manifest_sha256": source_manifest_sha256,
            "clean_tree_verified_before_training": True,
            "source_stable_through_completion": (
                _source_still_exact(source_commit, source_manifest_sha256)
            ),
        },
        "runtime_provenance": {
            "payload": runtime_provenance,
            "file": runtime_provenance_reference,
        },
        "learner_seed": learner_seed,
        "learner_seed_role": "scale_learner",
        "arm": arm,
        "configuration": run_contract,
        "training": {
            "completed_updates": len(update_records),
            "worlds": total_worlds,
            "agent_transitions": total_transitions,
            "update_record_digests": [
                record["exact_digest"] for record in update_records
            ],
            "update_journals": [
                _file_reference(update_directory / f"update-{index:04d}.json")
                for index in range(len(update_records))
            ],
            "environment_seed_provenance": seed_provenance,
            "treatment_delivery": treatment_delivery,
            "checkpoint": _file_reference(checkpoint_path),
        },
        "artifact": {
            "path": str(artifact_path),
            "artifact_sha256": artifact["artifact_sha256"],
            "file": _file_reference(artifact_path),
            "runtime_provenance_digest": runtime_provenance_digest,
            "runtime_policy_eligible": False,
            "full_world_replay_manifest": full_world_manifest,
        },
        "evaluations": {
            "sampling_stream_id": sampling_stream_id,
            "selection_seed_plan_sha256": seed_plan.digest,
            "modes": evaluation_evidence,
        },
        "elapsed_seconds": round(
            elapsed_seconds_before_resume + time.perf_counter() - started,
            6,
        ),
        "lifecycle": {
            "development_only": True,
            "campaign_slice_consumed": True,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
            "validation_seeds_accessed": False,
            "lockbox_seeds_accessed": False,
            "gate_relaxation_authorized": False,
        },
    }
    if report["source"]["source_stable_through_completion"] is not True:  # type: ignore[index]
        raise RecurrentScaleCampaignError(
            "source tree changed during the scale run; report withheld"
        )
    report["exact_digest"] = stable_payload_digest(report)
    validate_recurrent_scale_arm_report(report, preregistration=preregistration)
    write_atomic_json(report_path, report)
    return report


def validate_recurrent_scale_arm_report(
    report: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
) -> None:
    validate_recurrent_scale_campaign_preregistration(preregistration)
    if report.get("schema_version") != RECURRENT_SCALE_ARM_REPORT_SCHEMA_VERSION:
        raise RecurrentScaleCampaignError("scale arm report schema drifted")
    exact_digest = _sha256(report.get("exact_digest"), field="report.exact_digest")
    unsigned = dict(report)
    unsigned.pop("exact_digest", None)
    if stable_payload_digest(unsigned) != exact_digest:
        raise RecurrentScaleCampaignError("scale arm report digest mismatched")
    if report.get("preregistration_digest") != preregistration.get("exact_digest"):
        raise RecurrentScaleCampaignError("scale arm report preregistration drifted")
    learner_seed = _positive_int(report.get("learner_seed"), field="learner_seed")
    arm = _text(report.get("arm"), field="arm")
    expected_run_id = recurrent_scale_arm_run_id(
        learner_seed=learner_seed,
        arm=arm,
    )
    if report.get("run_id") != expected_run_id:
        raise RecurrentScaleCampaignError("scale arm run identity drifted")
    if report.get("learner_seed_role") != "scale_learner":
        raise RecurrentScaleCampaignError("scale learner seed role drifted")
    source = _mapping(report.get("source"), field="source")
    preregistered_source = _mapping(
        preregistration.get("source"), field="preregistration.source"
    )
    if (
        source.get("commit") != preregistered_source.get("commit")
        or source.get("manifest_sha256") != preregistered_source.get("manifest_sha256")
        or source.get("clean_tree_verified_before_training") is not True
        or source.get("source_stable_through_completion") is not True
    ):
        raise RecurrentScaleCampaignError("scale arm source evidence drifted")
    runtime_evidence = _mapping(
        report.get("runtime_provenance"),
        field="runtime_provenance",
    )
    runtime_payload = _mapping(
        runtime_evidence.get("payload"),
        field="runtime_provenance.payload",
    )
    validate_recurrent_scale_runtime_provenance(runtime_payload)
    _validate_file_reference_contract(
        _mapping(
            runtime_evidence.get("file"),
            field="runtime_provenance.file",
        ),
        field="runtime_provenance.file",
    )
    runtime_binding = _mapping(
        runtime_payload.get("contract_binding"),
        field="runtime_provenance.contract_binding",
    )
    runtime_device = _mapping(
        runtime_payload.get("device"),
        field="runtime_provenance.device",
    )
    execution_contract = _mapping(
        preregistration.get("execution"),
        field="preregistration.execution",
    )
    if (
        runtime_binding.get("preregistration_digest")
        != preregistration.get("exact_digest")
        or runtime_binding.get("source_commit") != preregistered_source.get("commit")
        or runtime_binding.get("source_manifest_sha256")
        != preregistered_source.get("manifest_sha256")
        or runtime_device.get("type") != execution_contract.get("training_device_type")
    ):
        raise RecurrentScaleCampaignError(
            "scale arm runtime provenance binding drifted"
        )
    runtime_provenance_digest = _sha256(
        runtime_payload.get("exact_digest"),
        field="runtime_provenance.exact_digest",
    )
    model, ppo, counterfactual, schedule = build_scale_run_components(
        preregistration,
        learner_seed=learner_seed,
        arm=arm,
        counterfactual_workers=1,
    )
    expected_configuration = _scale_run_contract(
        preregistration=preregistration,
        learner_seed=learner_seed,
        arm=arm,
        model_config=model,
        ppo_config=ppo,
        counterfactual_config=counterfactual,
        schedule=schedule,
    )
    configuration = _mapping(report.get("configuration"), field="configuration")
    if dict(configuration) != expected_configuration:
        raise RecurrentScaleCampaignError(
            "scale arm executable configuration or schedule drifted"
        )
    training = _mapping(report.get("training"), field="training")
    if (
        training.get("completed_updates") != RECURRENT_SCALE_UPDATE_COUNT
        or training.get("worlds")
        != RECURRENT_SCALE_UPDATE_COUNT * RECURRENT_SCALE_WORLDS_PER_UPDATE
    ):
        raise RecurrentScaleCampaignError("scale arm training budget is incomplete")
    _positive_int(training.get("agent_transitions"), field="training.agent_transitions")
    update_record_digests = training.get("update_record_digests")
    if (
        not isinstance(update_record_digests, list)
        or len(update_record_digests) != RECURRENT_SCALE_UPDATE_COUNT
    ):
        raise RecurrentScaleCampaignError(
            "scale arm update digest evidence is incomplete"
        )
    parsed_update_digests = tuple(
        _sha256(value, field="training.update_record_digests[]")
        for value in update_record_digests
    )
    if len(set(parsed_update_digests)) != len(parsed_update_digests):
        raise RecurrentScaleCampaignError(
            "scale arm update digest evidence contains duplicates"
        )
    update_journals = training.get("update_journals")
    if (
        not isinstance(update_journals, list)
        or len(update_journals) != RECURRENT_SCALE_UPDATE_COUNT
    ):
        raise RecurrentScaleCampaignError(
            "scale arm update journal file evidence is incomplete"
        )
    for index, reference in enumerate(update_journals):
        _validate_file_reference_contract(
            _mapping(reference, field=f"training.update_journals[{index}]"),
            field=f"training.update_journals[{index}]",
        )
    expected_seed_provenance = _scale_training_seed_provenance(schedule)
    if training.get("environment_seed_provenance") != expected_seed_provenance:
        raise RecurrentScaleCampaignError("scale arm training seed provenance drifted")
    _validate_file_reference_contract(
        _mapping(training.get("checkpoint"), field="training.checkpoint"),
        field="training.checkpoint",
    )
    delivery = _mapping(
        training.get("treatment_delivery"), field="training.treatment_delivery"
    )
    if arm == BASE_ARM:
        if (
            delivery.get("treatment_expected") is not False
            or delivery.get("attempted_update_count") != 0
            or delivery.get("accepted_update_count") != 0
            or delivery.get("meets_preregistered_delivery_floor") is not True
        ):
            raise RecurrentScaleCampaignError("base treatment-delivery proof drifted")
    else:
        accepted_update_count = _nonnegative_int(
            delivery.get("accepted_update_count"), field="accepted_update_count"
        )
        parameter_delta_l2_sum = _float(
            delivery.get("parameter_delta_l2_sum"), field="parameter_delta_l2_sum"
        )
        within_bounds = delivery.get("all_transactions_within_kl_bounds") is True
        expected_delivery = (
            accepted_update_count >= 12
            and parameter_delta_l2_sum > 0.0
            and within_bounds
        )
        if (
            delivery.get("treatment_expected") is not True
            or delivery.get("attempted_update_count") != RECURRENT_SCALE_UPDATE_COUNT
            or accepted_update_count > RECURRENT_SCALE_UPDATE_COUNT
            or delivery.get("meets_preregistered_delivery_floor")
            is not expected_delivery
        ):
            raise RecurrentScaleCampaignError(
                "treatment-delivery proof is internally inconsistent"
            )
    artifact = _mapping(report.get("artifact"), field="artifact")
    _text(artifact.get("path"), field="artifact.path")
    _sha256(artifact.get("artifact_sha256"), field="artifact.artifact_sha256")
    _validate_file_reference_contract(
        _mapping(artifact.get("file"), field="artifact.file"),
        field="artifact.file",
    )
    if artifact.get("runtime_policy_eligible") is not False:
        raise RecurrentScaleCampaignError("scale artifact eligibility drifted")
    if artifact.get("runtime_provenance_digest") != runtime_provenance_digest:
        raise RecurrentScaleCampaignError(
            "scale artifact runtime provenance digest drifted"
        )
    _validate_full_world_replay_manifest(
        _mapping(
            artifact.get("full_world_replay_manifest"),
            field="artifact.full_world_replay_manifest",
        )
    )
    evaluations = _mapping(report.get("evaluations"), field="evaluations")
    expected_sampling_stream_id = _scale_sampling_stream_id(
        _sha256(
            preregistration.get("exact_digest"),
            field="preregistration.exact_digest",
        ),
        learner_seed=learner_seed,
    )
    if evaluations.get("sampling_stream_id") != expected_sampling_stream_id:
        raise RecurrentScaleCampaignError("evaluation sampling stream drifted")
    expected_seed_plan = recurrent_scale_selection_seed_plan()
    if evaluations.get("selection_seed_plan_sha256") != expected_seed_plan.digest:
        raise RecurrentScaleCampaignError("evaluation seed-plan pin drifted")
    modes = _mapping(
        evaluations.get("modes"),
        field="evaluations.modes",
    )
    if set(modes) != set(_ACTION_SELECTION_MODES):
        raise RecurrentScaleCampaignError("scale arm evaluation modes drifted")
    for mode in _ACTION_SELECTION_MODES:
        evidence = _mapping(modes.get(mode), field=f"evaluations.modes.{mode}")
        if evidence.get("exact_cpu_outcome_replay_match") is not True:
            raise RecurrentScaleCampaignError("CPU artifact replay was not exact")
        _validate_file_reference_contract(
            _mapping(
                evidence.get("in_memory_report"),
                field=f"evaluations.modes.{mode}.in_memory_report",
            ),
            field=f"evaluations.modes.{mode}.in_memory_report",
        )
        _validate_file_reference_contract(
            _mapping(
                evidence.get("artifact_cpu_report"),
                field=f"evaluations.modes.{mode}.artifact_cpu_report",
            ),
            field=f"evaluations.modes.{mode}.artifact_cpu_report",
        )
        _validate_candidate_outcome_summary_contract(
            _mapping(
                evidence.get("candidate_outcome_summary"),
                field=f"evaluations.modes.{mode}.candidate_outcome_summary",
            ),
            mode=mode,
            expected_sampling_stream_id=expected_sampling_stream_id,
            expected_seed_plan_digest=expected_seed_plan.digest,
            expected_environment_seeds=tuple(expected_seed_plan.broad_seeds),
        )
    lifecycle = _mapping(report.get("lifecycle"), field="lifecycle")
    if (
        lifecycle.get("development_only") is not True
        or lifecycle.get("campaign_slice_consumed") is not True
        or lifecycle.get("runtime_artifact_created") is not False
        or lifecycle.get("runtime_action_selection_changed") is not False
        or lifecycle.get("validation_seeds_accessed") is not False
        or lifecycle.get("lockbox_seeds_accessed") is not False
        or lifecycle.get("runtime_integration_authorized") is not False
        or lifecycle.get("promotion_authorized") is not False
        or lifecycle.get("gate_relaxation_authorized") is not False
    ):
        raise RecurrentScaleCampaignError("scale arm lifecycle flags drifted")
    elapsed_seconds = _float(report.get("elapsed_seconds"), field="elapsed_seconds")
    if elapsed_seconds < 0.0:
        raise RecurrentScaleCampaignError("scale arm elapsed time cannot be negative")


def _validate_file_reference_contract(
    reference: Mapping[str, object],
    *,
    field: str,
) -> None:
    _text(reference.get("path"), field=f"{field}.path")
    _sha256(reference.get("sha256"), field=f"{field}.sha256")
    _positive_int(reference.get("byte_length"), field=f"{field}.byte_length")


def _validate_full_world_replay_manifest(
    manifest: Mapping[str, object],
) -> None:
    expected_world_count = (
        RECURRENT_SCALE_SELECTION_SEED_COUNT
        * (1 + RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT)
        * 2
    )
    if (
        manifest.get("schema_version") != FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION
        or manifest.get("environment_seed_registry_sha256")
        != SCALE_DEVELOPMENT_CANONICAL_SHA256
        or manifest.get("environment_seed_roles") != ["scale_selection"]
        or manifest.get("scenario_names") != ["broad", "carrion_only"]
        or manifest.get("tick_horizons") != [RECURRENT_SCALE_ROLLOUT_TICKS]
        or manifest.get("world_count") != expected_world_count
        or manifest.get("replay_verified_world_count") != expected_world_count
        or manifest.get("policy_sampling_stream_count")
        != 1 + RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
        or manifest.get("all_replays_exact") is not True
        or manifest.get("verification_runner")
        != RECURRENT_SCALE_FULL_WORLD_VERIFICATION_RUNNER
    ):
        raise RecurrentScaleCampaignError(
            "scale artifact full-world replay manifest drifted"
        )
    _sha256(
        manifest.get("manifest_sha256"),
        field="full_world_replay_manifest.manifest_sha256",
    )
    _sha256(
        manifest.get("replay_engine_contract_sha256"),
        field="full_world_replay_manifest.replay_engine_contract_sha256",
    )
    _sha256(
        manifest.get("verification_runner_sha256"),
        field="full_world_replay_manifest.verification_runner_sha256",
    )


def _validate_candidate_outcome_summary_contract(
    summary: Mapping[str, object],
    *,
    mode: str,
    expected_sampling_stream_id: str,
    expected_seed_plan_digest: str,
    expected_environment_seeds: tuple[int, ...],
) -> None:
    contract = _mapping(
        summary.get("evaluation_contract"), field="summary.evaluation_contract"
    )
    expected_stream_count = (
        1
        if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
        else RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
    )
    expected_sampling_contract = (
        "none_argmax"
        if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
        else "sha256_namespace_sampling_stream_id_and_replicate_index_first_63_bits"
    )
    raw_sampling_seeds = contract.get("candidate_sampling_seeds")
    if not isinstance(raw_sampling_seeds, list):
        raise RecurrentScaleCampaignError(
            "candidate evaluation sampling seeds must be a list"
        )
    sampling_seeds = tuple(
        _nonnegative_int(value, field="candidate_sampling_seeds[]")
        for value in raw_sampling_seeds
    )
    expected_seed_count_without_argmax_sentinel = (
        0
        if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
        else RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
    )
    exact_expected_sampling_seeds = (
        ()
        if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION
        else _scale_policy_sampling_seeds(expected_sampling_stream_id)
    )
    expected_contract = {
        "ticks": RECURRENT_SCALE_ROLLOUT_TICKS,
        "world_horizon_policy": "exact_configured_120_tick_horizon",
        "candidate_action_selection": mode,
        "candidate_sampling_seeds": list(exact_expected_sampling_seeds),
        "candidate_sampling_seed_count": expected_stream_count,
        "candidate_sampling_stream_id": expected_sampling_stream_id,
        "candidate_sampling_stream_id_source": (
            "caller_supplied_arm_independent_identifier"
        ),
        "candidate_sampling_seed_contract": expected_sampling_contract,
        "candidate_recurrent_state": "one_hidden_state_per_agent",
        "candidate_policy_inputs": [
            "current_public_ecological_observation",
            "current_public_action_mask",
            "same_agent_previous_public_outcome",
            "same_agent_recurrent_state",
        ],
        "candidate_forbidden_inputs": [
            "world_seed",
            "fixture_name",
            "private_world_state",
            "evaluation_split_role",
        ],
        "candidate_factory_receives_seed_or_fixture": False,
        "candidate_replay_verification": "every_run_exact_digest_repeat",
        "strict_zero_unsupported_requested_actions": True,
        "strict_zero_heuristic_candidate_actions": True,
    }
    if (
        len(sampling_seeds) != expected_seed_count_without_argmax_sentinel
        or len(set(sampling_seeds)) != len(sampling_seeds)
        or sampling_seeds != exact_expected_sampling_seeds
        or dict(contract) != expected_contract
    ):
        raise RecurrentScaleCampaignError("candidate evaluation contract drifted")
    if summary.get("seed_plan_digest") != expected_seed_plan_digest:
        raise RecurrentScaleCampaignError("candidate summary seed-plan drifted")
    contexts = summary.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 2:
        raise RecurrentScaleCampaignError(
            "candidate summary must contain broad and carrion contexts"
        )
    parsed_contexts = {
        _text(
            _mapping(value, field="candidate context").get("context"),
            field="candidate context name",
        ): _mapping(value, field="candidate context")
        for value in contexts
    }
    if set(parsed_contexts) != {"broad_default", "fixture:carrion_only"}:
        raise RecurrentScaleCampaignError("candidate summary contexts drifted")
    if len(parsed_contexts) != len(contexts):
        raise RecurrentScaleCampaignError("candidate summary contexts are duplicated")
    for context_name in ("broad_default", "fixture:carrion_only"):
        context = parsed_contexts[context_name]
        candidate_runs = _validated_context_runs_by_identity(
            context,
            run_field="runs",
            expected_context=context_name,
        )
        _validate_expected_run_identities(
            set(candidate_runs),
            mode=mode,
            expected_environment_seeds=expected_environment_seeds,
            expected_sampling_seeds=sampling_seeds,
        )
        linear_runs = _validated_context_runs_by_identity(
            context,
            run_field="linear_runs",
            expected_context=context_name,
        )
        expected_linear_identities = {
            (seed, None) for seed in expected_environment_seeds
        }
        if set(linear_runs) != expected_linear_identities:
            raise RecurrentScaleCampaignError(
                "fixed linear evaluation identities drifted"
            )


def _validated_context_runs_by_identity(
    context: Mapping[str, object],
    *,
    run_field: str,
    expected_context: str,
) -> dict[tuple[int, int | None], Mapping[str, object]]:
    raw_runs = context.get(run_field)
    if not isinstance(raw_runs, list) or not raw_runs:
        raise RecurrentScaleCampaignError(f"candidate summary {run_field} are missing")
    result: dict[tuple[int, int | None], Mapping[str, object]] = {}
    for raw_run in raw_runs:
        run = _mapping(raw_run, field=f"candidate summary {run_field}[]")
        if run.get("context") != expected_context:
            raise RecurrentScaleCampaignError(
                f"candidate summary {run_field} context drifted"
            )
        seed = _positive_int(run.get("seed"), field=f"{run_field}[].seed")
        sampling_seed = run.get("policy_sampling_seed")
        if sampling_seed is not None:
            sampling_seed = _nonnegative_int(
                sampling_seed,
                field=f"{run_field}[].policy_sampling_seed",
            )
        identity = (seed, sampling_seed)
        if identity in result:
            raise RecurrentScaleCampaignError(
                f"duplicate {run_field} evaluation identity"
            )
        _sha256(run.get("behavior_digest"), field=f"{run_field}[].behavior_digest")
        result[identity] = run
    return result


def _validate_expected_run_identities(
    identities: set[tuple[int, int | None]],
    *,
    mode: str,
    expected_environment_seeds: tuple[int, ...],
    expected_sampling_seeds: tuple[int, ...],
) -> None:
    if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION:
        expected = {(seed, None) for seed in expected_environment_seeds}
    else:
        expected = {
            (seed, sampling_seed)
            for seed in expected_environment_seeds
            for sampling_seed in expected_sampling_seeds
        }
    if identities != expected:
        raise RecurrentScaleCampaignError(
            "candidate evaluation run identities drifted from the seed plan"
        )


def _scale_policy_sampling_seeds(sampling_stream_id: str) -> tuple[int, ...]:
    namespace = "evolution-sim|mind-v3-public-recurrent-evaluation|artifact-sampling-v1"
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                (f"{namespace}|{sampling_stream_id}|replicate-{index:04d}").encode(
                    "ascii"
                )
            ).digest()[:8],
            byteorder="big",
        )
        & (2**63 - 1)
        for index in range(RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT)
    )


def analyze_recurrent_scale_campaign(
    preregistration: Mapping[str, object],
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reconcile all 24 cells and apply only the preregistered development gates."""

    validate_recurrent_scale_campaign_preregistration(preregistration)
    expected_seeds = tuple(SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"])
    expected_cells = {
        (seed, arm) for seed in expected_seeds for arm in RECURRENT_SCALE_ARMS
    }
    cells: dict[tuple[int, str], Mapping[str, object]] = {}
    for report in reports:
        validate_recurrent_scale_arm_report(report, preregistration=preregistration)
        key = (
            _positive_int(report.get("learner_seed"), field="learner_seed"),
            _text(report.get("arm"), field="arm"),
        )
        if key in cells:
            raise RecurrentScaleCampaignError("duplicate scale arm/learner report")
        cells[key] = report
    if set(cells) != expected_cells:
        missing = sorted(expected_cells - set(cells))
        extra = sorted(set(cells) - expected_cells)
        raise RecurrentScaleCampaignError(
            f"scale campaign report matrix is incomplete: missing={missing} extra={extra}"
        )
    runtime_payloads = [
        _mapping(
            _mapping(report.get("runtime_provenance"), field="runtime_provenance").get(
                "payload"
            ),
            field="runtime_provenance.payload",
        )
        for report in cells.values()
    ]
    runtime_digests = {
        _sha256(payload.get("exact_digest"), field="runtime_provenance.exact_digest")
        for payload in runtime_payloads
    }
    runtime_file_digests = {
        _sha256(
            _mapping(
                _mapping(
                    report.get("runtime_provenance"), field="runtime_provenance"
                ).get("file"),
                field="runtime_provenance.file",
            ).get("sha256"),
            field="runtime_provenance.file.sha256",
        )
        for report in cells.values()
    }
    if len(runtime_digests) != 1 or len(runtime_file_digests) != 1:
        raise RecurrentScaleCampaignError(
            "scale campaign cells do not share one pinned runtime provenance"
        )
    runtime_payload = runtime_payloads[0]
    for learner_seed in expected_seeds:
        learner_streams = {
            _mapping(
                cells[(learner_seed, arm)].get("evaluations"),
                field="evaluations",
            ).get("sampling_stream_id")
            for arm in RECURRENT_SCALE_ARMS
        }
        if len(learner_streams) != 1:
            raise RecurrentScaleCampaignError(
                "paired scale arms did not use one arm-independent sampling stream"
            )

    learner_rows: list[dict[str, object]] = []
    exact_beats_both = 0
    exact_nonzero = 0
    exact_beats_shuffled = 0
    argmax_improves = 0
    broad_nonregression = True
    dominant_share_max = 0.0
    heuristic_count = 0
    unsupported_count = 0
    treatment_delivery_passed = True
    for learner_seed in expected_seeds:
        summaries = {
            arm: _evaluation_mode_summaries(cells[(learner_seed, arm)])
            for arm in RECURRENT_SCALE_ARMS
        }
        for mode in _ACTION_SELECTION_MODES:
            _verify_paired_fixture_run_identities(
                {arm: summaries[arm][mode] for arm in RECURRENT_SCALE_ARMS}
            )
        sampled_alive = {
            arm: _terminal_alive_total(
                summaries[arm][PUBLIC_RECURRENT_SAMPLED_SELECTION],
                context="fixture:carrion_only",
            )
            for arm in RECURRENT_SCALE_ARMS
        }
        argmax_alive = {
            arm: _terminal_alive_total(
                summaries[arm][PUBLIC_RECURRENT_ARGMAX_SELECTION],
                context="fixture:carrion_only",
            )
            for arm in RECURRENT_SCALE_ARMS
        }
        beats_both = sampled_alive[EXACT_ARM] > max(
            sampled_alive[BASE_ARM], sampled_alive[SHUFFLED_ARM]
        )
        nonzero = sampled_alive[EXACT_ARM] > 0
        beats_shuffled = sampled_alive[EXACT_ARM] > sampled_alive[SHUFFLED_ARM]
        argmax_better = argmax_alive[EXACT_ARM] > max(
            argmax_alive[BASE_ARM], argmax_alive[SHUFFLED_ARM]
        )
        exact_beats_both += int(beats_both)
        exact_nonzero += int(nonzero)
        exact_beats_shuffled += int(beats_shuffled)
        argmax_improves += int(argmax_better)
        learner_nonregression = _broad_alive_birth_nonregression(
            summaries[EXACT_ARM],
            summaries[BASE_ARM],
            summaries[SHUFFLED_ARM],
        )
        broad_nonregression = broad_nonregression and learner_nonregression
        treatment_by_arm = {
            arm: _mapping(
                _mapping(
                    cells[(learner_seed, arm)].get("training"), field="training"
                ).get("treatment_delivery"),
                field="training.treatment_delivery",
            )
            for arm in RECURRENT_SCALE_ARMS
        }
        for treatment_arm in (EXACT_ARM, SHUFFLED_ARM):
            delivery = treatment_by_arm[treatment_arm]
            treatment_delivery_passed = treatment_delivery_passed and bool(
                delivery.get("meets_preregistered_delivery_floor") is True
            )
        for mode_summary in summaries[EXACT_ARM].values():
            for context in mode_summary["contexts"]:
                context_action_counts: Counter[str] = Counter()
                for run in context["runs"]:
                    requested_counts = _mapping(
                        run.get("requested_action_counts"),
                        field="requested_action_counts",
                    )
                    for action, count in requested_counts.items():
                        context_action_counts[_text(action, field="action")] += (
                            _nonnegative_int(count, field="requested action count")
                        )
                    heuristic_count += _nonnegative_int(
                        run.get("heuristic_action_source_count"),
                        field="heuristic action source count",
                    )
                    unsupported_count += _nonnegative_int(
                        run.get("unsupported_requested_action_count"),
                        field="unsupported requested action count",
                    )
                context_action_total = sum(context_action_counts.values())
                if context_action_total <= 0:
                    raise RecurrentScaleCampaignError(
                        "candidate context has no requested actions"
                    )
                dominant_share_max = max(
                    dominant_share_max,
                    max(context_action_counts.values()) / context_action_total,
                )
        learner_rows.append(
            {
                "learner_seed": learner_seed,
                "sampled_carrion_terminal_alive": sampled_alive,
                "argmax_carrion_terminal_alive": argmax_alive,
                "exact_beats_both_controls": beats_both,
                "exact_nonzero_carrion_survival": nonzero,
                "exact_beats_shuffled": beats_shuffled,
                "argmax_exact_beats_both_controls": argmax_better,
                "broad_alive_birth_nonregression": learner_nonregression,
                "treatment_delivery": {
                    arm: dict(treatment_by_arm[arm]) for arm in RECURRENT_SCALE_ARMS
                },
            }
        )

    acceptance = _mapping(preregistration.get("acceptance"), field="acceptance")
    gates = {
        "all_learner_runs_present": len(cells) == len(expected_cells),
        "exact_beats_both_controls": exact_beats_both
        >= _positive_int(
            acceptance.get("exact_beats_both_controls_learner_count_min"),
            field="exact beats both threshold",
        ),
        "exact_nonzero_carrion_survival": exact_nonzero
        >= _positive_int(
            acceptance.get("exact_nonzero_carrion_survival_learner_count_min"),
            field="nonzero survival threshold",
        ),
        "exact_beats_shuffled": exact_beats_shuffled
        >= _positive_int(
            acceptance.get("exact_beats_shuffled_learner_count_min"),
            field="shuffled threshold",
        ),
        "argmax_carrion_improves": argmax_improves
        >= _positive_int(
            acceptance.get("argmax_carrion_improves_learner_count_min"),
            field="argmax threshold",
        ),
        "dominant_action_share": dominant_share_max
        <= _float(
            acceptance.get("dominant_requested_action_share_max"),
            field="dominant action cap",
        ),
        "zero_heuristic_actions": heuristic_count
        <= _nonnegative_int(
            acceptance.get("heuristic_action_source_count_max"),
            field="heuristic cap",
        ),
        "zero_unsupported_actions": unsupported_count
        <= _nonnegative_int(
            acceptance.get("unsupported_requested_action_count_max"),
            field="unsupported cap",
        ),
        "broad_alive_birth_strict_per_seed_nonregression": broad_nonregression,
        "treatment_delivery": treatment_delivery_passed,
        "validation_and_lockbox_sealed": True,
        "runtime_provenance_consistent": True,
    }
    total_worlds = sum(
        _positive_int(
            _mapping(report.get("training"), field="training").get("worlds"),
            field="training.worlds",
        )
        for report in cells.values()
    )
    if total_worlds != RECURRENT_SCALE_TOTAL_TRAINING_WORLDS:
        raise RecurrentScaleCampaignError("campaign world reconciliation failed")
    input_reports = {
        _text(report.get("run_id"), field="run_id"): _campaign_input_report_manifest(
            report,
            preregistration=preregistration,
        )
        for _, report in sorted(cells.items())
    }
    analysis: dict[str, object] = {
        "schema_version": RECURRENT_SCALE_CAMPAIGN_ANALYSIS_SCHEMA_VERSION,
        "preregistration_digest": preregistration["exact_digest"],
        "source": dict(_mapping(preregistration.get("source"), field="source")),
        "runtime_provenance": {
            "exact_digest": next(iter(runtime_digests)),
            "file_sha256": next(iter(runtime_file_digests)),
            "dependency_freeze_sha256": _sha256(
                _mapping(
                    runtime_payload.get("dependency_freeze"),
                    field="runtime_provenance.dependency_freeze",
                ).get("packages_sha256"),
                field="runtime_provenance.dependency_freeze.packages_sha256",
            ),
            "torch_sha256": stable_payload_digest(
                _mapping(runtime_payload.get("torch"), field="runtime_provenance.torch")
            ),
            "device_sha256": stable_payload_digest(
                _mapping(
                    runtime_payload.get("device"), field="runtime_provenance.device"
                )
            ),
            "workers_sha256": stable_payload_digest(
                _mapping(
                    runtime_payload.get("workers"), field="runtime_provenance.workers"
                )
            ),
        },
        "report_count": len(cells),
        "input_reports": input_reports,
        "total_training_worlds": total_worlds,
        "learner_rows": learner_rows,
        "gate_statistics": {
            "exact_beats_both_controls_learner_count": exact_beats_both,
            "exact_nonzero_carrion_survival_learner_count": exact_nonzero,
            "exact_beats_shuffled_learner_count": exact_beats_shuffled,
            "argmax_carrion_improves_learner_count": argmax_improves,
            "exact_dominant_requested_action_share_max": dominant_share_max,
            "exact_heuristic_action_source_count": heuristic_count,
            "exact_unsupported_requested_action_count": unsupported_count,
        },
        "gates": gates,
        "accepted": all(gates.values()),
        "scientific_interpretation": (
            "development_scale_acceptance_passed"
            if all(gates.values())
            else "development_scale_acceptance_failed_no_promotion"
        ),
        "lifecycle": {
            "development_only": True,
            "campaign_slice_consumed": True,
            "runtime_integration_authorized": False,
            "promotion_authorized": False,
            "validation_seeds_accessed": False,
            "lockbox_seeds_accessed": False,
            "gate_relaxation_authorized": False,
        },
    }
    analysis["exact_digest"] = stable_payload_digest(analysis)
    return analysis


def recurrent_candidate_outcome_summary(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Return artifact-label-independent outcomes for exact CPU comparison."""

    contexts = [
        _candidate_context_summary(
            _mapping(evaluation.get("broad"), field="evaluation.broad")
        )
    ]
    fixtures = evaluation.get("fixtures")
    if not isinstance(fixtures, list):
        raise RecurrentScaleCampaignError("evaluation fixtures must be a list")
    contexts.extend(
        _candidate_context_summary(_mapping(fixture, field="evaluation.fixture"))
        for fixture in fixtures
    )
    return {
        "evaluation_contract": dict(
            _mapping(evaluation.get("evaluation_contract"), field="evaluation_contract")
        ),
        "seed_plan_digest": _mapping(
            evaluation.get("seed_plan"), field="seed_plan"
        ).get("digest"),
        "contexts": contexts,
    }


def _counterfactual_config(
    preregistration: Mapping[str, object],
    *,
    learner_seed: int,
    arm: str,
    workers: int,
) -> RecurrentCounterfactualExperimentConfig | None:
    if arm == BASE_ARM:
        return None
    if arm not in {EXACT_ARM, SHUFFLED_ARM}:
        raise RecurrentScaleCampaignError("unsupported scale arm")
    payload = _mapping(preregistration.get("counterfactual"), field="counterfactual")
    auxiliary = _mapping(payload.get("auxiliary"), field="counterfactual.auxiliary")
    outcomes = _mapping(
        payload.get("outcome_scalarization"), field="outcome_scalarization"
    )
    horizons = tuple(
        _positive_int(value, field="relative_horizons[]")
        for value in _sequence(
            payload.get("relative_horizons"), field="relative_horizons"
        )
    )
    raw_weights = tuple(
        _float(value, field="relative_horizon_weights[]")
        for value in _sequence(
            payload.get("relative_horizon_weights"),
            field="relative_horizon_weights",
        )
    )
    terminal_weight = _float(
        payload.get("absolute_terminal_weight"), field="absolute_terminal_weight"
    )
    relative_mass = 1.0 - terminal_weight
    if relative_mass <= 0.0 or len(horizons) != len(raw_weights):
        raise RecurrentScaleCampaignError("counterfactual horizon weights are invalid")
    normalized_relative_weights = tuple(
        weight / relative_mass for weight in raw_weights
    )
    permutation_mode = (
        RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
        if arm == SHUFFLED_ARM
        else RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
    )
    permutation_seed = (
        _derived_63_bit_seed(
            {
                "policy": "scale_counterfactual_label_permutation_master_v1",
                "preregistration_digest": preregistration["exact_digest"],
                "learner_seed": learner_seed,
            }
        )
        if arm == SHUFFLED_ARM
        else None
    )
    uncertainty = _mapping(payload.get("uncertainty"), field="uncertainty")
    collection = RecurrentCounterfactualCollectionConfig(
        horizons=horizons,
        gamma=_float(
            _mapping(
                _mapping(preregistration.get("training"), field="training").get("ppo"),
                field="training.ppo",
            ).get("gamma"),
            field="gamma",
        ),
        continuation_tape_count=_positive_int(
            payload.get("independent_rng_tapes_per_branch"),
            field="independent_rng_tapes_per_branch",
        ),
        terminal_target_world_tick=_positive_int(
            payload.get("absolute_terminal_target_tick"),
            field="absolute_terminal_target_tick",
        ),
        uncertainty_penalty=_float(
            uncertainty.get("lambda"), field="uncertainty.lambda"
        ),
    )
    auxiliary_config = RecurrentCounterfactualAuxiliaryConfig(
        scalarization=CounterfactualHorizonScalarization(
            horizon_weights=tuple(
                zip(horizons, normalized_relative_weights, strict=True)
            ),
            focal_discounted_return_weight=_float(
                outcomes.get("focal_discounted_return_weight"),
                field="focal_discounted_return_weight",
            ),
            focal_terminal_alive_weight=_float(
                outcomes.get("focal_terminal_alive_weight"),
                field="focal_terminal_alive_weight",
            ),
            population_alive_weight=_float(
                outcomes.get("population_alive_weight"),
                field="population_alive_weight",
            ),
            births_during_horizon_weight=_float(
                outcomes.get("births_during_horizon_weight"),
                field="births_during_horizon_weight",
            ),
            deaths_during_horizon_weight=_float(
                outcomes.get("deaths_during_horizon_weight"),
                field="deaths_during_horizon_weight",
            ),
        ),
        temperature=_float(auxiliary.get("temperature"), field="temperature"),
        advantage_clip=_float(auxiliary.get("advantage_clip"), field="advantage_clip"),
        policy_improvement_coefficient=_float(
            auxiliary.get("policy_improvement_coefficient"),
            field="policy_improvement_coefficient",
        ),
        behavior_kl_coefficient=_float(
            auxiliary.get("behavior_kl_coefficient"),
            field="behavior_kl_coefficient",
        ),
        value_loss_coefficient=_float(
            auxiliary.get("value_loss_coefficient"),
            field="value_loss_coefficient",
        ),
        value_target_mode=RECURRENT_COUNTERFACTUAL_VALUE_TARGET_DISABLED,
        target_permutation_mode=permutation_mode,
        target_permutation_seed=permutation_seed,
        terminal_target_weight=terminal_weight,
    )
    return RecurrentCounterfactualExperimentConfig(
        collection=collection,
        auxiliary=auxiliary_config,
        step=RecurrentCounterfactualAuxiliaryStepConfig(
            learning_rate_multiplier=_float(
                auxiliary.get("learning_rate_multiplier"),
                field="learning_rate_multiplier",
            ),
            max_gradient_norm=_float(
                auxiliary.get("max_gradient_norm"), field="max_gradient_norm"
            ),
            mean_behavior_kl_limit=_float(
                auxiliary.get("mean_behavior_kl_limit"),
                field="mean_behavior_kl_limit",
            ),
            max_state_behavior_kl_limit=_float(
                auxiliary.get("max_state_behavior_kl_limit"),
                field="max_state_behavior_kl_limit",
            ),
        ),
        bundles_per_update=_positive_int(
            payload.get("bundles_per_update"), field="bundles_per_update"
        ),
        branch_tick_candidates=tuple(
            _positive_int(value, field="branch_tick_candidates[]")
            for value in _sequence(
                payload.get("branch_tick_candidates"),
                field="branch_tick_candidates",
            )
        ),
        workers=workers,
    )


def _scale_update_payload(
    update: RecurrentTrainingUpdateResult,
    *,
    run_id: str,
    raw_collection_evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": RECURRENT_SCALE_UPDATE_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "update_index": update.update_index,
        "tasks": [asdict(task) for task in update.tasks],
        "rollout": asdict(update.rollout),
        "optimizer": asdict(update.optimizer),
        "counterfactual_collection": (
            None
            if update.counterfactual_collection is None
            else {
                "contract_version": update.counterfactual_collection.contract_version,
                "source_model_state_sha256": (
                    update.counterfactual_collection.source_model_state_sha256
                ),
                "source_artifact_digest": (
                    update.counterfactual_collection.source_artifact_digest
                ),
                "config": asdict(update.counterfactual_collection.config),
                "bundle_exact_digests": [
                    bundle.exact_digest
                    for bundle in update.counterfactual_collection.bundles
                ],
                "aggregate_compute": dict(
                    update.counterfactual_collection.aggregate_compute
                ),
                "raw_evidence": dict(raw_collection_evidence or {}),
            }
        ),
        "counterfactual_auxiliary": (
            None
            if update.counterfactual_auxiliary is None
            else asdict(update.counterfactual_auxiliary)
        ),
    }


def _treatment_delivery_summary(
    update_records: Sequence[Mapping[str, object]],
    *,
    arm: str,
) -> dict[str, object]:
    diagnostics = [record.get("counterfactual_auxiliary") for record in update_records]
    if arm == BASE_ARM:
        if any(value is not None for value in diagnostics):
            raise RecurrentScaleCampaignError(
                "base arm unexpectedly performed a counterfactual auxiliary update"
            )
        return {
            "treatment_expected": False,
            "attempted_update_count": 0,
            "accepted_update_count": 0,
            "parameter_delta_l2_sum": 0.0,
            "all_transactions_within_kl_bounds": True,
            "meets_preregistered_delivery_floor": True,
        }
    parsed = [
        _mapping(value, field="counterfactual_auxiliary") for value in diagnostics
    ]
    accepted = sum(value.get("accepted") is True for value in parsed)
    parameter_delta = sum(
        _float(value.get("parameter_delta_l2"), field="parameter_delta_l2")
        for value in parsed
        if value.get("accepted") is True
    )
    within_bounds = all(
        _float(
            value.get("mean_behavior_kl_old_to_post"),
            field="mean_behavior_kl_old_to_post",
        )
        <= _float(value.get("mean_behavior_kl_limit"), field="mean_behavior_kl_limit")
        and _float(
            value.get("max_state_behavior_kl_old_to_post"),
            field="max_state_behavior_kl_old_to_post",
        )
        <= _float(
            value.get("max_state_behavior_kl_limit"),
            field="max_state_behavior_kl_limit",
        )
        for value in parsed
        if value.get("accepted") is True
    )
    floor = 12
    delivered = (
        len(parsed) == RECURRENT_SCALE_UPDATE_COUNT
        and accepted >= floor
        and parameter_delta > 0.0
        and within_bounds
    )
    return {
        "treatment_expected": True,
        "attempted_update_count": len(parsed),
        "accepted_update_count": accepted,
        "accepted_update_count_min": floor,
        "parameter_delta_l2_sum": parameter_delta,
        "parameter_delta_l2_sum_min_exclusive": 0.0,
        "all_transactions_within_kl_bounds": within_bounds,
        "meets_preregistered_delivery_floor": delivered,
    }


def _write_collection_evidence(
    update: RecurrentTrainingUpdateResult,
    *,
    evidence_directory: Path,
) -> dict[str, object] | None:
    collection = update.counterfactual_collection
    if collection is None:
        return None
    path = evidence_directory / f"update-{update.update_index:04d}-collection.json"
    write_atomic_json(path, asdict(collection))
    return _file_reference(path)


def _load_update_record(path: Path, *, index: int) -> dict[str, object]:
    record = load_strict_json(path)
    if (
        record.get("schema_version") != RECURRENT_SCALE_UPDATE_RECORD_SCHEMA_VERSION
        or record.get("update_index") != index
    ):
        raise RecurrentScaleCampaignError("scale update journal is incomplete")
    digest = _sha256(record.get("exact_digest"), field="update.exact_digest")
    unsigned = dict(record)
    unsigned.pop("exact_digest", None)
    if stable_payload_digest(unsigned) != digest:
        raise RecurrentScaleCampaignError("scale update record digest mismatched")
    return record


def _full_world_replay_manifest(
    evaluations: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    checks: list[object] = []
    contracts: dict[str, object] = {}
    for mode in _ACTION_SELECTION_MODES:
        report = _mapping(evaluations.get(mode), field=f"evaluations.{mode}")
        replay = _mapping(
            report.get("replay_verification"), field="replay_verification"
        )
        mode_checks = replay.get("checks")
        if replay.get("all_passed") is not True or not isinstance(mode_checks, list):
            raise RecurrentScaleCampaignError(
                "full-world replay verification is incomplete"
            )
        checks.extend(mode_checks)
        contracts[mode] = dict(
            _mapping(report.get("evaluation_contract"), field="evaluation_contract")
        )
    if not checks:
        raise RecurrentScaleCampaignError("full-world replay manifest has no checks")
    verification_path = Path(__file__).with_name("recurrent_evaluation.py")
    return {
        "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": stable_payload_digest(
            {"checks": checks, "contracts": contracts}
        ),
        "replay_engine_contract_sha256": stable_payload_digest(contracts),
        "environment_seed_registry_sha256": SCALE_DEVELOPMENT_CANONICAL_SHA256,
        "environment_seed_roles": ["scale_selection"],
        "scenario_names": ["broad", "carrion_only"],
        "tick_horizons": [RECURRENT_SCALE_ROLLOUT_TICKS],
        "world_count": len(checks),
        "replay_verified_world_count": len(checks),
        "policy_sampling_stream_count": (
            1 + RECURRENT_SCALE_POLICY_SAMPLING_STREAM_COUNT
        ),
        "all_replays_exact": True,
        "verification_runner": RECURRENT_SCALE_FULL_WORLD_VERIFICATION_RUNNER,
        "verification_runner_sha256": _file_sha256(verification_path),
    }


def _candidate_context_summary(context: Mapping[str, object]) -> dict[str, object]:
    policies = _mapping(context.get("policies"), field="context.policies")
    candidate = _mapping(
        policies.get("public_recurrent"), field="policies.public_recurrent"
    )
    linear = _mapping(policies.get("mind_v3_linear"), field="policies.mind_v3_linear")
    kept_fields = (
        "context",
        "seed",
        "policy_sampling_seed",
        "behavior_digest",
        "horizon_ticks",
        "ticks_executed",
        "terminal_alive",
        "births",
        "deaths",
        "reward_total",
        "reward_component_totals",
        "trajectory_record_count",
        "policy_decision_record_count",
        "requested_action_counts",
        "dominant_requested_action",
        "dominant_requested_action_count",
        "dominant_requested_action_share",
        "unsupported_requested_action_count",
        "heuristic_action_source_count",
        "eat_requested_count",
        "eat_without_positive_resource_gain_count",
        "eat_without_positive_resource_gain_share",
        "learned_masked_distribution",
    )
    compact_runs = _compact_policy_runs(
        candidate,
        kept_fields=kept_fields,
        field="candidate",
    )
    linear_runs = _compact_policy_runs(
        linear,
        kept_fields=kept_fields,
        field="mind_v3_linear",
    )
    compact_runs.sort(
        key=lambda run: (
            int(run["seed"]),
            -1
            if run["policy_sampling_seed"] is None
            else int(run["policy_sampling_seed"]),
        )
    )
    return {
        "context": compact_runs[0]["context"],
        "runs": compact_runs,
        "linear_runs": linear_runs,
    }


def _compact_policy_runs(
    policy: Mapping[str, object],
    *,
    kept_fields: Sequence[str],
    field: str,
) -> list[dict[str, object]]:
    runs = policy.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RecurrentScaleCampaignError(f"{field} evaluation runs are missing")
    result = [
        {name: run.get(name) for name in kept_fields}
        for run in (_mapping(value, field=f"{field} run") for value in runs)
    ]
    result.sort(
        key=lambda run: (
            int(run["seed"]),
            -1
            if run["policy_sampling_seed"] is None
            else int(run["policy_sampling_seed"]),
        )
    )
    return result


def _evaluation_mode_summaries(
    report: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    modes = _mapping(
        _mapping(report.get("evaluations"), field="evaluations").get("modes"),
        field="evaluations.modes",
    )
    return {
        mode: _mapping(
            _mapping(modes.get(mode), field=f"modes.{mode}").get(
                "candidate_outcome_summary"
            ),
            field=f"modes.{mode}.candidate_outcome_summary",
        )
        for mode in _ACTION_SELECTION_MODES
    }


def _verify_paired_fixture_run_identities(
    summaries_by_arm: Mapping[str, Mapping[str, object]],
) -> None:
    identities_by_arm = {
        arm: set(
            _runs_by_identity(
                summaries_by_arm[arm],
                context="fixture:carrion_only",
            )
        )
        for arm in RECURRENT_SCALE_ARMS
    }
    reference = identities_by_arm[BASE_ARM]
    if any(identities_by_arm[arm] != reference for arm in (EXACT_ARM, SHUFFLED_ARM)):
        raise RecurrentScaleCampaignError(
            "paired carrion fixture evaluation identities differ across arms"
        )


def _terminal_alive_total(summary: Mapping[str, object], *, context: str) -> int:
    return sum(
        _nonnegative_int(
            run.get("terminal_alive"),
            field="terminal_alive",
        )
        for run in _runs_by_identity(summary, context=context).values()
    )


def _broad_alive_birth_nonregression(
    exact_modes: Mapping[str, Mapping[str, object]],
    base_modes: Mapping[str, Mapping[str, object]],
    shuffled_modes: Mapping[str, Mapping[str, object]],
) -> bool:
    for mode in _ACTION_SELECTION_MODES:
        exact = _runs_by_identity(exact_modes[mode], context="broad_default")
        base = _runs_by_identity(base_modes[mode], context="broad_default")
        shuffled = _runs_by_identity(shuffled_modes[mode], context="broad_default")
        exact_linear = _linear_runs_by_seed(exact_modes[mode], context="broad_default")
        base_linear = _linear_runs_by_seed(base_modes[mode], context="broad_default")
        shuffled_linear = _linear_runs_by_seed(
            shuffled_modes[mode], context="broad_default"
        )
        if set(exact) != set(base) or set(exact) != set(shuffled):
            raise RecurrentScaleCampaignError(
                "paired broad evaluation identities differ across arms"
            )
        if exact_linear != base_linear or exact_linear != shuffled_linear:
            raise RecurrentScaleCampaignError(
                "fixed linear broad controls differ across paired arms"
            )
        for identity, exact_run in exact.items():
            linear_run = exact_linear.get(identity[0])
            if linear_run is None:
                raise RecurrentScaleCampaignError(
                    "candidate broad run has no same-seed linear control"
                )
            for control in (base[identity], shuffled[identity]):
                if _nonnegative_int(
                    exact_run.get("terminal_alive"), field="terminal_alive"
                ) < _nonnegative_int(
                    control.get("terminal_alive"), field="terminal_alive"
                ) or _nonnegative_int(
                    exact_run.get("births"), field="births"
                ) < _nonnegative_int(control.get("births"), field="births"):
                    return False
            if _nonnegative_int(
                exact_run.get("terminal_alive"), field="terminal_alive"
            ) < _nonnegative_int(
                linear_run.get("terminal_alive"), field="terminal_alive"
            ) or _nonnegative_int(
                exact_run.get("births"), field="births"
            ) < _nonnegative_int(linear_run.get("births"), field="births"):
                return False
    return True


def _runs_by_identity(
    summary: Mapping[str, object],
    *,
    context: str,
) -> dict[tuple[int, int | None], Mapping[str, object]]:
    contexts = summary.get("contexts")
    if not isinstance(contexts, list):
        raise RecurrentScaleCampaignError("candidate contexts are missing")
    for value in contexts:
        parsed = _mapping(value, field="candidate context")
        if parsed.get("context") != context:
            continue
        runs = parsed.get("runs")
        if not isinstance(runs, list):
            break
        result: dict[tuple[int, int | None], Mapping[str, object]] = {}
        for run in runs:
            parsed_run = _mapping(run, field="candidate run")
            seed = _positive_int(parsed_run.get("seed"), field="seed")
            sampling_seed = parsed_run.get("policy_sampling_seed")
            if sampling_seed is not None:
                sampling_seed = _nonnegative_int(
                    sampling_seed, field="policy_sampling_seed"
                )
            identity = (seed, sampling_seed)
            if identity in result:
                raise RecurrentScaleCampaignError("duplicate candidate run identity")
            result[identity] = parsed_run
        return result
    raise RecurrentScaleCampaignError(f"candidate context {context!r} is missing")


def _linear_runs_by_seed(
    summary: Mapping[str, object],
    *,
    context: str,
) -> dict[int, Mapping[str, object]]:
    contexts = summary.get("contexts")
    if not isinstance(contexts, list):
        raise RecurrentScaleCampaignError("candidate contexts are missing")
    for value in contexts:
        parsed = _mapping(value, field="candidate context")
        if parsed.get("context") != context:
            continue
        runs = parsed.get("linear_runs")
        if not isinstance(runs, list):
            break
        result: dict[int, Mapping[str, object]] = {}
        for run in runs:
            parsed_run = _mapping(run, field="linear run")
            seed = _positive_int(parsed_run.get("seed"), field="linear seed")
            if seed in result:
                raise RecurrentScaleCampaignError("duplicate linear run seed")
            result[seed] = parsed_run
        return result
    raise RecurrentScaleCampaignError(f"linear control context {context!r} is missing")


def _scale_training_seed_provenance(
    schedule: Sequence[Sequence[object]],
) -> dict[str, object]:
    by_role: dict[str, set[int]] = {"scale_train": set(), "scale_curriculum": set()}
    for update in schedule:
        for task in update:
            role = getattr(task, "seed_role", None)
            seed = getattr(task, "environment_seed", None)
            if role not in by_role or seed not in SCALE_DEVELOPMENT_SEED_REGISTRY[role]:
                raise RecurrentScaleCampaignError(
                    "scale schedule seed provenance drifted"
                )
            by_role[role].add(seed)
    ordered = {
        role: [
            seed
            for seed in SCALE_DEVELOPMENT_SEED_REGISTRY[role]
            if seed in by_role[role]
        ]
        for role in ("scale_train", "scale_curriculum")
    }
    return {
        "schema_version": "mind_public_recurrent_scale_training_seed_provenance_v1",
        "seed_registry_contract": RECURRENT_TRAINING_SEED_REGISTRY_SCALE_DEVELOPMENT,
        "environment_seed_roles": ["scale_train", "scale_curriculum"],
        "environment_seeds_by_role": ordered,
        "observed_environment_seed_count": sum(
            len(values) for values in ordered.values()
        ),
        "validation_seed_count": 0,
        "lockbox_seed_count": 0,
    }


def _campaign_input_report_manifest(
    report: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    source = _mapping(report.get("source"), field="source")
    configuration = _mapping(report.get("configuration"), field="configuration")
    training = _mapping(report.get("training"), field="training")
    artifact = _mapping(report.get("artifact"), field="artifact")
    runtime_evidence = _mapping(
        report.get("runtime_provenance"), field="runtime_provenance"
    )
    runtime_payload = _mapping(
        runtime_evidence.get("payload"), field="runtime_provenance.payload"
    )
    runtime_file = _mapping(
        runtime_evidence.get("file"), field="runtime_provenance.file"
    )
    evaluations = _mapping(report.get("evaluations"), field="evaluations")
    modes = _mapping(evaluations.get("modes"), field="evaluations.modes")
    checkpoint = _mapping(training.get("checkpoint"), field="training.checkpoint")
    seed_provenance = _mapping(
        training.get("environment_seed_provenance"),
        field="training.environment_seed_provenance",
    )
    update_record_digests = training.get("update_record_digests")
    if not isinstance(update_record_digests, list):
        raise RecurrentScaleCampaignError("update digest evidence must be a list")
    update_journals = training.get("update_journals")
    if not isinstance(update_journals, list):
        raise RecurrentScaleCampaignError("update journal file evidence must be a list")
    evaluation_files: dict[str, object] = {}
    for mode in _ACTION_SELECTION_MODES:
        evidence = _mapping(modes.get(mode), field=f"evaluations.modes.{mode}")
        in_memory = _mapping(
            evidence.get("in_memory_report"),
            field=f"evaluations.modes.{mode}.in_memory_report",
        )
        artifact_cpu = _mapping(
            evidence.get("artifact_cpu_report"),
            field=f"evaluations.modes.{mode}.artifact_cpu_report",
        )
        evaluation_files[mode] = {
            "in_memory_report_sha256": _sha256(
                in_memory.get("sha256"),
                field=f"evaluations.modes.{mode}.in_memory_report.sha256",
            ),
            "artifact_cpu_report_sha256": _sha256(
                artifact_cpu.get("sha256"),
                field=f"evaluations.modes.{mode}.artifact_cpu_report.sha256",
            ),
            "candidate_outcome_summary_sha256": stable_payload_digest(
                _mapping(
                    evidence.get("candidate_outcome_summary"),
                    field=f"evaluations.modes.{mode}.candidate_outcome_summary",
                )
            ),
        }
    full_world_manifest = _mapping(
        artifact.get("full_world_replay_manifest"),
        field="artifact.full_world_replay_manifest",
    )
    preregistered_seed_contract = _mapping(
        preregistration.get("seed_contract"), field="preregistration.seed_contract"
    )
    manifest: dict[str, object] = {
        "report_exact_digest": _sha256(
            report.get("exact_digest"), field="report.exact_digest"
        ),
        "artifact_sha256": _sha256(
            artifact.get("artifact_sha256"), field="artifact.artifact_sha256"
        ),
        "artifact_file_sha256": _sha256(
            _mapping(artifact.get("file"), field="artifact.file").get("sha256"),
            field="artifact.file.sha256",
        ),
        "runtime_provenance": {
            "exact_digest": _sha256(
                runtime_payload.get("exact_digest"),
                field="runtime_provenance.exact_digest",
            ),
            "payload_sha256": stable_payload_digest(runtime_payload),
            "file_sha256": _sha256(
                runtime_file.get("sha256"),
                field="runtime_provenance.file.sha256",
            ),
            "dependency_freeze_sha256": _sha256(
                _mapping(
                    runtime_payload.get("dependency_freeze"),
                    field="runtime_provenance.dependency_freeze",
                ).get("packages_sha256"),
                field="runtime_provenance.dependency_freeze.packages_sha256",
            ),
        },
        "full_world_replay_manifest_sha256": stable_payload_digest(full_world_manifest),
        "source": {
            "commit": _text(source.get("commit"), field="source.commit"),
            "manifest_sha256": _sha256(
                source.get("manifest_sha256"), field="source.manifest_sha256"
            ),
        },
        "configuration_sha256": stable_payload_digest(configuration),
        "schedule_sha256": _sha256(
            configuration.get("schedule_sha256"),
            field="configuration.schedule_sha256",
        ),
        "seed_pins": {
            "seed_registry_sha256": _sha256(
                preregistered_seed_contract.get("registry_sha256"),
                field="seed_contract.registry_sha256",
            ),
            "training_seed_provenance_sha256": stable_payload_digest(seed_provenance),
            "selection_seed_plan_sha256": _sha256(
                evaluations.get("selection_seed_plan_sha256"),
                field="evaluations.selection_seed_plan_sha256",
            ),
        },
        "checkpoint_sha256": _sha256(
            checkpoint.get("sha256"), field="training.checkpoint.sha256"
        ),
        "update_record_digests": [
            _sha256(value, field="training.update_record_digests[]")
            for value in update_record_digests
        ],
        "update_journal_file_sha256": [
            _sha256(
                _mapping(reference, field="training.update_journals[]").get("sha256"),
                field="training.update_journals[].sha256",
            )
            for reference in update_journals
        ],
        "evaluation_file_sha256": evaluation_files,
    }
    manifest["exact_digest"] = stable_payload_digest(manifest)
    return manifest


def _require_exact_clean_source(
    *,
    source_commit: str,
    source_manifest_sha256: str,
) -> None:
    head, clean = _git_source_state()
    if head != source_commit:
        raise RecurrentScaleCampaignError(
            "checked-out Git HEAD does not match the preregistered source commit"
        )
    if not clean:
        raise RecurrentScaleCampaignError(
            "scale training requires a clean, reproducible source tree"
        )
    observed = source_file_hash_manifest(_REPOSITORY_ROOT)["aggregate_sha256"]
    if observed != source_manifest_sha256:
        raise RecurrentScaleCampaignError(
            "runtime source manifest does not match the preregistration"
        )


def _source_still_exact(source_commit: str, source_manifest_sha256: str) -> bool:
    try:
        _require_exact_clean_source(
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
        )
    except RecurrentScaleCampaignError:
        return False
    return True


def _git_source_state() -> tuple[str | None, bool]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return head, not bool(status.strip())


def _scale_sampling_stream_id(
    preregistration_digest: str,
    *,
    learner_seed: int,
) -> str:
    return f"scale-v1-arm-independent:{preregistration_digest}:learner-{learner_seed}"


def _derived_63_bit_seed(payload: Mapping[str, object]) -> int:
    return int(stable_payload_digest(payload)[:16], 16) & (2**63 - 1)


def _file_reference(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "byte_length": path.stat().st_size,
    }


def _verify_completed_report_evidence(
    report: Mapping[str, object],
    *,
    run_directory: Path,
    runtime_provenance_path: Path,
    preregistration: Mapping[str, object],
) -> None:
    runtime_evidence = _mapping(
        report.get("runtime_provenance"),
        field="runtime_provenance",
    )
    _verify_file_reference(
        _mapping(runtime_evidence.get("file"), field="runtime_provenance.file"),
        expected_path=runtime_provenance_path,
    )
    runtime_payload = _mapping(
        runtime_evidence.get("payload"),
        field="runtime_provenance.payload",
    )
    runtime_digest = _sha256(
        runtime_payload.get("exact_digest"),
        field="runtime_provenance.exact_digest",
    )
    if load_strict_json(runtime_provenance_path) != dict(runtime_payload):
        raise RecurrentScaleCampaignError(
            "completed report runtime provenance payload mismatched its pinned file"
        )
    training = _mapping(report.get("training"), field="training")
    learner_seed = _positive_int(report.get("learner_seed"), field="learner_seed")
    arm = _text(report.get("arm"), field="arm")
    _model, _ppo, _counterfactual, schedule = build_scale_run_components(
        preregistration,
        learner_seed=learner_seed,
        arm=arm,
        counterfactual_workers=1,
    )
    expected_seed_provenance = _scale_training_seed_provenance(schedule)
    _verify_file_reference(
        _mapping(training.get("checkpoint"), field="training.checkpoint"),
        expected_path=run_directory / "training-checkpoint.json",
    )
    checkpoint = load_recurrent_training_crash_checkpoint(
        run_directory / "training-checkpoint.json"
    )
    progress = _mapping(checkpoint.checkpoint.get("progress"), field="progress")
    if progress.get("completed_updates") != RECURRENT_SCALE_UPDATE_COUNT:
        raise RecurrentScaleCampaignError(
            "completed report checkpoint is not at the terminal update"
        )
    checkpoint_rng_state = _mapping(
        checkpoint.rng_state,
        field="checkpoint.rng_state",
    )
    if checkpoint_rng_state.get("scale_runtime_provenance_digest") != runtime_digest:
        raise RecurrentScaleCampaignError(
            "completed checkpoint runtime provenance digest mismatched"
        )
    checkpoint_source = _mapping(
        checkpoint.checkpoint.get("source"), field="checkpoint.source"
    )
    report_source = _mapping(report.get("source"), field="source")
    checkpoint_configuration = _mapping(
        checkpoint.checkpoint.get("configuration"),
        field="checkpoint.configuration",
    )
    report_configuration = _mapping(report.get("configuration"), field="configuration")
    if (
        checkpoint_source.get("source_commit") != report_source.get("commit")
        or checkpoint_source.get("source_manifest_sha256")
        != report_source.get("manifest_sha256")
        or checkpoint_source.get("seed_registry_digest")
        != SCALE_DEVELOPMENT_CANONICAL_SHA256
        or checkpoint_configuration.get("model_config")
        != report_configuration.get("model")
        or checkpoint_configuration.get("training_config") != report_configuration
        or progress.get("run_id") != report.get("run_id")
        or progress.get("learner_seed") != report.get("learner_seed")
    ):
        raise RecurrentScaleCampaignError(
            "completed checkpoint provenance or run contract mismatched"
        )
    artifact_evidence = _mapping(report.get("artifact"), field="artifact")
    artifact_path = run_directory / "frozen-policy.json"
    if Path(_text(artifact_evidence.get("path"), field="artifact.path")).resolve() != (
        artifact_path.resolve()
    ):
        raise RecurrentScaleCampaignError(
            "completed report frozen artifact path escaped its run directory"
        )
    _verify_file_reference(
        _mapping(artifact_evidence.get("file"), field="artifact.file"),
        expected_path=artifact_path,
    )
    loaded_artifact = load_frozen_recurrent_policy_artifact(artifact_path)
    if loaded_artifact.artifact.get("artifact_sha256") != artifact_evidence.get(
        "artifact_sha256"
    ):
        raise RecurrentScaleCampaignError(
            "completed report frozen artifact digest mismatched"
        )
    artifact_provenance = _mapping(
        loaded_artifact.artifact.get("provenance"),
        field="artifact.provenance",
    )
    artifact_run_metadata = _mapping(
        artifact_provenance.get("run_metadata"),
        field="artifact.provenance.run_metadata",
    )
    artifact_data_metadata = _mapping(
        artifact_provenance.get("data_metadata"),
        field="artifact.provenance.data_metadata",
    )
    artifact_integrity = _mapping(
        loaded_artifact.artifact.get("integrity"), field="artifact.integrity"
    )
    if (
        artifact_run_metadata.get("runtime_provenance_digest") != runtime_digest
        or artifact_run_metadata.get("run_id") != report.get("run_id")
        or artifact_run_metadata.get("arm") != report.get("arm")
        or artifact_provenance.get("source_commit") != report_source.get("commit")
        or artifact_provenance.get("source_manifest_sha256")
        != report_source.get("manifest_sha256")
        or artifact_provenance.get("experiment_config") != report_configuration
        or artifact_provenance.get("training_config") != report_configuration.get("ppo")
        or artifact_provenance.get("seed_registry_digest")
        != SCALE_DEVELOPMENT_CANONICAL_SHA256
        or artifact_provenance.get("learner_seed") != report.get("learner_seed")
        or artifact_run_metadata.get("preregistration_digest")
        != preregistration.get("exact_digest")
        or artifact_run_metadata.get("runtime_integration_authorized") is not False
        or artifact_run_metadata.get("promotion_authorized") is not False
        or artifact_data_metadata.get("policy_induced") is not True
        or artifact_data_metadata.get("runtime_provenance_digest") != runtime_digest
        or artifact_integrity.get("parameters_sha256")
        != checkpoint.checkpoint.get("parameters_sha256")
    ):
        raise RecurrentScaleCampaignError(
            "completed artifact provenance, configuration, or model state mismatched"
        )
    in_memory_manifest = _verify_completed_evaluation_evidence(
        report,
        run_directory=run_directory,
    )
    reported_manifest = dict(
        _mapping(
            artifact_evidence.get("full_world_replay_manifest"),
            field="artifact.full_world_replay_manifest",
        )
    )
    artifact_verification = _mapping(
        loaded_artifact.artifact.get("verification"),
        field="artifact.verification",
    )
    artifact_manifest = dict(
        _mapping(
            artifact_verification.get("full_world_replay_manifest"),
            field="artifact.verification.full_world_replay_manifest",
        )
    )
    if not (in_memory_manifest == reported_manifest == artifact_manifest):
        raise RecurrentScaleCampaignError(
            "completed full-world replay manifest detached from evaluation bytes"
        )
    update_digests = training.get("update_record_digests")
    update_journals = training.get("update_journals")
    if (
        not isinstance(update_digests, list)
        or len(update_digests) != RECURRENT_SCALE_UPDATE_COUNT
        or not isinstance(update_journals, list)
        or len(update_journals) != RECURRENT_SCALE_UPDATE_COUNT
    ):
        raise RecurrentScaleCampaignError(
            "completed report update digest journal is incomplete"
        )
    update_records: list[Mapping[str, object]] = []
    for index, expected_digest in enumerate(update_digests):
        update_path = run_directory / "updates" / f"update-{index:04d}.json"
        _verify_file_reference(
            _mapping(
                update_journals[index],
                field=f"training.update_journals[{index}]",
            ),
            expected_path=update_path,
        )
        record = _load_update_record(
            update_path,
            index=index,
        )
        if record.get("exact_digest") != expected_digest:
            raise RecurrentScaleCampaignError(
                "completed report update journal digest mismatched"
            )
        if record.get("run_id") != report.get("run_id") or record.get("tasks") != [
            asdict(task) for task in schedule[index]
        ]:
            raise RecurrentScaleCampaignError(
                "completed update journal run identity or task schedule drifted"
            )
        collection = record.get("counterfactual_collection")
        if arm == BASE_ARM:
            if collection is not None:
                raise RecurrentScaleCampaignError(
                    "base update unexpectedly references collection evidence"
                )
        else:
            collection_payload = _mapping(
                collection,
                field=f"updates[{index}].counterfactual_collection",
            )
            raw_reference = _mapping(
                collection_payload.get("raw_evidence"),
                field=f"updates[{index}].counterfactual_collection.raw_evidence",
            )
            raw_path = (
                run_directory / "evidence" / f"update-{index:04d}-collection.json"
            )
            _verify_file_reference(raw_reference, expected_path=raw_path)
            raw_collection = load_strict_json(raw_path)
            raw_bundles = raw_collection.get("bundles")
            if not isinstance(raw_bundles, list):
                raise RecurrentScaleCampaignError(
                    "raw counterfactual collection bundles are missing"
                )
            for bundle_index, bundle_value in enumerate(raw_bundles):
                bundle = dict(
                    _mapping(
                        bundle_value,
                        field=f"raw collection bundle[{bundle_index}]",
                    )
                )
                bundle_digest = _sha256(
                    bundle.pop("exact_digest", None),
                    field=f"raw collection bundle[{bundle_index}].exact_digest",
                )
                if bundle.get("aggregate_rows") is None:
                    bundle.pop("aggregate_rows", None)
                    bundle.pop("terminal_target", None)
                    bundle.pop("multi_tape_compute", None)
                bundle_task = dict(
                    _mapping(
                        bundle.get("task"),
                        field=f"raw collection bundle[{bundle_index}].task",
                    )
                )
                if bundle_task.get("branch_tick_stratum_index") is None:
                    bundle_task.pop("branch_tick_stratum_index", None)
                bundle["task"] = bundle_task
                if stable_payload_digest(bundle) != bundle_digest:
                    raise RecurrentScaleCampaignError(
                        "raw counterfactual collection bundle digest mismatched"
                    )
            raw_collection_unsigned = dict(raw_collection)
            raw_collection_digest = _sha256(
                raw_collection_unsigned.pop("exact_digest", None),
                field="raw collection exact_digest",
            )
            if stable_payload_digest(raw_collection_unsigned) != raw_collection_digest:
                raise RecurrentScaleCampaignError(
                    "raw counterfactual collection digest mismatched"
                )
            raw_projection = {
                "contract_version": raw_collection.get("contract_version"),
                "source_model_state_sha256": raw_collection.get(
                    "source_model_state_sha256"
                ),
                "source_artifact_digest": raw_collection.get("source_artifact_digest"),
                "config": raw_collection.get("config"),
                "bundle_exact_digests": [
                    _mapping(bundle, field="raw collection bundle").get("exact_digest")
                    for bundle in raw_bundles
                ],
                "aggregate_compute": raw_collection.get("aggregate_compute"),
            }
            collection_projection = {
                key: collection_payload.get(key) for key in raw_projection
            }
            if raw_projection != collection_projection:
                raise RecurrentScaleCampaignError(
                    "update journal detached from raw counterfactual collection bytes"
                )
            auxiliary = _mapping(
                record.get("counterfactual_auxiliary"),
                field=f"updates[{index}].counterfactual_auxiliary",
            )
            if collection_payload.get("source_model_state_sha256") != auxiliary.get(
                "pre_model_state_sha256"
            ):
                raise RecurrentScaleCampaignError(
                    "counterfactual collection and auxiliary model provenance drifted"
                )
        update_records.append(record)

    derived_worlds = sum(
        _positive_int(
            _mapping(record.get("rollout"), field="update.rollout").get("world_count"),
            field="update.rollout.world_count",
        )
        for record in update_records
    )
    derived_transitions = sum(
        _positive_int(
            _mapping(record.get("rollout"), field="update.rollout").get(
                "transition_count"
            ),
            field="update.rollout.transition_count",
        )
        for record in update_records
    )
    derived_delivery = _treatment_delivery_summary(update_records, arm=arm)
    if (
        training.get("worlds") != derived_worlds
        or training.get("agent_transitions") != derived_transitions
        or training.get("treatment_delivery") != derived_delivery
        or training.get("environment_seed_provenance") != expected_seed_provenance
        or any(
            artifact_data_metadata.get(key) != value
            for key, value in expected_seed_provenance.items()
        )
        or artifact_data_metadata.get("worlds") != derived_worlds
        or artifact_data_metadata.get("agent_transitions") != derived_transitions
    ):
        raise RecurrentScaleCampaignError(
            "completed report training totals detached from update journal bytes"
        )


def _verify_completed_evaluation_evidence(
    report: Mapping[str, object],
    *,
    run_directory: Path,
) -> dict[str, object]:
    modes = _mapping(
        _mapping(report.get("evaluations"), field="evaluations").get("modes"),
        field="evaluations.modes",
    )
    in_memory_evaluations: dict[str, Mapping[str, object]] = {}
    artifact_evaluations: dict[str, Mapping[str, object]] = {}
    for mode in _ACTION_SELECTION_MODES:
        evidence = _mapping(modes.get(mode), field=f"modes.{mode}")
        in_memory_path = run_directory / "evaluations" / f"in-memory-{mode}.json"
        artifact_evaluation_path = (
            run_directory / "evaluations" / f"artifact-cpu-{mode}.json"
        )
        _verify_file_reference(
            _mapping(
                evidence.get("in_memory_report"),
                field=f"modes.{mode}.in_memory_report",
            ),
            expected_path=in_memory_path,
        )
        _verify_file_reference(
            _mapping(
                evidence.get("artifact_cpu_report"),
                field=f"modes.{mode}.artifact_cpu_report",
            ),
            expected_path=artifact_evaluation_path,
        )
        in_memory_evaluation = load_strict_json(in_memory_path)
        artifact_evaluation = load_strict_json(artifact_evaluation_path)
        validate_recurrent_evaluation_report(in_memory_evaluation)
        validate_recurrent_evaluation_report(artifact_evaluation)
        artifact_claim = _mapping(
            artifact_evaluation.get("artifact"),
            field=f"artifact evaluation {mode}.artifact",
        )
        report_artifact = _mapping(report.get("artifact"), field="artifact")
        if (
            artifact_claim.get("artifact_sha256")
            != report_artifact.get("artifact_sha256")
            or Path(
                _text(
                    artifact_claim.get("path"),
                    field=f"artifact evaluation {mode}.artifact.path",
                )
            ).resolve()
            != (run_directory / "frozen-policy.json").resolve()
        ):
            raise RecurrentScaleCampaignError(
                "completed artifact evaluation references a different frozen policy"
            )
        in_memory_summary = recurrent_candidate_outcome_summary(in_memory_evaluation)
        artifact_summary = recurrent_candidate_outcome_summary(artifact_evaluation)
        reported_summary = dict(
            _mapping(
                evidence.get("candidate_outcome_summary"),
                field=f"modes.{mode}.candidate_outcome_summary",
            )
        )
        if (
            in_memory_summary != artifact_summary
            or artifact_summary != reported_summary
            or evidence.get("exact_cpu_outcome_replay_match") is not True
        ):
            raise RecurrentScaleCampaignError(
                "completed report evaluation summary detached from evaluation bytes"
            )
        in_memory_evaluations[mode] = in_memory_evaluation
        artifact_evaluations[mode] = artifact_evaluation
    # Artifact replay checks contain the artifact-specific replay digest and are
    # therefore validated above but are not expected to byte-match the
    # in-memory manifest that was frozen into the policy artifact.
    _full_world_replay_manifest(artifact_evaluations)
    return _full_world_replay_manifest(in_memory_evaluations)


def _verify_file_reference(
    reference: Mapping[str, object],
    *,
    expected_path: Path,
) -> None:
    if not expected_path.is_file():
        raise RecurrentScaleCampaignError(
            f"completed evidence file is missing: {expected_path}"
        )
    reference_path = Path(
        _text(reference.get("path"), field="completed evidence reference.path")
    )
    if (
        reference_path.resolve() != expected_path.resolve()
        or reference.get("sha256") != _file_sha256(expected_path)
        or reference.get("byte_length") != expected_path.stat().st_size
    ):
        raise RecurrentScaleCampaignError(
            f"completed evidence file drifted: {expected_path}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scale_arm_reports(
    preregistration: Mapping[str, object],
    *,
    output_root: str | Path,
    runtime_provenance_path: str | Path,
) -> tuple[dict[str, object], ...]:
    validate_recurrent_scale_campaign_preregistration(preregistration)
    reports: list[dict[str, object]] = []
    root = Path(output_root)
    runtime_path = Path(runtime_provenance_path)
    for learner_seed in SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"]:
        for arm in RECURRENT_SCALE_ARMS:
            run_id = recurrent_scale_arm_run_id(learner_seed=learner_seed, arm=arm)
            run_directory = root / run_id
            report = load_strict_json(run_directory / "report.json")
            validate_recurrent_scale_arm_report(
                report,
                preregistration=preregistration,
            )
            _verify_completed_report_evidence(
                report,
                run_directory=run_directory,
                runtime_provenance_path=runtime_path,
                preregistration=preregistration,
            )
            reports.append(report)
    return tuple(reports)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentScaleCampaignError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecurrentScaleCampaignError(f"{field} must be a sequence")
    return value


def _text_sequence(value: object, *, field: str) -> tuple[str, ...]:
    parsed = _sequence(value, field=field)
    result = tuple(_text(item, field=f"{field}[]") for item in parsed)
    if not result or len(result) != len(set(result)):
        raise RecurrentScaleCampaignError(f"{field} must be non-empty and unique")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentScaleCampaignError(f"{field} must be non-empty trimmed text")
    return value


def _sha256(value: object, *, field: str) -> str:
    parsed = _text(value, field=field)
    if len(parsed) != 64 or any(
        character not in "0123456789abcdef" for character in parsed
    ):
        raise RecurrentScaleCampaignError(f"{field} must be lowercase SHA-256")
    return parsed


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentScaleCampaignError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentScaleCampaignError(f"{field} must be a nonnegative integer")
    return value


def _float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentScaleCampaignError(f"{field} must be numeric")
    parsed = float(value)
    if not torch.isfinite(torch.tensor(parsed)).item():
        raise RecurrentScaleCampaignError(f"{field} must be finite")
    return parsed


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise RecurrentScaleCampaignError(f"{field} must be an exact boolean")
    return value


__all__ = [
    "RECURRENT_SCALE_FULL_WORLD_VERIFICATION_RUNNER",
    "RECURRENT_SCALE_UPDATE_RECORD_SCHEMA_VERSION",
    "analyze_recurrent_scale_campaign",
    "build_scale_run_components",
    "load_scale_arm_reports",
    "recurrent_candidate_outcome_summary",
    "run_recurrent_scale_arm",
    "validate_recurrent_scale_arm_report",
]
