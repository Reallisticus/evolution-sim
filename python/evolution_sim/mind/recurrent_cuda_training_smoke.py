from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_artifact import (
    load_recurrent_training_crash_checkpoint,
    save_recurrent_training_crash_checkpoint,
)
from evolution_sim.mind.recurrent_counterfactual_comparison import EXACT_ARM
from evolution_sim.mind.recurrent_experiment import (
    RecurrentExperimentRunner,
    RecurrentTrainingUpdateResult,
    configure_recurrent_training_determinism,
)
from evolution_sim.mind.recurrent_policy import recurrent_model_state_sha256
from evolution_sim.mind.recurrent_runtime_provenance import (
    assert_recurrent_scale_runtime_provenance_match,
    build_recurrent_scale_runtime_provenance,
    validate_recurrent_scale_runtime_provenance,
)
from evolution_sim.mind.recurrent_scale_campaign import (
    RecurrentScaleCampaignError,
    load_strict_json,
    source_file_hash_manifest,
    validate_recurrent_scale_campaign_preregistration,
    write_atomic_json,
)
from evolution_sim.mind.recurrent_scale_execution import build_scale_run_components
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_SEED_REGISTRY,
)


RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_cuda_training_smoke_v1"
)
RECURRENT_CUDA_TRAINING_SMOKE_CONTRACT_VERSION = (
    "mind_v3_public_recurrent_ippo_cuda_training_smoke_contract_v1"
)
RECURRENT_CUDA_TRAINING_SMOKE_RUN_ID = "cuda-training-smoke-preflight-v1"
RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT = 1
RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT = 1
RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT = 1
RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT = 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_recurrent_cuda_training_smoke_contract(
    preregistration: Mapping[str, object],
    *,
    learner_seed: int,
    rollout_workers: int,
    counterfactual_workers: int,
    evaluation_workers: int,
) -> dict[str, object]:
    """Describe the isolated mid-training resume gate without running training.

    Model, PPO, multi-tape target, terminal target, auxiliary loss, and KL gate
    are resolved from the exact campaign preregistration.  Only the number of
    updates, worlds, and branch bundles is reduced for this preflight.
    """

    validate_recurrent_scale_campaign_preregistration(preregistration)
    _positive_int(rollout_workers, field="rollout_workers")
    _positive_int(counterfactual_workers, field="counterfactual_workers")
    _positive_int(evaluation_workers, field="evaluation_workers")
    model, ppo, counterfactual, schedule = build_scale_run_components(
        preregistration,
        learner_seed=learner_seed,
        arm=EXACT_ARM,
        counterfactual_workers=counterfactual_workers,
    )
    if counterfactual is None:
        raise RecurrentScaleCampaignError(
            "exact scale arm unexpectedly omitted counterfactual learning"
        )
    warmup_tasks = tuple(schedule[0][:RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT])
    continuation_tasks = tuple(
        schedule[RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT][
            :RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT
        ]
    )
    if (
        len(warmup_tasks) != RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT
        or len(continuation_tasks) != RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT
    ):
        raise RecurrentScaleCampaignError("scale schedule cannot supply smoke worlds")
    contract: dict[str, object] = {
        "schema_version": RECURRENT_CUDA_TRAINING_SMOKE_CONTRACT_VERSION,
        "purpose": "pre_tmux_cuda_training_and_crash_resume_launch_gate",
        "preregistration_digest": preregistration["exact_digest"],
        "source": dict(_mapping(preregistration.get("source"), field="source")),
        "learner_seed": learner_seed,
        "arm": EXACT_ARM,
        "algorithm": {
            "model": asdict(model),
            "ppo": asdict(ppo),
            "counterfactual_collection": asdict(counterfactual.collection),
            "counterfactual_auxiliary": counterfactual.auxiliary.as_contract(),
            "counterfactual_step": counterfactual.step.as_contract(),
        },
        "bounded_workload": {
            "warmup_updates_before_checkpoint": (
                RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT
            ),
            "parity_compared_continuation_updates": (
                RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT
            ),
            "updates_on_uninterrupted_path": (
                RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT
                + RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT
            ),
            "worlds_per_update": RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT,
            "counterfactual_bundles_per_update": (
                RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT
            ),
            "rollout_ticks": warmup_tasks[0].rollout_ticks,
            "full_campaign_rollout_ticks_preserved": True,
            "full_multi_tape_count_preserved": True,
            "full_terminal_target_preserved": True,
            "warmup_task": asdict(warmup_tasks[0]),
            "continued_task": asdict(continuation_tasks[0]),
        },
        "workers": {
            "rollout_workers": rollout_workers,
            "counterfactual_workers": counterfactual_workers,
            "evaluation_workers": evaluation_workers,
        },
        "isolation": {
            "campaign_run_id_used": False,
            "campaign_cell_output_used": False,
            "campaign_slice_consumed": False,
            "validation_seeds_accessed": False,
            "lockbox_seeds_accessed": False,
            "runtime_artifact_created": False,
            "promotion_authorized": False,
        },
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
            "CUDA training smoke contract must normalize to an object"
        )
    normalized["exact_digest"] = stable_payload_digest(normalized)
    return normalized


def run_recurrent_cuda_training_smoke(
    preregistration: Mapping[str, object],
    *,
    expected_preregistration_digest: str,
    runtime_provenance_path: str | Path,
    output_root: str | Path,
    device: torch.device | str,
    rollout_workers: int,
    counterfactual_workers: int,
    evaluation_workers: int,
) -> dict[str, object]:
    """Compare one real exact-arm continuation across a mid-training checkpoint.

    A real warmup update first populates Adam moments, update counters, and the
    one-use auxiliary ledger.  The uninterrupted path then runs update 1.  A
    pristine runner loads the JSON checkpoint at completed_updates=1, restores
    model/Adam/RNG/ledger state, and runs that same update 1.  Exact final model
    and update-evidence digests must match before the long campaign may launch.
    """

    validate_recurrent_scale_campaign_preregistration(preregistration)
    preregistration_digest = _sha256(
        preregistration.get("exact_digest"), field="preregistration.exact_digest"
    )
    if preregistration_digest != _sha256(
        expected_preregistration_digest,
        field="expected_preregistration_digest",
    ):
        raise RecurrentScaleCampaignError("CUDA smoke preregistration pin mismatched")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RecurrentScaleCampaignError(
            "CUDA training smoke requires an available CUDA device"
        )
    try:
        torch.empty(1, device=resolved_device)
        torch.cuda.synchronize(resolved_device)
    except RuntimeError as error:
        raise RecurrentScaleCampaignError(
            "CUDA training smoke cannot access the requested device"
        ) from error

    source = _mapping(preregistration.get("source"), field="source")
    source_commit = _text(source.get("commit"), field="source.commit")
    source_manifest_sha256 = _sha256(
        source.get("manifest_sha256"), field="source.manifest_sha256"
    )
    _require_exact_clean_source(
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
    )
    runtime_path = Path(runtime_provenance_path)
    runtime_provenance = load_strict_json(runtime_path)
    validate_recurrent_scale_runtime_provenance(runtime_provenance)
    workers = {
        "rollout_workers": _positive_int(rollout_workers, field="rollout_workers"),
        "counterfactual_workers": _positive_int(
            counterfactual_workers, field="counterfactual_workers"
        ),
        "evaluation_workers": _positive_int(
            evaluation_workers, field="evaluation_workers"
        ),
    }
    learner_seed = int(SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0])
    configure_recurrent_training_determinism(
        learner_seed=learner_seed,
        device=resolved_device,
    )
    observed_runtime = build_recurrent_scale_runtime_provenance(
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
        preregistration_digest=preregistration_digest,
        repository_clean=True,
        device=resolved_device,
        **workers,
    )
    assert_recurrent_scale_runtime_provenance_match(
        runtime_provenance,
        observed_runtime,
    )
    runtime_digest = _sha256(
        runtime_provenance.get("exact_digest"),
        field="runtime_provenance.exact_digest",
    )
    contract = build_recurrent_cuda_training_smoke_contract(
        preregistration,
        learner_seed=learner_seed,
        **workers,
    )
    output_directory = Path(output_root)
    report_path = output_directory / "report.json"
    checkpoint_path = output_directory / "mid-training-crash-checkpoint.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    if report_path.is_file():
        existing = load_strict_json(report_path)
        validate_recurrent_cuda_training_smoke_report(
            existing,
            preregistration=preregistration,
            runtime_provenance=runtime_provenance,
        )
        existing_runtime = _mapping(
            existing.get("runtime_provenance"), field="runtime_provenance"
        )
        checkpoint = _mapping(existing.get("checkpoint"), field="checkpoint")
        if (
            existing_runtime.get("path") != str(runtime_path)
            or existing_runtime.get("file_sha256") != _file_sha256(runtime_path)
            or checkpoint.get("path") != str(checkpoint_path)
            or _file_sha256(checkpoint_path) != checkpoint.get("file_sha256")
        ):
            raise RecurrentScaleCampaignError(
                "completed CUDA smoke runtime or checkpoint evidence drifted"
            )
        loaded_existing = load_recurrent_training_crash_checkpoint(checkpoint_path)
        existing_configuration = _mapping(
            loaded_existing.checkpoint.get("configuration"),
            field="loaded checkpoint configuration",
        )
        if existing_configuration.get("training_config") != contract:
            raise RecurrentScaleCampaignError(
                "completed CUDA smoke checkpoint contract drifted"
            )
        existing_progress = _mapping(
            loaded_existing.checkpoint.get("progress"),
            field="loaded checkpoint progress",
        )
        existing_rng = _mapping(
            loaded_existing.rng_state,
            field="loaded checkpoint rng state",
        )
        existing_parity = _mapping(existing.get("resume_parity"), field="resume_parity")
        existing_attempted = existing_rng.get(
            "attempted_counterfactual_auxiliary_bundles"
        )
        if (
            existing_progress.get("completed_updates") != 1
            or recurrent_model_state_sha256(loaded_existing.model)
            != existing_parity.get("checkpoint_model_sha256")
            or existing_rng.get("trainer_update_index") != 1
            or existing_rng.get("counterfactual_auxiliary_update_count") != 1
            or not isinstance(existing_attempted, list)
            or len(existing_attempted) != 1
            or not _optimizer_state_populated(loaded_existing.optimizer_state)
        ):
            raise RecurrentScaleCampaignError(
                "completed CUDA smoke mid-training checkpoint evidence drifted"
            )
        return existing

    model_config, ppo_config, counterfactual_config, schedule = (
        build_scale_run_components(
            preregistration,
            learner_seed=learner_seed,
            arm=EXACT_ARM,
            counterfactual_workers=counterfactual_workers,
        )
    )
    if counterfactual_config is None:
        raise RecurrentScaleCampaignError(
            "CUDA smoke requires the exact counterfactual arm"
        )
    smoke_counterfactual = replace(
        counterfactual_config,
        bundles_per_update=RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT,
    )
    warmup_tasks = tuple(schedule[0][:RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT])
    continued_tasks = tuple(
        schedule[RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT][
            :RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT
        ]
    )
    uninterrupted = RecurrentExperimentRunner(
        learner_seed=learner_seed,
        device=resolved_device,
        model_config=model_config,
        ppo_config=ppo_config,
        rollout_workers=rollout_workers,
        counterfactual_config=smoke_counterfactual,
    )
    initial_model_digest = recurrent_model_state_sha256(uninterrupted.model)
    warmup_update = uninterrupted.train_update(warmup_tasks)
    checkpoint_model_digest = recurrent_model_state_sha256(uninterrupted.model)
    _validate_real_training_update(
        warmup_update,
        expected_update_index=0,
        pre_update_model_digest=initial_model_digest,
        final_model_digest=checkpoint_model_digest,
        preregistration=preregistration,
    )
    checkpoint_state = uninterrupted.export_training_checkpoint_state()
    checkpoint_rng_state = dict(
        _mapping(checkpoint_state.get("rng_state"), field="checkpoint.rng_state")
    )
    checkpoint_rng_state["cuda_training_smoke_runtime_provenance_digest"] = (
        runtime_digest
    )
    attempted_bundles = checkpoint_rng_state.get(
        "attempted_counterfactual_auxiliary_bundles"
    )
    if (
        checkpoint_rng_state.get("trainer_update_index") != 1
        or checkpoint_rng_state.get("counterfactual_auxiliary_update_count") != 1
        or not isinstance(attempted_bundles, list)
        or len(attempted_bundles) != 1
    ):
        raise RecurrentScaleCampaignError(
            "CUDA smoke warmup did not populate update counters and auxiliary ledger"
        )
    if not _optimizer_state_populated(checkpoint_state.get("optimizer_state")):
        raise RecurrentScaleCampaignError(
            "CUDA smoke warmup did not populate Adam optimizer moments"
        )
    checkpoint = save_recurrent_training_crash_checkpoint(
        checkpoint_path,
        uninterrupted.model,
        optimizer_state=checkpoint_state["optimizer_state"],
        rng_state=checkpoint_rng_state,
        optimizer_type="torch.optim.Adam",
        training_config=contract,
        seed_registry_digest=SCALE_DEVELOPMENT_CANONICAL_SHA256,
        source_commit=source_commit,
        source_manifest_sha256=source_manifest_sha256,
        learner_seed=learner_seed,
        completed_updates=RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT,
        run_id=RECURRENT_CUDA_TRAINING_SMOKE_RUN_ID,
    )
    loaded = load_recurrent_training_crash_checkpoint(checkpoint_path)
    loaded_configuration = _mapping(
        loaded.checkpoint.get("configuration"),
        field="loaded checkpoint configuration",
    )
    if loaded_configuration.get("training_config") != contract:
        raise RecurrentScaleCampaignError(
            "CUDA smoke checkpoint training contract changed on write/load"
        )
    if recurrent_model_state_sha256(loaded.model) != checkpoint_model_digest:
        raise RecurrentScaleCampaignError(
            "CUDA smoke crash checkpoint changed warmup model parameters"
        )
    loaded_rng = _mapping(loaded.rng_state, field="loaded checkpoint rng_state")
    if (
        loaded_rng.get("cuda_training_smoke_runtime_provenance_digest")
        != runtime_digest
        or loaded_rng.get("trainer_update_index") != 1
        or loaded_rng.get("counterfactual_auxiliary_update_count") != 1
        or loaded_rng.get("attempted_counterfactual_auxiliary_bundles")
        != attempted_bundles
        or not _optimizer_state_populated(loaded.optimizer_state)
    ):
        raise RecurrentScaleCampaignError(
            "CUDA smoke checkpoint optimizer, counters, ledger, or runtime binding drifted"
        )

    uninterrupted_update = uninterrupted.train_update(continued_tasks)
    uninterrupted_final_digest = recurrent_model_state_sha256(uninterrupted.model)
    uninterrupted_evidence = _update_evidence(uninterrupted_update)
    _validate_real_training_update(
        uninterrupted_update,
        expected_update_index=1,
        pre_update_model_digest=checkpoint_model_digest,
        final_model_digest=uninterrupted_final_digest,
        preregistration=preregistration,
    )

    resumed = RecurrentExperimentRunner(
        learner_seed=learner_seed,
        device=resolved_device,
        model_config=model_config,
        ppo_config=ppo_config,
        rollout_workers=rollout_workers,
        counterfactual_config=smoke_counterfactual,
    )
    resumed.restore_training_checkpoint_state(
        model_state=loaded.model.state_dict(),
        optimizer_state=loaded.optimizer_state,
        rng_state=loaded.rng_state,
        completed_updates=RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT,
    )
    if (
        recurrent_model_state_sha256(resumed.model) != checkpoint_model_digest
        or resumed.completed_update_count != 1
        or resumed.trainer.update_index != 1
        or resumed.trainer.counterfactual_auxiliary_update_count != 1
        or resumed.trainer._attempted_counterfactual_auxiliary_bundles
        != set(attempted_bundles)
    ):
        raise RecurrentScaleCampaignError(
            "CUDA smoke restored runner differs from mid-training checkpoint"
        )
    resumed_update = resumed.train_update(continued_tasks)
    resumed_final_digest = recurrent_model_state_sha256(resumed.model)
    resumed_evidence = _update_evidence(resumed_update)
    _validate_real_training_update(
        resumed_update,
        expected_update_index=1,
        pre_update_model_digest=checkpoint_model_digest,
        final_model_digest=resumed_final_digest,
        preregistration=preregistration,
    )
    if uninterrupted_final_digest != resumed_final_digest:
        raise RecurrentScaleCampaignError(
            "CUDA resumed training model digest differs from uninterrupted training"
        )
    if uninterrupted_evidence != resumed_evidence:
        raise RecurrentScaleCampaignError(
            "CUDA resumed update evidence differs from uninterrupted training"
        )

    auxiliary = resumed_update.counterfactual_auxiliary
    collection = resumed_update.counterfactual_collection
    assert auxiliary is not None
    assert collection is not None
    report: dict[str, object] = {
        "schema_version": RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION,
        "kind": "isolated_cuda_training_preflight",
        "preregistration_digest": preregistration_digest,
        "source": {
            "commit": source_commit,
            "manifest_sha256": source_manifest_sha256,
            "clean_tree_verified": True,
        },
        "runtime_provenance": {
            "exact_digest": runtime_digest,
            "path": str(runtime_path),
            "file_sha256": _file_sha256(runtime_path),
        },
        "contract": contract,
        "device": str(resolved_device),
        "checkpoint": {
            "schema_version": checkpoint["schema_version"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "path": str(checkpoint_path),
            "file_sha256": _file_sha256(checkpoint_path),
            "completed_updates": RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT,
            "model_loaded_exactly": True,
            "optimizer_moments_nonempty_and_loaded": True,
            "rng_loaded": True,
            "trainer_update_index_restored": 1,
            "auxiliary_update_count_restored": 1,
            "attempted_auxiliary_bundle_ledger_count": len(attempted_bundles),
            "attempted_auxiliary_bundle_ledger_restored": True,
            "runtime_provenance_bound": True,
        },
        "training": {
            "real_simulator_interactions": True,
            "warmup_updates_before_checkpoint": (
                RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT
            ),
            "parity_compared_continuation_updates": (
                RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT
            ),
            "worlds_per_update": RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT,
            "warmup_rollout_transitions": warmup_update.rollout.transition_count,
            "continued_rollout_transitions": (resumed_update.rollout.transition_count),
            "continued_ppo_parameter_delta_l2": (
                resumed_update.optimizer.parameter_delta_l2
            ),
            "counterfactual": {
                "aggregate_multi_tape": True,
                "bundle_count": len(collection.bundles),
                "continuation_tape_count": collection.config.continuation_tape_count,
                "terminal_target_world_tick": (
                    collection.config.terminal_target_world_tick
                ),
                "auxiliary_step_accepted": auxiliary.accepted,
                "optimizer_step_count": auxiliary.optimizer_step_count,
                "backward_pass_count": auxiliary.backward_pass_count,
                "parameter_delta_l2": auxiliary.parameter_delta_l2,
                "mean_behavior_kl_old_to_post": (
                    auxiliary.mean_behavior_kl_old_to_post
                ),
                "max_state_behavior_kl_old_to_post": (
                    auxiliary.max_state_behavior_kl_old_to_post
                ),
            },
        },
        "resume_parity": {
            "initial_model_sha256": initial_model_digest,
            "checkpoint_model_sha256": checkpoint_model_digest,
            "uninterrupted_final_model_sha256": uninterrupted_final_digest,
            "resumed_final_model_sha256": resumed_final_digest,
            "model_digest_exact_match": True,
            "uninterrupted_update_evidence_sha256": stable_payload_digest(
                uninterrupted_evidence
            ),
            "resumed_update_evidence_sha256": stable_payload_digest(resumed_evidence),
            "update_evidence_exact_match": True,
        },
        "lifecycle": {
            "preflight_only": True,
            "campaign_cell_output_used": False,
            "campaign_slice_consumed": False,
            "validation_seeds_accessed": False,
            "lockbox_seeds_accessed": False,
            "runtime_artifact_created": False,
            "runtime_action_selection_changed": False,
            "promotion_authorized": False,
            "gate_relaxation_authorized": False,
        },
    }
    report["exact_digest"] = stable_payload_digest(report)
    validate_recurrent_cuda_training_smoke_report(
        report,
        preregistration=preregistration,
        runtime_provenance=runtime_provenance,
    )
    write_atomic_json(report_path, report)
    return report


def validate_recurrent_cuda_training_smoke_report(
    report: Mapping[str, object],
    *,
    preregistration: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
) -> None:
    validate_recurrent_scale_campaign_preregistration(preregistration)
    validate_recurrent_scale_runtime_provenance(runtime_provenance)
    if report.get("schema_version") != RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION:
        raise RecurrentScaleCampaignError("CUDA training smoke schema drifted")
    exact_digest = _sha256(report.get("exact_digest"), field="report.exact_digest")
    unsigned = dict(report)
    unsigned.pop("exact_digest", None)
    if stable_payload_digest(unsigned) != exact_digest:
        raise RecurrentScaleCampaignError("CUDA training smoke digest mismatched")
    if report.get("kind") != "isolated_cuda_training_preflight":
        raise RecurrentScaleCampaignError("CUDA training smoke kind drifted")
    if report.get("preregistration_digest") != preregistration.get("exact_digest"):
        raise RecurrentScaleCampaignError("CUDA training smoke preregistration drifted")
    runtime = _mapping(report.get("runtime_provenance"), field="runtime_provenance")
    runtime_device = _mapping(
        runtime_provenance.get("device"), field="runtime_provenance.device"
    )
    if (
        runtime.get("exact_digest") != runtime_provenance.get("exact_digest")
        or runtime_device.get("type") != "cuda"
        or not isinstance(runtime.get("path"), str)
        or not runtime.get("path")
        or not isinstance(runtime.get("file_sha256"), str)
    ):
        raise RecurrentScaleCampaignError("CUDA training smoke runtime drifted")
    _sha256(runtime.get("file_sha256"), field="runtime_provenance.file_sha256")
    contract = _mapping(report.get("contract"), field="contract")
    expected_contract = build_recurrent_cuda_training_smoke_contract(
        preregistration,
        learner_seed=int(SCALE_DEVELOPMENT_SEED_REGISTRY["scale_learner"][0]),
        **dict(_mapping(contract.get("workers"), field="contract.workers")),
    )
    if dict(contract) != expected_contract:
        raise RecurrentScaleCampaignError("CUDA training smoke contract drifted")
    checkpoint = _mapping(report.get("checkpoint"), field="checkpoint")
    if (
        checkpoint.get("completed_updates")
        != RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT
        or checkpoint.get("model_loaded_exactly") is not True
        or checkpoint.get("optimizer_moments_nonempty_and_loaded") is not True
        or checkpoint.get("rng_loaded") is not True
        or checkpoint.get("trainer_update_index_restored") != 1
        or checkpoint.get("auxiliary_update_count_restored") != 1
        or checkpoint.get("attempted_auxiliary_bundle_ledger_count") != 1
        or checkpoint.get("attempted_auxiliary_bundle_ledger_restored") is not True
        or checkpoint.get("runtime_provenance_bound") is not True
        or not isinstance(checkpoint.get("path"), str)
        or not checkpoint.get("path")
    ):
        raise RecurrentScaleCampaignError("CUDA training smoke restore evidence failed")
    _sha256(checkpoint.get("checkpoint_sha256"), field="checkpoint.checkpoint_sha256")
    _sha256(checkpoint.get("file_sha256"), field="checkpoint.file_sha256")
    training = _mapping(report.get("training"), field="training")
    counterfactual = _mapping(
        training.get("counterfactual"), field="training.counterfactual"
    )
    expected_tape_count = _mapping(
        preregistration.get("counterfactual"), field="counterfactual"
    ).get("independent_rng_tapes_per_branch")
    expected_terminal_tick = _mapping(
        preregistration.get("counterfactual"), field="counterfactual"
    ).get("absolute_terminal_target_tick")
    if (
        training.get("real_simulator_interactions") is not True
        or training.get("warmup_updates_before_checkpoint") != 1
        or training.get("parity_compared_continuation_updates") != 1
        or training.get("worlds_per_update") != 1
        or not isinstance(training.get("warmup_rollout_transitions"), int)
        or int(training["warmup_rollout_transitions"]) <= 0
        or not isinstance(training.get("continued_rollout_transitions"), int)
        or int(training["continued_rollout_transitions"]) <= 0
        or not isinstance(
            training.get("continued_ppo_parameter_delta_l2"), (float, int)
        )
        or float(training["continued_ppo_parameter_delta_l2"]) <= 0.0
        or counterfactual.get("aggregate_multi_tape") is not True
        or counterfactual.get("bundle_count") != 1
        or counterfactual.get("continuation_tape_count") != expected_tape_count
        or counterfactual.get("terminal_target_world_tick") != expected_terminal_tick
        or counterfactual.get("auxiliary_step_accepted") is not True
        or counterfactual.get("optimizer_step_count") != 1
        or counterfactual.get("backward_pass_count") != 1
        or not isinstance(counterfactual.get("parameter_delta_l2"), (float, int))
        or float(counterfactual["parameter_delta_l2"]) <= 0.0
    ):
        raise RecurrentScaleCampaignError(
            "CUDA training smoke did not exercise real PPO and exact auxiliary learning"
        )
    parity = _mapping(report.get("resume_parity"), field="resume_parity")
    if (
        parity.get("model_digest_exact_match") is not True
        or parity.get("update_evidence_exact_match") is not True
        or parity.get("uninterrupted_final_model_sha256")
        != parity.get("resumed_final_model_sha256")
        or parity.get("uninterrupted_update_evidence_sha256")
        != parity.get("resumed_update_evidence_sha256")
    ):
        raise RecurrentScaleCampaignError("CUDA training smoke resume parity failed")
    for field in (
        "initial_model_sha256",
        "checkpoint_model_sha256",
        "uninterrupted_final_model_sha256",
        "resumed_final_model_sha256",
        "uninterrupted_update_evidence_sha256",
        "resumed_update_evidence_sha256",
    ):
        _sha256(parity.get(field), field=f"resume_parity.{field}")
    if parity.get("initial_model_sha256") == parity.get(
        "checkpoint_model_sha256"
    ) or parity.get("checkpoint_model_sha256") == parity.get(
        "uninterrupted_final_model_sha256"
    ):
        raise RecurrentScaleCampaignError(
            "CUDA training smoke warmup or continued update did not learn"
        )
    lifecycle = _mapping(report.get("lifecycle"), field="lifecycle")
    if lifecycle != {
        "preflight_only": True,
        "campaign_cell_output_used": False,
        "campaign_slice_consumed": False,
        "validation_seeds_accessed": False,
        "lockbox_seeds_accessed": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "promotion_authorized": False,
        "gate_relaxation_authorized": False,
    }:
        raise RecurrentScaleCampaignError("CUDA training smoke lifecycle drifted")


def _validate_real_training_update(
    update: RecurrentTrainingUpdateResult,
    *,
    expected_update_index: int,
    pre_update_model_digest: str,
    final_model_digest: str,
    preregistration: Mapping[str, object],
) -> None:
    collection = update.counterfactual_collection
    auxiliary = update.counterfactual_auxiliary
    if (
        update.update_index != expected_update_index
        or update.rollout.world_count != RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT
        or update.rollout.transition_count <= 0
        or update.optimizer.parameter_delta_l2 <= 0.0
        or collection is None
        or auxiliary is None
    ):
        raise RecurrentScaleCampaignError(
            "CUDA smoke did not complete one real PPO plus counterfactual update"
        )
    counterfactual = _mapping(
        preregistration.get("counterfactual"), field="counterfactual"
    )
    if (
        len(collection.bundles) != RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT
        or collection.config.continuation_tape_count
        != counterfactual.get("independent_rng_tapes_per_branch")
        or collection.config.terminal_target_world_tick
        != counterfactual.get("absolute_terminal_target_tick")
        or any(bundle.aggregate_rows is None for bundle in collection.bundles)
        or auxiliary.optimizer_step_count != 1
        or auxiliary.backward_pass_count != 1
        or not auxiliary.accepted
        or auxiliary.parameter_delta_l2 <= 0.0
        or auxiliary.final_model_state_sha256 != final_model_digest
        or final_model_digest == pre_update_model_digest
    ):
        raise RecurrentScaleCampaignError(
            "CUDA smoke exact aggregate auxiliary learning gate failed"
        )


def _update_evidence(update: RecurrentTrainingUpdateResult) -> dict[str, object]:
    collection = update.counterfactual_collection
    auxiliary = update.counterfactual_auxiliary
    if collection is None or auxiliary is None:
        raise RecurrentScaleCampaignError("CUDA smoke omitted counterfactual evidence")
    return {
        "update_index": update.update_index,
        "tasks": [asdict(task) for task in update.tasks],
        "rollout": asdict(update.rollout),
        "ppo": asdict(update.optimizer),
        "counterfactual_collection_exact_digest": collection.exact_digest,
        "counterfactual_collection_compute": dict(collection.aggregate_compute),
        "counterfactual_bundle_digests": [
            bundle.exact_digest for bundle in collection.bundles
        ],
        "counterfactual_auxiliary": asdict(auxiliary),
    }


def _require_exact_clean_source(
    *,
    source_commit: str,
    source_manifest_sha256: str,
) -> None:
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
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RecurrentScaleCampaignError(
            "CUDA training smoke could not verify Git source state"
        ) from error
    manifest = source_file_hash_manifest(_REPOSITORY_ROOT)
    if (
        head != source_commit
        or status
        or manifest.get("aggregate_sha256") != source_manifest_sha256
    ):
        raise RecurrentScaleCampaignError(
            "CUDA training smoke requires the preregistered exact clean source"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer_state_populated(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    state = value.get("state")
    parameter_groups = value.get("param_groups")
    if (
        not isinstance(state, Mapping)
        or not state
        or not isinstance(parameter_groups, list)
        or not parameter_groups
    ):
        return False
    return any(
        isinstance(record, Mapping)
        and {"step", "exp_avg", "exp_avg_sq"}.issubset(record)
        for record in state.values()
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentScaleCampaignError(f"{field} must be an object")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentScaleCampaignError(f"{field} must be non-empty text")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentScaleCampaignError(f"{field} must be a positive integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecurrentScaleCampaignError(f"{field} must be a lowercase SHA256")
    return value


__all__ = [
    "RECURRENT_CUDA_TRAINING_SMOKE_BUNDLE_COUNT",
    "RECURRENT_CUDA_TRAINING_SMOKE_CONTRACT_VERSION",
    "RECURRENT_CUDA_TRAINING_SMOKE_SCHEMA_VERSION",
    "RECURRENT_CUDA_TRAINING_SMOKE_UPDATE_COUNT",
    "RECURRENT_CUDA_TRAINING_SMOKE_WARMUP_UPDATE_COUNT",
    "RECURRENT_CUDA_TRAINING_SMOKE_WORLD_COUNT",
    "build_recurrent_cuda_training_smoke_contract",
    "run_recurrent_cuda_training_smoke",
    "validate_recurrent_cuda_training_smoke_report",
]
