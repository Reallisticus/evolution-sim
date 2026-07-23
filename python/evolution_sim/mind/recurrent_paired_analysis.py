from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import TypeAlias

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_evaluation_contract import (
    RECURRENT_EVALUATION_SCHEMA_VERSION,
    RecurrentEvaluationError,
)


PAIRED_ANALYSIS_SCHEMA_VERSION = (
    "mind_public_recurrent_gru_feed_forward_paired_analysis_v1"
)
CAUSAL_CANARY_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_development_causal_canary_v3"
)
CAUSAL_CANARY_POLICY_VERSION = "public_recurrent_ippo_in_memory_causal_canary_v3"
CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION = (
    "mind_v3_recurrent_causal_canary_preregistration_v2"
)
CAUSAL_CANARY_SCHEMA_VERSION_V4 = (
    "mind_v3_public_recurrent_ippo_development_causal_canary_v4"
)
CAUSAL_CANARY_POLICY_VERSION_V4 = "public_recurrent_ippo_in_memory_causal_canary_v4"
CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION_V4 = (
    "mind_v3_recurrent_causal_canary_preregistration_v4"
)
_CAUSAL_REPORT_CONTRACTS = {
    CAUSAL_CANARY_SCHEMA_VERSION: (
        CAUSAL_CANARY_POLICY_VERSION,
        CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION,
    ),
    CAUSAL_CANARY_SCHEMA_VERSION_V4: (
        CAUSAL_CANARY_POLICY_VERSION_V4,
        CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION_V4,
    ),
}
ARGMAX_SELECTION = "deterministic_masked_argmax"
SAMPLED_SELECTION = "replay_deterministic_masked_sampling"
SELECTION_MODES = (ARGMAX_SELECTION, SAMPLED_SELECTION)
UNPINNED_LABEL_PREFIX = "unpinned-noncandidate-development-canary:"
TRAINING_POLICY_SAMPLING_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-ippo|policy-action-sampling-v1|2026-07-21"
)
EVALUATION_POLICY_SAMPLING_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-evaluation|artifact-sampling-v1"
)
_MIN_TRAINING_POLICY_SAMPLING_SEED = 2**31
_MAX_TRAINING_POLICY_SAMPLING_SEED = 2**63 - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40,64}$")
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
_TOP_LEVEL_CLOSED_FLAGS = (
    "campaign_training_slice_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "validation_seeds_accessed",
    "lockbox_seeds_accessed",
)
_BASE_METRICS = (
    "terminal_survival",
    "terminal_alive",
    "births",
    "deaths",
    "reward_total",
    "trajectory_record_count",
    "policy_decision_record_count",
    "passive_trajectory_record_count",
    "dominant_requested_action_share",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
    "eat_requested_count",
    "eat_without_positive_resource_gain_count",
    "eat_without_positive_resource_gain_share",
)
_REQUIRED_RUN_COUNT_FIELDS = (
    "terminal_alive",
    "births",
    "deaths",
    "trajectory_record_count",
    "policy_decision_record_count",
    "passive_trajectory_record_count",
    "dominant_requested_action_count",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
    "eat_requested_count",
    "eat_without_positive_resource_gain_count",
)


ReportSource: TypeAlias = Mapping[str, object] | str | Path
RunIdentity: TypeAlias = tuple[str, int, int | None]


class RecurrentPairedAnalysisError(ValueError):
    """Raised when matched recurrent-ablation evidence fails closed."""


def analyze_recurrent_causal_canary_pair(
    gru_report: ReportSource,
    feed_forward_report: ReportSource,
) -> dict[str, object]:
    """Validate and descriptively compare one matched GRU/true-FF learner pair.

    The learner seed, initialization, source, training schedule, and evaluation
    streams are fixed across arms. Policy-sampling streams remain repeated
    measurements within one learner run; this function deliberately emits no
    p-values, learner-level confidence interval, or replication claim.
    """

    gru = _load_and_validate_report(gru_report, label="gru")
    feed_forward = _load_and_validate_report(
        feed_forward_report,
        label="feed_forward",
    )
    pair = _validate_pair_contract(gru, feed_forward)

    selection_modes: dict[str, object] = {}
    arm_effects_by_mode: dict[
        str,
        tuple[
            dict[RunIdentity, dict[str, object]], dict[RunIdentity, dict[str, object]]
        ],
    ] = {}
    for mode in SELECTION_MODES:
        gru_effects = _arm_training_effects(gru, mode=mode, label="gru")
        ff_effects = _arm_training_effects(
            feed_forward,
            mode=mode,
            label="feed_forward",
        )
        arm_effects_by_mode[mode] = (gru_effects, ff_effects)
        selection_modes[mode] = _difference_in_differences(
            gru_effects,
            ff_effects,
            mode=mode,
        )

    learner_seed = pair["learner_seed"]
    report: dict[str, object] = {
        "schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": (
            "matched_descriptive_gru_minus_true_feed_forward_difference_in_differences"
        ),
        "evidence": {
            "gru_report_exact_digest": gru["exact_digest"],
            "feed_forward_report_exact_digest": feed_forward["exact_digest"],
            "git_head_observed": pair["git_head_observed"],
            "source_manifest_aggregate_sha256": pair[
                "source_manifest_aggregate_sha256"
            ],
            "training_schedule_sha256": pair["training_schedule_sha256"],
            "before_model_state_sha256": pair["before_model_state_sha256"],
            "evaluation_sampling_stream_id": pair["evaluation_sampling_stream_id"],
            "all_report_digests_verified": True,
            "source_and_initialization_exactly_matched": True,
            "training_and_evaluation_streams_exactly_matched": True,
            "replay_and_action_integrity_verified": True,
            "validation_and_lockbox_remained_closed": True,
        },
        "pairing_contract": {
            "gru_feed_forward_history_ablation": False,
            "feed_forward_feed_forward_history_ablation": True,
            "only_permitted_experimental_configuration_difference": (
                "feed_forward_history_ablation"
            ),
            "learner_seed": learner_seed,
            "learner_replication_count_per_arm": 1,
            "policy_sampling_streams_are_repeated_measurements": True,
            "policy_sampling_streams_are_learner_replications": False,
            "environment_seeds_are_learner_replications": False,
            "inferential_claim_authorized": False,
            "interpretation": (
                "descriptive effect for one matched learner initialization; "
                "confirmation requires fresh learner-seed replications"
            ),
        },
        "difference_in_differences": {
            "estimand": ("(GRU_after - GRU_before) - (true_FF_after - true_FF_before)"),
            "selection_modes": selection_modes,
            "learner_level_carrion_effect": _learner_level_carrion_effect(
                learner_seed=int(learner_seed),
                arm_effects_by_mode=arm_effects_by_mode,
            ),
        },
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def _load_and_validate_report(
    source: ReportSource,
    *,
    label: str,
) -> dict[str, object]:
    if isinstance(source, Mapping):
        report = _canonical_clone(source, field=f"{label} report")
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() != ".json":
            raise RecurrentPairedAnalysisError(
                f"{label} report path must have a .json suffix"
            )
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RecurrentPairedAnalysisError(
                f"cannot read {label} report: {path}"
            ) from exc
        try:
            loaded = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, RecurrentPairedAnalysisError) as exc:
            raise RecurrentPairedAnalysisError(
                f"{label} report is not canonical finite JSON"
            ) from exc
        if not isinstance(loaded, dict):
            raise RecurrentPairedAnalysisError(f"{label} report root must be a mapping")
        report = loaded
    else:
        raise TypeError(f"{label} report must be a mapping or JSON path")

    exact_digest = _sha256(report.get("exact_digest"), field=f"{label}.exact_digest")
    digest_payload = copy.deepcopy(report)
    digest_payload.pop("exact_digest")
    if stable_payload_digest(digest_payload) != exact_digest:
        raise RecurrentPairedAnalysisError(
            f"{label} report exact digest does not match its canonical payload"
        )
    _validate_report(report, label=label)
    return report


def _validate_report(report: Mapping[str, object], *, label: str) -> None:
    schema_version = report.get("schema_version")
    expected_contract = (
        _CAUSAL_REPORT_CONTRACTS.get(schema_version)
        if isinstance(schema_version, str)
        else None
    )
    if expected_contract is None:
        raise RecurrentPairedAnalysisError(f"{label} causal canary schema drifted")
    expected_policy, _expected_preregistration = expected_contract
    if report.get("policy") != expected_policy:
        raise RecurrentPairedAnalysisError(f"{label} causal canary policy drifted")
    for field in (
        "development_run",
        "development_experiment",
        "noncandidate_development_canary",
        "source_unpinned",
        "artifact_output_refused_by_contract",
        "non_promoted",
    ):
        if report.get(field) is not True:
            raise RecurrentPairedAnalysisError(f"{label}.{field} must be true")
    for field in ("source_pinned", "artifact_output_requested"):
        if report.get(field) is not False:
            raise RecurrentPairedAnalysisError(f"{label}.{field} must be false")
    if report.get("artifact_path") is not None:
        raise RecurrentPairedAnalysisError(
            f"{label} development report cannot name an artifact"
        )
    _validate_closed_lifecycle(report, label=label)
    _validate_source(report, label=label)
    _validate_preregistration(report, label=label)
    _validate_training_contract(report, label=label)
    _validate_model_states(report, label=label)
    evaluations = _mapping(report.get("evaluations"), field=f"{label}.evaluations")
    if set(evaluations) != {"before", "after"}:
        raise RecurrentPairedAnalysisError(
            f"{label} evaluations must contain exactly before and after"
        )
    for phase in ("before", "after"):
        phase_report = _mapping(
            evaluations.get(phase),
            field=f"{label}.evaluations.{phase}",
        )
        if tuple(phase_report) != SELECTION_MODES and set(phase_report) != set(
            SELECTION_MODES
        ):
            raise RecurrentPairedAnalysisError(
                f"{label}.{phase} evaluation modes drifted"
            )
        for mode in SELECTION_MODES:
            _validate_evaluation(
                _mapping(
                    phase_report.get(mode),
                    field=f"{label}.evaluations.{phase}.{mode}",
                ),
                label=f"{label}.evaluations.{phase}.{mode}",
                mode=mode,
            )
    _validate_within_arm_evaluation_pairing(report, label=label)


def _validate_closed_lifecycle(
    report: Mapping[str, object],
    *,
    label: str,
) -> None:
    lifecycle = _mapping(report.get("lifecycle"), field=f"{label}.lifecycle")
    if set(lifecycle) != set(_LIFECYCLE_FLAGS):
        raise RecurrentPairedAnalysisError(
            f"{label} lifecycle keys differ from the closed contract"
        )
    open_flags = sorted(key for key, value in lifecycle.items() if value is not False)
    if open_flags:
        raise RecurrentPairedAnalysisError(f"{label} lifecycle is open: {open_flags}")
    for field in _TOP_LEVEL_CLOSED_FLAGS:
        if report.get(field) is not False:
            raise RecurrentPairedAnalysisError(
                f"{label} lifecycle top-level {field} must be false"
            )
    roles = _mapping(report.get("seed_roles"), field=f"{label}.seed_roles")
    for role in ("validation", "lockbox"):
        role_payload = _mapping(roles.get(role), field=f"{label}.seed_roles.{role}")
        if role_payload.get("accessed") is not False:
            raise RecurrentPairedAnalysisError(
                f"{label} {role} seeds must remain inaccessible"
            )
        if role_payload.get("seeds_materialized_by_canary") is not False:
            raise RecurrentPairedAnalysisError(
                f"{label} {role} seeds must not be materialized"
            )


def _validate_source(report: Mapping[str, object], *, label: str) -> None:
    source = _mapping(report.get("source"), field=f"{label}.source")
    if source.get("source_stable_during_run") is not True:
        raise RecurrentPairedAnalysisError(
            f"{label} source was not stable during the run"
        )
    if source.get("git_state_available") is not True:
        raise RecurrentPairedAnalysisError(f"{label} source git state is unavailable")
    git_head = source.get("git_head_observed")
    if not isinstance(git_head, str) or _GIT_HEAD_RE.fullmatch(git_head) is None:
        raise RecurrentPairedAnalysisError(f"{label} source git head is invalid")
    manifest = _mapping(
        source.get("source_file_hash_manifest"),
        field=f"{label}.source.source_file_hash_manifest",
    )
    if manifest.get("hash_algorithm") != "sha256":
        raise RecurrentPairedAnalysisError(f"{label} source manifest hash drifted")
    files = _mapping(manifest.get("files"), field=f"{label}.source.manifest.files")
    if not files:
        raise RecurrentPairedAnalysisError(f"{label} source manifest is empty")
    for path, digest in files.items():
        if not isinstance(path, str) or not path or path.startswith("/"):
            raise RecurrentPairedAnalysisError(
                f"{label} source manifest has an invalid path"
            )
        _sha256(digest, field=f"{label}.source.manifest.files[{path!r}]")
    if _nonnegative_int(
        manifest.get("file_count"), field=f"{label}.manifest.file_count"
    ) != len(files):
        raise RecurrentPairedAnalysisError(
            f"{label} source manifest file count is inconsistent"
        )
    aggregate = _sha256(
        manifest.get("aggregate_sha256"),
        field=f"{label}.source.manifest.aggregate_sha256",
    )
    if stable_payload_digest(dict(files)) != aggregate:
        raise RecurrentPairedAnalysisError(
            f"{label} source manifest aggregate is invalid"
        )


def _validate_preregistration(
    report: Mapping[str, object],
    *,
    label: str,
) -> None:
    prereg = _mapping(
        report.get("evaluation_preregistration"),
        field=f"{label}.evaluation_preregistration",
    )
    report_schema = report.get("schema_version")
    contract = (
        _CAUSAL_REPORT_CONTRACTS.get(report_schema)
        if isinstance(report_schema, str)
        else None
    )
    if contract is None:
        raise RecurrentPairedAnalysisError(
            f"{label} causal canary schema drifted before preregistration"
        )
    _policy_version, expected_preregistration = contract
    if prereg.get("schema_version") != expected_preregistration:
        raise RecurrentPairedAnalysisError(f"{label} preregistration schema drifted")
    exact = _sha256(prereg.get("exact_digest"), field=f"{label}.prereg.exact_digest")
    payload = copy.deepcopy(dict(prereg))
    payload.pop("exact_digest")
    synthetic_label = payload.pop("synthetic_noncandidate_digest_label", None)
    if stable_payload_digest(payload) != exact:
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration exact digest is invalid"
        )
    if synthetic_label != UNPINNED_LABEL_PREFIX + exact:
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration synthetic label is invalid"
        )
    if prereg.get("created_before_model_initialization") is not True:
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration was not created before initialization"
        )
    if tuple(prereg.get("action_selection_modes_in_order", ())) != SELECTION_MODES:
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration evaluation mode order drifted"
        )
    if prereg.get("before_after_sampling_stream_identical") is not True:
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration did not pin one before/after stream"
        )
    if prereg.get("candidate_controls") != ["mind_v3_linear", "masked_random"]:
        raise RecurrentPairedAnalysisError(f"{label} preregistration controls drifted")
    configuration = _mapping(
        report.get("configuration"),
        field=f"{label}.configuration",
    )
    if prereg.get("run_config") != configuration:
        raise RecurrentPairedAnalysisError(
            f"{label} configuration differs from its preregistration"
        )
    if prereg.get("seed_registry_sha256") != report.get("seed_registry_sha256"):
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration seed registry drifted"
        )
    if prereg.get("seed_roles") != report.get("seed_roles"):
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration seed roles drifted"
        )
    source = _mapping(report.get("source"), field=f"{label}.source")
    manifest = _mapping(
        source.get("source_file_hash_manifest"),
        field=f"{label}.source.manifest",
    )
    if prereg.get("source_manifest_aggregate_sha256") != manifest.get(
        "aggregate_sha256"
    ):
        raise RecurrentPairedAnalysisError(
            f"{label} preregistration source manifest drifted"
        )


def _validate_training_contract(
    report: Mapping[str, object],
    *,
    label: str,
) -> None:
    config = _mapping(report.get("configuration"), field=f"{label}.configuration")
    training = _mapping(report.get("training"), field=f"{label}.training")
    ppo = _mapping(config.get("ppo"), field=f"{label}.configuration.ppo")
    training_ppo = _mapping(
        training.get("ppo_config"),
        field=f"{label}.training.ppo_config",
    )
    if ppo != training_ppo:
        raise RecurrentPairedAnalysisError(
            f"{label} training PPO configuration differs from preregistration"
        )
    if config.get("model") != training.get("model_config"):
        raise RecurrentPairedAnalysisError(
            f"{label} training model configuration drifted"
        )
    learner_seed = _positive_int(
        training.get("learner_seed"), field=f"{label}.learner_seed"
    )
    if ppo.get("learner_seed") != learner_seed:
        raise RecurrentPairedAnalysisError(f"{label} learner seed drifted")
    seed_roles = _mapping(report.get("seed_roles"), field=f"{label}.seed_roles")
    learner = _mapping(seed_roles.get("learner"), field=f"{label}.seed_roles.learner")
    if learner.get("seed") != learner_seed:
        raise RecurrentPairedAnalysisError(f"{label} learner seed role drifted")
    if training.get("deterministic_algorithms_enabled") is not True:
        raise RecurrentPairedAnalysisError(
            f"{label} training did not enable deterministic algorithms"
        )
    if config.get("resolved_device") != training.get("device"):
        raise RecurrentPairedAnalysisError(f"{label} training device drifted")
    if config.get("training_scenarios") != training.get("training_scenarios"):
        raise RecurrentPairedAnalysisError(
            f"{label} training scenario sequence drifted"
        )
    schedule = config.get("training_schedule")
    if not isinstance(schedule, list) or not schedule:
        raise RecurrentPairedAnalysisError(f"{label} training schedule is invalid")
    if stable_payload_digest(schedule) != config.get("training_schedule_sha256"):
        raise RecurrentPairedAnalysisError(
            f"{label} training schedule digest is invalid"
        )
    updates = training.get("updates")
    if not isinstance(updates, list) or len(updates) != len(schedule):
        raise RecurrentPairedAnalysisError(
            f"{label} training update count differs from its schedule"
        )
    seen_ids: set[str] = set()
    seen_policy_identities: set[str] = set()
    seen_policy_seeds: set[int] = set()
    total_tasks = 0
    for update_index, (scheduled_tasks, update) in enumerate(
        zip(schedule, updates, strict=True)
    ):
        if not isinstance(scheduled_tasks, list) or not scheduled_tasks:
            raise RecurrentPairedAnalysisError(
                f"{label} training schedule update {update_index} is empty"
            )
        update_payload = _mapping(
            update, field=f"{label}.training.updates[{update_index}]"
        )
        if update_payload.get("update_index") != update_index:
            raise RecurrentPairedAnalysisError(f"{label} training update index drifted")
        if update_payload.get("tasks") != scheduled_tasks:
            raise RecurrentPairedAnalysisError(
                f"{label} training task schedule differs from executed tasks"
            )
        for task_index, task in enumerate(scheduled_tasks):
            parsed = _mapping(
                task,
                field=f"{label}.configuration.training_schedule[{update_index}][{task_index}]",
            )
            task_id = parsed.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in seen_ids:
                raise RecurrentPairedAnalysisError(
                    f"{label} training task id is missing or duplicated"
                )
            seen_ids.add(task_id)
            _positive_int(
                parsed.get("environment_seed"), field=f"{label}.environment_seed"
            )
            _positive_int(parsed.get("rollout_ticks"), field=f"{label}.rollout_ticks")
            identity = parsed.get("policy_sampling_identity")
            if (
                not isinstance(identity, str)
                or not identity
                or identity != identity.strip()
                or identity in seen_policy_identities
            ):
                raise RecurrentPairedAnalysisError(
                    f"{label} training policy sampling identity is invalid"
                )
            seen_policy_identities.add(identity)
            policy_seed = _positive_int(
                parsed.get("policy_sampling_seed"),
                field=f"{label} training policy sampling seed",
            )
            if policy_seed in seen_policy_seeds:
                raise RecurrentPairedAnalysisError(
                    f"{label} training policy sampling seed is duplicated"
                )
            seen_policy_seeds.add(policy_seed)
            if policy_seed != _derive_training_policy_sampling_seed(identity):
                raise RecurrentPairedAnalysisError(
                    f"{label} training policy sampling seed does not match its identity"
                )
            total_tasks += 1
    if _positive_int(
        config.get("updates"), field=f"{label}.configuration.updates"
    ) != len(schedule):
        raise RecurrentPairedAnalysisError(f"{label} configured update count drifted")
    worlds_per_update = _positive_int(
        config.get("worlds_per_update"),
        field=f"{label}.configuration.worlds_per_update",
    )
    if any(len(update) != worlds_per_update for update in schedule):
        raise RecurrentPairedAnalysisError(
            f"{label} training worlds-per-update drifted"
        )
    if (
        _positive_int(
            training.get("total_worlds"), field=f"{label}.training.total_worlds"
        )
        != total_tasks
    ):
        raise RecurrentPairedAnalysisError(f"{label} training world total drifted")


def _validate_model_states(report: Mapping[str, object], *, label: str) -> None:
    states = _mapping(report.get("model_states"), field=f"{label}.model_states")
    if states.get("changed") is not True:
        raise RecurrentPairedAnalysisError(
            f"{label} training did not change model state"
        )
    for field in (
        "same_before_snapshot_used_for_both_selection_modes",
        "same_after_model_used_for_both_selection_modes",
    ):
        if states.get(field) is not True:
            raise RecurrentPairedAnalysisError(
                f"{label} model/evaluation integrity flag {field} is false"
            )
    before = _mapping(states.get("before"), field=f"{label}.model_states.before")
    after = _mapping(states.get("after"), field=f"{label}.model_states.after")
    before_digest = _sha256(
        before.get("state_sha256"),
        field=f"{label}.model_states.before.state_sha256",
    )
    after_digest = _sha256(
        after.get("state_sha256"),
        field=f"{label}.model_states.after.state_sha256",
    )
    if before_digest == after_digest:
        raise RecurrentPairedAnalysisError(
            f"{label} before and after model fingerprints are identical"
        )


def _validate_evaluation(
    evaluation: Mapping[str, object],
    *,
    label: str,
    mode: str,
) -> None:
    evaluation_schema = evaluation.get("schema_version")
    if evaluation_schema == RECURRENT_EVALUATION_SCHEMA_VERSION:
        try:
            from evolution_sim.mind.recurrent_evaluation import (
                validate_recurrent_evaluation_report,
            )
        except ModuleNotFoundError as error:
            if error.name not in {"numpy", "torch"}:
                raise
            raise RecurrentPairedAnalysisError(
                f"{label} v4 recurrent evaluation validation requires the "
                "optional Mind ML dependencies"
            ) from error
        try:
            validate_recurrent_evaluation_report(evaluation)
        except RecurrentEvaluationError as error:
            raise RecurrentPairedAnalysisError(
                f"{label} nested recurrent evaluation failed its v4 contract"
            ) from error
    elif evaluation_schema != "mind_public_recurrent_evaluation_v3":
        raise RecurrentPairedAnalysisError(
            f"{label} recurrent evaluation schema drifted"
        )
    contract = _mapping(
        evaluation.get("evaluation_contract"),
        field=f"{label}.evaluation_contract",
    )
    if contract.get("candidate_action_selection") != mode:
        raise RecurrentPairedAnalysisError(f"{label} action selection mode drifted")
    if contract.get("ticks") != 120:
        raise RecurrentPairedAnalysisError(f"{label} horizon drifted")
    if contract.get("candidate_replay_verification") != "every_run_exact_digest_repeat":
        raise RecurrentPairedAnalysisError(f"{label} replay contract drifted")
    for field in (
        "strict_zero_unsupported_requested_actions",
        "strict_zero_heuristic_candidate_actions",
    ):
        if contract.get(field) is not True:
            raise RecurrentPairedAnalysisError(
                f"{label} integrity gate {field} is open"
            )
    stream_id = contract.get("candidate_sampling_stream_id")
    if not isinstance(stream_id, str) or not stream_id:
        raise RecurrentPairedAnalysisError(f"{label} sampling stream id is invalid")
    sampling_seeds = _sampling_seed_vector(
        contract.get("candidate_sampling_seeds"),
        field=f"{label}.candidate_sampling_seeds",
    )
    sampling_count = _positive_int(
        contract.get("candidate_sampling_seed_count"),
        field=f"{label}.candidate_sampling_seed_count",
    )
    if mode == ARGMAX_SELECTION:
        if sampling_seeds or sampling_count != 1:
            raise RecurrentPairedAnalysisError(
                f"{label} argmax sampling contract is invalid"
            )
        expected_sampling_seeds: tuple[int | None, ...] = (None,)
    else:
        if not sampling_seeds or sampling_count != len(sampling_seeds):
            raise RecurrentPairedAnalysisError(
                f"{label} sampled evaluation stream vector is invalid"
            )
        expected_derived_seeds = tuple(
            _derive_evaluation_policy_sampling_seed(
                stream_id=stream_id,
                replicate_index=index,
            )
            for index in range(sampling_count)
        )
        if sampling_seeds != expected_derived_seeds:
            raise RecurrentPairedAnalysisError(
                f"{label} sampled evaluation seeds do not match their stream derivation"
            )
        expected_sampling_seeds = sampling_seeds

    provenance = _mapping(
        evaluation.get("candidate_provenance"),
        field=f"{label}.candidate_provenance",
    )
    if (
        provenance.get("source_pinned") is not False
        or provenance.get("noncandidate_development_canary") is not True
        or provenance.get("promotion_evidence_eligible_from_provenance") is not False
    ):
        raise RecurrentPairedAnalysisError(
            f"{label} candidate provenance is not closed development evidence"
        )
    seed_plan = _mapping(evaluation.get("seed_plan"), field=f"{label}.seed_plan")
    broad_seeds = _seed_vector(
        seed_plan.get("broad_seeds"), field=f"{label}.broad_seeds"
    )
    fixture_seeds = _seed_vector(
        seed_plan.get("fixture_seeds"),
        field=f"{label}.fixture_seeds",
    )
    if seed_plan.get("train_evaluation_overlap_count") != 0:
        raise RecurrentPairedAnalysisError(
            f"{label} evaluation seeds overlap training seeds"
        )
    _positive_int(
        seed_plan.get("excluded_training_seed_count"),
        field=f"{label}.excluded_training_seed_count",
    )
    _sha256(seed_plan.get("digest"), field=f"{label}.seed_plan.digest")

    contexts = _evaluation_contexts(evaluation, label=label)
    replay_rows: dict[RunIdentity, str] = {}
    for context_name, context in contexts.items():
        expected_environment_seeds = (
            broad_seeds if context_name == "broad_default" else fixture_seeds
        )
        if tuple(context.get("seeds", ())) != expected_environment_seeds:
            raise RecurrentPairedAnalysisError(
                f"{label} context environment seed plan drifted"
            )
        if context.get("controls_run_once_per_environment") is not True:
            raise RecurrentPairedAnalysisError(
                f"{label} controls were not run once per environment"
            )
        if context.get("candidate_run_count_per_environment") != len(
            expected_sampling_seeds
        ):
            raise RecurrentPairedAnalysisError(
                f"{label} candidate sampling replicate count drifted"
            )
        if tuple(context.get("candidate_sampling_seeds", ())) != tuple(
            seed for seed in expected_sampling_seeds if seed is not None
        ):
            raise RecurrentPairedAnalysisError(
                f"{label} context sampling seed vector drifted"
            )
        policies = _mapping(
            context.get("policies"), field=f"{label}.{context_name}.policies"
        )
        if set(policies) != {"public_recurrent", "mind_v3_linear", "masked_random"}:
            raise RecurrentPairedAnalysisError(f"{label} policy set drifted")
        candidates = _policy_runs(policies, policy="public_recurrent", field=label)
        expected_identities = {
            (context_name, environment_seed, policy_sampling_seed)
            for environment_seed in expected_environment_seeds
            for policy_sampling_seed in expected_sampling_seeds
        }
        actual_identities: set[RunIdentity] = set()
        for run in candidates:
            identity = _validate_run(run, expected_context=context_name, label=label)
            if identity in actual_identities:
                raise RecurrentPairedAnalysisError(
                    f"{label} candidate run identity is duplicated"
                )
            actual_identities.add(identity)
            replay_rows[identity] = str(run["replay_digest"])
        if actual_identities != expected_identities:
            raise RecurrentPairedAnalysisError(
                f"{label} candidate environment/sampling stream grid drifted"
            )
        for control in ("mind_v3_linear", "masked_random"):
            control_runs = _policy_runs(policies, policy=control, field=label)
            if len(control_runs) != len(expected_environment_seeds):
                raise RecurrentPairedAnalysisError(
                    f"{label} {control} control count drifted"
                )
            for run, seed in zip(control_runs, expected_environment_seeds, strict=True):
                identity = _validate_run(
                    run,
                    expected_context=context_name,
                    label=f"{label}.{control}",
                    require_candidate_integrity=False,
                )
                if identity != (context_name, seed, None):
                    raise RecurrentPairedAnalysisError(
                        f"{label} {control} control identity drifted"
                    )

    replay = _mapping(
        evaluation.get("replay_verification"),
        field=f"{label}.replay_verification",
    )
    if replay.get("all_passed") is not True:
        raise RecurrentPairedAnalysisError(f"{label} replay verification failed")
    checks = replay.get("checks")
    if not isinstance(checks, list):
        raise RecurrentPairedAnalysisError(f"{label} replay checks must be a list")
    if replay.get("checked_run_count") != len(checks) or len(checks) != len(
        replay_rows
    ):
        raise RecurrentPairedAnalysisError(f"{label} replay check count drifted")
    check_rows: dict[RunIdentity, str] = {}
    for index, check in enumerate(checks):
        parsed = _mapping(check, field=f"{label}.replay.checks[{index}]")
        if parsed.get("passed") is not True:
            raise RecurrentPairedAnalysisError(f"{label} replay check failed")
        identity = _run_identity(parsed, field=f"{label}.replay.checks[{index}]")
        if identity in check_rows:
            raise RecurrentPairedAnalysisError(
                f"{label} replay check identity is duplicated"
            )
        check_rows[identity] = _sha256(
            parsed.get("digest"),
            field=f"{label}.replay.checks[{index}].digest",
        )
    if check_rows != replay_rows:
        raise RecurrentPairedAnalysisError(
            f"{label} replay checks do not match candidate runs"
        )
    carrion = contexts.get("fixture:carrion_only")
    if carrion is not None:
        policies = _mapping(carrion.get("policies"), field=f"{label}.carrion.policies")
        survivors = sum(
            int(run["terminal_alive"]) > 0
            for run in _policy_runs(
                policies,
                policy="public_recurrent",
                field=f"{label}.carrion",
            )
        )
        alive_agent_total = sum(
            int(run["terminal_alive"])
            for run in _policy_runs(
                policies,
                policy="public_recurrent",
                field=f"{label}.carrion",
            )
        )
        if evaluation_schema == RECURRENT_EVALUATION_SCHEMA_VERSION:
            if (
                evaluation.get("carrion_fixture_terminal_nonextinct_run_count")
                != survivors
                or evaluation.get("carrion_fixture_terminal_alive_agent_total")
                != alive_agent_total
            ):
                raise RecurrentPairedAnalysisError(
                    f"{label} carrion run/agent terminal counts drifted"
                )
        elif evaluation.get("carrion_fixture_terminal_survivor_count") != survivors:
            raise RecurrentPairedAnalysisError(
                f"{label} legacy carrion terminal survivor count drifted"
            )


def _validate_run(
    run: Mapping[str, object],
    *,
    expected_context: str,
    label: str,
    require_candidate_integrity: bool = True,
) -> RunIdentity:
    identity = _run_identity(run, field=label)
    if identity[0] != expected_context:
        raise RecurrentPairedAnalysisError(f"{label} run context drifted")
    for field in _REQUIRED_RUN_COUNT_FIELDS:
        _nonnegative_int(run.get(field), field=f"{label}.{field}")
    _finite_number(run.get("reward_total"), field=f"{label}.reward_total")
    _finite_number(
        run.get("dominant_requested_action_share"),
        field=f"{label}.dominant_requested_action_share",
    )
    horizon_ticks = _positive_int(
        run.get("horizon_ticks"),
        field=f"{label}.horizon_ticks",
    )
    if horizon_ticks != 120:
        raise RecurrentPairedAnalysisError(f"{label} run horizon drifted")
    ticks_executed = _nonnegative_int(
        run.get("ticks_executed"),
        field=f"{label}.ticks_executed",
    )
    if ticks_executed > horizon_ticks:
        raise RecurrentPairedAnalysisError(
            f"{label} executed ticks exceed the configured horizon"
        )
    counts = _mapping(
        run.get("requested_action_counts"),
        field=f"{label}.requested_action_counts",
    )
    if not counts or any(action not in ACTION_NAMES for action in counts):
        raise RecurrentPairedAnalysisError(
            f"{label} requested action vocabulary drifted"
        )
    parsed_counts = {
        action: _nonnegative_int(
            count, field=f"{label}.requested_action_counts.{action}"
        )
        for action, count in counts.items()
    }
    trajectory_count = int(run["trajectory_record_count"])
    decision_count = int(run["policy_decision_record_count"])
    passive_count = int(run["passive_trajectory_record_count"])
    if (
        trajectory_count <= 0
        or decision_count <= 0
        or passive_count < 0
        or decision_count + passive_count != trajectory_count
    ):
        raise RecurrentPairedAnalysisError(
            f"{label} decision and passive counts do not cover trajectory records"
        )
    if sum(parsed_counts.values()) != decision_count:
        raise RecurrentPairedAnalysisError(
            f"{label} requested action counts do not match policy decisions"
        )
    dominant_count = max(parsed_counts.values())
    dominant_action = run.get("dominant_requested_action")
    if (
        dominant_action not in parsed_counts
        or parsed_counts[dominant_action] != dominant_count
    ):
        raise RecurrentPairedAnalysisError(f"{label} dominant action is inconsistent")
    if run.get("dominant_requested_action_count") != dominant_count:
        raise RecurrentPairedAnalysisError(
            f"{label} dominant action count is inconsistent"
        )
    if not math.isclose(
        float(run["dominant_requested_action_share"]),
        dominant_count / decision_count,
        rel_tol=0.0,
        abs_tol=1.0e-4,
    ):
        raise RecurrentPairedAnalysisError(
            f"{label} dominant action share is inconsistent"
        )
    if run.get("eat_requested_count") != parsed_counts.get("eat", 0):
        raise RecurrentPairedAnalysisError(f"{label} eat action count is inconsistent")
    without_gain = int(run["eat_without_positive_resource_gain_count"])
    eat_count = int(run["eat_requested_count"])
    if without_gain > eat_count:
        raise RecurrentPairedAnalysisError(
            f"{label} eat-without-gain count exceeds eat requests"
        )
    without_gain_share = run.get("eat_without_positive_resource_gain_share")
    if eat_count == 0:
        if without_gain_share is not None:
            raise RecurrentPairedAnalysisError(
                f"{label} zero-eat run must have null eat-without-gain share"
            )
    else:
        parsed_share = _finite_number(
            without_gain_share,
            field=f"{label}.eat_without_positive_resource_gain_share",
        )
        if not math.isclose(
            parsed_share,
            without_gain / eat_count,
            rel_tol=0.0,
            abs_tol=1.0e-4,
        ):
            raise RecurrentPairedAnalysisError(
                f"{label} eat-without-gain share is inconsistent"
            )
    if require_candidate_integrity and (
        run.get("unsupported_requested_action_count") != 0
        or run.get("heuristic_action_source_count") != 0
    ):
        raise RecurrentPairedAnalysisError(f"{label} candidate action integrity failed")
    reward_components = run.get("reward_component_totals")
    if reward_components is not None:
        components = _mapping(
            reward_components,
            field=f"{label}.reward_component_totals",
        )
        if not components:
            raise RecurrentPairedAnalysisError(f"{label} reward components are empty")
        component_total = 0.0
        for component, value in components.items():
            if not isinstance(component, str) or not component:
                raise RecurrentPairedAnalysisError(
                    f"{label} reward component name is invalid"
                )
            component_total += _finite_number(
                value,
                field=f"{label}.reward_component_totals.{component}",
            )
        if not math.isclose(
            component_total,
            float(run["reward_total"]),
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise RecurrentPairedAnalysisError(
                f"{label} reward component totals do not sum to reward_total"
            )
    distribution = run.get("learned_masked_distribution")
    if distribution is not None:
        distribution_payload = _mapping(
            distribution,
            field=f"{label}.learned_masked_distribution",
        )
        if require_candidate_integrity:
            _validate_learned_distribution_summary(
                distribution_payload,
                policy_decision_count=decision_count,
                label=label,
            )
        else:
            _validate_absent_control_distribution(
                distribution_payload,
                label=label,
            )
    _sha256(run.get("replay_digest"), field=f"{label}.replay_digest")
    return identity


def _validate_within_arm_evaluation_pairing(
    report: Mapping[str, object],
    *,
    label: str,
) -> None:
    evaluations = _mapping(report.get("evaluations"), field=f"{label}.evaluations")
    before = _mapping(evaluations.get("before"), field=f"{label}.before")
    after = _mapping(evaluations.get("after"), field=f"{label}.after")
    for mode in SELECTION_MODES:
        before_eval = _mapping(before.get(mode), field=f"{label}.before.{mode}")
        after_eval = _mapping(after.get(mode), field=f"{label}.after.{mode}")
        if before_eval.get("evaluation_contract") != after_eval.get(
            "evaluation_contract"
        ):
            raise RecurrentPairedAnalysisError(
                f"{label} before/after sampling contract drifted"
            )
        if before_eval.get("seed_plan") != after_eval.get("seed_plan"):
            raise RecurrentPairedAnalysisError(
                f"{label} before/after evaluation seed plan drifted"
            )
        before_contexts = _evaluation_contexts(
            before_eval, label=f"{label}.before.{mode}"
        )
        after_contexts = _evaluation_contexts(after_eval, label=f"{label}.after.{mode}")
        if tuple(before_contexts) != tuple(after_contexts):
            raise RecurrentPairedAnalysisError(
                f"{label} before/after evaluation contexts drifted"
            )
        for context in before_contexts:
            before_policies = _mapping(
                before_contexts[context].get("policies"),
                field=f"{label}.before.{context}.policies",
            )
            after_policies = _mapping(
                after_contexts[context].get("policies"),
                field=f"{label}.after.{context}.policies",
            )
            for control in ("mind_v3_linear", "masked_random"):
                if before_policies.get(control) != after_policies.get(control):
                    raise RecurrentPairedAnalysisError(
                        f"{label} {control} control drifted before/after"
                    )
            before_runs = _run_map(before_policies, field=f"{label}.before.{context}")
            after_runs = _run_map(after_policies, field=f"{label}.after.{context}")
            if set(before_runs) != set(after_runs):
                raise RecurrentPairedAnalysisError(
                    f"{label} before/after candidate run identity drifted"
                )
            for identity in before_runs:
                if set(_run_metric_vector(before_runs[identity])) != set(
                    _run_metric_vector(after_runs[identity])
                ):
                    raise RecurrentPairedAnalysisError(
                        f"{label} before/after reward or action metric schema drifted"
                    )


def _validate_pair_contract(
    gru: Mapping[str, object],
    feed_forward: Mapping[str, object],
) -> dict[str, object]:
    gru_source = _mapping(gru.get("source"), field="gru.source")
    ff_source = _mapping(feed_forward.get("source"), field="feed_forward.source")
    gru_manifest = _mapping(
        gru_source.get("source_file_hash_manifest"),
        field="gru.source.manifest",
    )
    ff_manifest = _mapping(
        ff_source.get("source_file_hash_manifest"),
        field="feed_forward.source.manifest",
    )
    if (
        gru_source.get("git_head_observed") != ff_source.get("git_head_observed")
        or gru_manifest != ff_manifest
    ):
        raise RecurrentPairedAnalysisError(
            "GRU and feed-forward source head/manifest differ"
        )
    if gru.get("seed_registry_sha256") != feed_forward.get("seed_registry_sha256"):
        raise RecurrentPairedAnalysisError("paired seed registry drifted")

    gru_config = _mapping(gru.get("configuration"), field="gru.configuration")
    ff_config = _mapping(
        feed_forward.get("configuration"),
        field="feed_forward.configuration",
    )
    _assert_ablation_role(gru, expected=False, label="gru")
    _assert_ablation_role(feed_forward, expected=True, label="feed_forward")
    if _normalized_ablation_config(gru_config) != _normalized_ablation_config(
        ff_config
    ):
        raise RecurrentPairedAnalysisError(
            "paired configuration differs beyond feed_forward_history_ablation"
        )
    gru_training = _mapping(gru.get("training"), field="gru.training")
    ff_training = _mapping(
        feed_forward.get("training"),
        field="feed_forward.training",
    )
    training_contract_fields = (
        "contract_version",
        "learner_seed",
        "device",
        "deterministic_algorithms_enabled",
        "model_config",
        "ppo_config",
        "rollout_execution",
        "training_scenarios",
    )
    gru_training_contract = {
        field: copy.deepcopy(gru_training.get(field))
        for field in training_contract_fields
    }
    ff_training_contract = {
        field: copy.deepcopy(ff_training.get(field))
        for field in training_contract_fields
    }
    _replace_nested_ablation(gru_training_contract, path=("ppo_config",))
    _replace_nested_ablation(ff_training_contract, path=("ppo_config",))
    if gru_training_contract != ff_training_contract:
        raise RecurrentPairedAnalysisError(
            "paired training configuration differs beyond the ablation"
        )
    if gru_config.get("training_schedule") != ff_config.get("training_schedule"):
        raise RecurrentPairedAnalysisError(
            "paired training task schedule or policy sampling seeds differ"
        )
    gru_states = _mapping(gru.get("model_states"), field="gru.model_states")
    ff_states = _mapping(
        feed_forward.get("model_states"),
        field="feed_forward.model_states",
    )
    if gru_states.get("before") != ff_states.get("before"):
        raise RecurrentPairedAnalysisError("paired before model fingerprint differs")
    gru_prereg = _normalized_preregistration(gru)
    ff_prereg = _normalized_preregistration(feed_forward)
    if gru_prereg != ff_prereg:
        raise RecurrentPairedAnalysisError(
            "paired preregistration differs beyond feed_forward_history_ablation"
        )

    for phase in ("before", "after"):
        for mode in SELECTION_MODES:
            gru_eval = _evaluation_for(gru, phase=phase, mode=mode)
            ff_eval = _evaluation_for(feed_forward, phase=phase, mode=mode)
            gru_contract = copy.deepcopy(
                dict(
                    _mapping(
                        gru_eval.get("evaluation_contract"),
                        field=f"gru.{phase}.{mode}.contract",
                    )
                )
            )
            ff_contract = copy.deepcopy(
                dict(
                    _mapping(
                        ff_eval.get("evaluation_contract"),
                        field=f"feed_forward.{phase}.{mode}.contract",
                    )
                )
            )
            gru_contract["candidate_recurrent_state"] = "<ABLATION>"
            ff_contract["candidate_recurrent_state"] = "<ABLATION>"
            if gru_contract != ff_contract:
                raise RecurrentPairedAnalysisError(
                    "paired evaluation sampling contract or stream vector differs"
                )
            if gru_eval.get("seed_plan") != ff_eval.get("seed_plan"):
                raise RecurrentPairedAnalysisError(
                    "paired evaluation environment seed plan differs"
                )
            gru_contexts = _evaluation_contexts(gru_eval, label=f"gru.{phase}.{mode}")
            ff_contexts = _evaluation_contexts(
                ff_eval,
                label=f"feed_forward.{phase}.{mode}",
            )
            if tuple(gru_contexts) != tuple(ff_contexts):
                raise RecurrentPairedAnalysisError(
                    "paired evaluation fixture contexts differ"
                )
            for context in gru_contexts:
                gru_policies = _mapping(
                    gru_contexts[context].get("policies"),
                    field=f"gru.{context}.policies",
                )
                ff_policies = _mapping(
                    ff_contexts[context].get("policies"),
                    field=f"feed_forward.{context}.policies",
                )
                for control in ("mind_v3_linear", "masked_random"):
                    if gru_policies.get(control) != ff_policies.get(control):
                        raise RecurrentPairedAnalysisError(
                            f"paired {control} control differs across arms"
                        )
                gru_runs = _run_map(gru_policies, field=f"gru.{context}")
                ff_runs = _run_map(ff_policies, field=f"feed_forward.{context}")
                if set(gru_runs) != set(ff_runs):
                    raise RecurrentPairedAnalysisError(
                        "paired environment/policy-sampling run grid differs"
                    )
                for identity in gru_runs:
                    if set(_run_metric_vector(gru_runs[identity])) != set(
                        _run_metric_vector(ff_runs[identity])
                    ):
                        raise RecurrentPairedAnalysisError(
                            "paired reward/action metric schema differs"
                        )

    stream = _mapping(
        gru_config.get("evaluation_sampling_stream"),
        field="gru.configuration.evaluation_sampling_stream",
    )
    if stream.get("arm_independent") is not True:
        raise RecurrentPairedAnalysisError(
            "paired evaluation sampling stream is not arm-independent"
        )
    stream_id = stream.get("id")
    if not isinstance(stream_id, str) or not stream_id:
        raise RecurrentPairedAnalysisError(
            "paired evaluation sampling stream id is invalid"
        )
    sampled_contract = _mapping(
        _evaluation_for(gru, phase="before", mode=SAMPLED_SELECTION).get(
            "evaluation_contract"
        ),
        field="gru.sampled.evaluation_contract",
    )
    if sampled_contract.get("candidate_sampling_stream_id") != stream_id:
        raise RecurrentPairedAnalysisError(
            "preregistered and executed evaluation sampling stream differ"
        )
    stream_contract = _mapping(
        stream.get("contract"),
        field="gru.configuration.evaluation_sampling_stream.contract",
    )
    sampled_seed_count = _positive_int(
        sampled_contract.get("candidate_sampling_seed_count"),
        field="paired sampled seed count",
    )
    if (
        gru_config.get("candidate_sampling_seed_count") != sampled_seed_count
        or stream_contract.get("candidate_sampling_seed_count") != sampled_seed_count
    ):
        raise RecurrentPairedAnalysisError(
            "configured and executed policy-sampling stream counts differ"
        )
    seed_plan = _mapping(
        _evaluation_for(gru, phase="before", mode=SAMPLED_SELECTION).get("seed_plan"),
        field="gru.sampled.seed_plan",
    )
    if stream_contract.get("seed_plan_digest") != seed_plan.get("digest"):
        raise RecurrentPairedAnalysisError(
            "preregistered sampling stream seed plan digest drifted"
        )
    actual_fixtures = [
        context.removeprefix("fixture:")
        for context in _evaluation_contexts(
            _evaluation_for(gru, phase="before", mode=SAMPLED_SELECTION),
            label="gru.sampled",
        )
        if context.startswith("fixture:")
    ]
    if (
        stream_contract.get("fixtures") != actual_fixtures
        or gru_config.get("evaluation_fixtures") != actual_fixtures
    ):
        raise RecurrentPairedAnalysisError(
            "preregistered and executed evaluation fixture plan differs"
        )
    learner_seed = _positive_int(
        gru_training.get("learner_seed"),
        field="paired learner seed",
    )
    return {
        "learner_seed": learner_seed,
        "git_head_observed": gru_source["git_head_observed"],
        "source_manifest_aggregate_sha256": gru_manifest["aggregate_sha256"],
        "training_schedule_sha256": gru_config["training_schedule_sha256"],
        "before_model_state_sha256": _mapping(
            gru_states.get("before"),
            field="gru.model_states.before",
        )["state_sha256"],
        "evaluation_sampling_stream_id": stream_id,
    }


def _assert_ablation_role(
    report: Mapping[str, object],
    *,
    expected: bool,
    label: str,
) -> None:
    config = _mapping(report.get("configuration"), field=f"{label}.configuration")
    training = _mapping(report.get("training"), field=f"{label}.training")
    prereg = _mapping(
        report.get("evaluation_preregistration"),
        field=f"{label}.evaluation_preregistration",
    )
    prereg_config = _mapping(
        prereg.get("run_config"), field=f"{label}.prereg.run_config"
    )
    values = (
        config.get("feed_forward_history_ablation"),
        _mapping(config.get("ppo"), field=f"{label}.configuration.ppo").get(
            "feed_forward_history_ablation"
        ),
        _mapping(training.get("ppo_config"), field=f"{label}.training.ppo_config").get(
            "feed_forward_history_ablation"
        ),
        prereg_config.get("feed_forward_history_ablation"),
        _mapping(prereg_config.get("ppo"), field=f"{label}.prereg.ppo").get(
            "feed_forward_history_ablation"
        ),
    )
    if any(value is not expected for value in values):
        raise RecurrentPairedAnalysisError(
            f"{label} feed-forward history ablation role is inconsistent"
        )
    expected_state = (
        "zero_before_every_decision" if expected else "one_hidden_state_per_agent"
    )
    for phase in ("before", "after"):
        for mode in SELECTION_MODES:
            contract = _mapping(
                _evaluation_for(report, phase=phase, mode=mode).get(
                    "evaluation_contract"
                ),
                field=f"{label}.{phase}.{mode}.evaluation_contract",
            )
            # Older v2-compatible synthetic reports may omit the descriptive
            # field; when present it must prove the intended arm semantics.
            state = contract.get("candidate_recurrent_state")
            if state is not None and state != expected_state:
                raise RecurrentPairedAnalysisError(
                    f"{label} evaluation recurrent ablation semantics drifted"
                )


def _normalized_ablation_config(config: Mapping[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(dict(config))
    normalized["feed_forward_history_ablation"] = "<ABLATION>"
    ppo = _mapping(normalized.get("ppo"), field="normalized configuration.ppo")
    ppo["feed_forward_history_ablation"] = "<ABLATION>"  # type: ignore[index]
    return normalized


def _normalized_preregistration(report: Mapping[str, object]) -> dict[str, object]:
    prereg = copy.deepcopy(
        dict(
            _mapping(
                report.get("evaluation_preregistration"),
                field="evaluation_preregistration",
            )
        )
    )
    prereg.pop("exact_digest", None)
    prereg.pop("synthetic_noncandidate_digest_label", None)
    run_config = _mapping(prereg.get("run_config"), field="prereg.run_config")
    prereg["run_config"] = _normalized_ablation_config(run_config)
    return prereg


def _replace_nested_ablation(
    payload: dict[str, object], *, path: Sequence[str]
) -> None:
    current: Mapping[str, object] = payload
    for key in path:
        current = _mapping(current.get(key), field=f"normalized.{key}")
    current["feed_forward_history_ablation"] = "<ABLATION>"  # type: ignore[index]


def _arm_training_effects(
    report: Mapping[str, object],
    *,
    mode: str,
    label: str,
) -> dict[RunIdentity, dict[str, object]]:
    before = _candidate_run_map(report, phase="before", mode=mode)
    after = _candidate_run_map(report, phase="after", mode=mode)
    if set(before) != set(after):
        raise RecurrentPairedAnalysisError(
            f"{label} before/after candidate identity drifted"
        )
    effects: dict[RunIdentity, dict[str, object]] = {}
    for identity in sorted(before, key=_identity_sort_key):
        before_metrics = _run_metric_vector(before[identity])
        after_metrics = _run_metric_vector(after[identity])
        if set(before_metrics) != set(after_metrics):
            raise RecurrentPairedAnalysisError(
                f"{label} before/after metric schema drifted"
            )
        effects[identity] = {
            "metrics": _subtract_vectors(after_metrics, before_metrics),
            "requested_action_count": _subtract_vectors(
                _action_count_vector(after[identity]),
                _action_count_vector(before[identity]),
            ),
            "requested_action_share": _subtract_vectors(
                _action_share_vector(after[identity]),
                _action_share_vector(before[identity]),
            ),
        }
    return effects


def _difference_in_differences(
    gru_effects: Mapping[RunIdentity, Mapping[str, object]],
    ff_effects: Mapping[RunIdentity, Mapping[str, object]],
    *,
    mode: str,
) -> dict[str, object]:
    if set(gru_effects) != set(ff_effects) or not gru_effects:
        raise RecurrentPairedAnalysisError(
            f"{mode} paired run identities differ across arms"
        )
    rows: list[dict[str, object]] = []
    for identity in sorted(gru_effects, key=_identity_sort_key):
        gru_effect = gru_effects[identity]
        ff_effect = ff_effects[identity]
        row = {
            "context": identity[0],
            "environment_seed": identity[1],
            "policy_sampling_seed": identity[2],
            "metrics": _subtract_vectors(
                _mapping(gru_effect.get("metrics"), field="gru effect metrics"),
                _mapping(ff_effect.get("metrics"), field="feed-forward effect metrics"),
            ),
            "requested_action_count": _subtract_vectors(
                _mapping(
                    gru_effect.get("requested_action_count"),
                    field="gru action count effect",
                ),
                _mapping(
                    ff_effect.get("requested_action_count"),
                    field="feed-forward action count effect",
                ),
            ),
            "requested_action_share": _subtract_vectors(
                _mapping(
                    gru_effect.get("requested_action_share"),
                    field="gru action share effect",
                ),
                _mapping(
                    ff_effect.get("requested_action_share"),
                    field="feed-forward action share effect",
                ),
            ),
            "arm_training_effects": {
                "gru_after_minus_before": copy.deepcopy(dict(gru_effect)),
                "true_feed_forward_after_minus_before": copy.deepcopy(dict(ff_effect)),
            },
        }
        rows.append(row)
    return {
        "matched_run_count": len(rows),
        "environment_count": len({int(row["environment_seed"]) for row in rows}),
        "policy_sampling_stream_count": len(
            {row["policy_sampling_seed"] for row in rows}
        ),
        "rows": rows,
        "overall_mean": _mean_effect_rows(rows),
        "by_context": {
            context: _mean_effect_rows(
                [row for row in rows if row["context"] == context]
            )
            for context in sorted({str(row["context"]) for row in rows})
        },
        "by_context_and_environment": _grouped_effect_rows(
            rows,
            keys=("context", "environment_seed"),
        ),
        "by_context_and_sampling_stream": _grouped_effect_rows(
            rows,
            keys=("context", "policy_sampling_seed"),
        ),
        "repeated_measurement_warning": (
            "policy-sampling streams within an environment and environments "
            "within this run are not independent learner replications"
        ),
    }


def _learner_level_carrion_effect(
    *,
    learner_seed: int,
    arm_effects_by_mode: Mapping[
        str,
        tuple[
            Mapping[RunIdentity, Mapping[str, object]],
            Mapping[RunIdentity, Mapping[str, object]],
        ],
    ],
) -> dict[str, object]:
    modes: dict[str, object] = {}
    for mode, (gru_effects, ff_effects) in arm_effects_by_mode.items():
        gru_carrion = {
            identity: value
            for identity, value in gru_effects.items()
            if identity[0] == "fixture:carrion_only"
        }
        ff_carrion = {
            identity: value
            for identity, value in ff_effects.items()
            if identity[0] == "fixture:carrion_only"
        }
        if not gru_carrion or set(gru_carrion) != set(ff_carrion):
            raise RecurrentPairedAnalysisError(
                "paired reports must contain matched carrion fixture runs"
            )
        gru_clustered = _environment_clustered_arm_mean(gru_carrion)
        ff_clustered = _environment_clustered_arm_mean(ff_carrion)
        modes[mode] = {
            "environment_count": len({identity[1] for identity in gru_carrion}),
            "policy_sampling_stream_count": len(
                {identity[2] for identity in gru_carrion}
            ),
            "matched_environment_stream_observation_count": len(gru_carrion),
            "aggregation": (
                "mean_within_environment_over_policy_streams_then_mean_"
                "across_environments"
            ),
            "gru_after_minus_before": gru_clustered,
            "true_feed_forward_after_minus_before": ff_clustered,
            "gru_minus_true_feed_forward_difference_in_differences": {
                "metrics": _subtract_vectors(
                    _mapping(gru_clustered.get("metrics"), field="gru carrion metrics"),
                    _mapping(ff_clustered.get("metrics"), field="ff carrion metrics"),
                ),
                "requested_action_count": _subtract_vectors(
                    _mapping(
                        gru_clustered.get("requested_action_count"),
                        field="gru carrion action counts",
                    ),
                    _mapping(
                        ff_clustered.get("requested_action_count"),
                        field="ff carrion action counts",
                    ),
                ),
                "requested_action_share": _subtract_vectors(
                    _mapping(
                        gru_clustered.get("requested_action_share"),
                        field="gru carrion action shares",
                    ),
                    _mapping(
                        ff_clustered.get("requested_action_share"),
                        field="ff carrion action shares",
                    ),
                ),
            },
        }
    return {
        "experimental_unit": "learner_seed",
        "learner_seed": learner_seed,
        "learner_replication_count_per_arm": 1,
        "sampling_streams_are_learner_replications": False,
        "environment_seeds_are_learner_replications": False,
        "inferential_claim_authorized": False,
        "selection_modes": modes,
    }


def _environment_clustered_arm_mean(
    effects: Mapping[RunIdentity, Mapping[str, object]],
) -> dict[str, object]:
    by_environment: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for identity, effect in effects.items():
        by_environment[identity[1]].append(effect)
    environment_means = [_mean_arm_effects(rows) for rows in by_environment.values()]
    return _mean_arm_effects(environment_means)


def _mean_arm_effects(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise RecurrentPairedAnalysisError("cannot average an empty effect set")
    return {
        key: _mean_vectors(
            [_mapping(row.get(key), field=f"effect.{key}") for row in rows]
        )
        for key in ("metrics", "requested_action_count", "requested_action_share")
    }


def _mean_effect_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise RecurrentPairedAnalysisError("cannot average an empty paired row set")
    return {
        "observation_count": len(rows),
        "metrics": _mean_vectors(
            [_mapping(row.get("metrics"), field="row.metrics") for row in rows]
        ),
        "requested_action_count": _mean_vectors(
            [
                _mapping(row.get("requested_action_count"), field="row.action_count")
                for row in rows
            ]
        ),
        "requested_action_share": _mean_vectors(
            [
                _mapping(row.get("requested_action_share"), field="row.action_share")
                for row in rows
            ]
        ),
    }


def _grouped_effect_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity in sorted(
        groups, key=lambda item: tuple(_sort_value(v) for v in item)
    ):
        output.append(
            {
                **dict(zip(keys, identity, strict=True)),
                **_mean_effect_rows(groups[identity]),
            }
        )
    return output


def _run_metric_vector(run: Mapping[str, object]) -> dict[str, float]:
    eat_share = run.get("eat_without_positive_resource_gain_share")
    metrics = {
        "terminal_survival": 1.0 if int(run["terminal_alive"]) > 0 else 0.0,
        "terminal_alive": float(run["terminal_alive"]),
        "births": float(run["births"]),
        "deaths": float(run["deaths"]),
        "reward_total": float(run["reward_total"]),
        "trajectory_record_count": float(run["trajectory_record_count"]),
        "policy_decision_record_count": float(run["policy_decision_record_count"]),
        "passive_trajectory_record_count": float(
            run["passive_trajectory_record_count"]
        ),
        "dominant_requested_action_share": float(
            run["dominant_requested_action_share"]
        ),
        "unsupported_requested_action_count": float(
            run["unsupported_requested_action_count"]
        ),
        "heuristic_action_source_count": float(run["heuristic_action_source_count"]),
        "eat_requested_count": float(run["eat_requested_count"]),
        "eat_without_positive_resource_gain_count": float(
            run["eat_without_positive_resource_gain_count"]
        ),
        "eat_without_positive_resource_gain_share": (
            0.0 if eat_share is None else float(eat_share)
        ),
    }
    components = run.get("reward_component_totals")
    if components is not None:
        for component, value in _mapping(
            components,
            field="run.reward_component_totals",
        ).items():
            metrics[f"reward_component:{component}"] = float(value)
    distribution = run.get("learned_masked_distribution")
    if distribution is not None:
        distribution_payload = _mapping(
            distribution,
            field="run.learned_masked_distribution",
        )
        means = _mapping(
            distribution_payload.get("metric_means"),
            field="run.learned_masked_distribution.metric_means",
        )
        for metric, value in means.items():
            if value is not None:
                metrics[f"learned_distribution:{metric}"] = _finite_number(
                    value,
                    field=f"run.learned_distribution.{metric}",
                )
        action_probabilities = _mapping(
            distribution_payload.get("mean_action_probabilities"),
            field="run.learned_masked_distribution.mean_action_probabilities",
        )
        for action, value in action_probabilities.items():
            if value is not None:
                metrics[f"learned_probability:{action}"] = _finite_number(
                    value,
                    field=f"run.learned_probability.{action}",
                )
    if set(_BASE_METRICS) - set(metrics):
        raise RecurrentPairedAnalysisError("run metric vector is incomplete")
    return metrics


def _action_count_vector(run: Mapping[str, object]) -> dict[str, float]:
    counts = _mapping(
        run.get("requested_action_counts"), field="requested_action_counts"
    )
    return {action: float(counts.get(action, 0)) for action in ACTION_NAMES}


def _action_share_vector(run: Mapping[str, object]) -> dict[str, float]:
    counts = _action_count_vector(run)
    total = float(run["policy_decision_record_count"])
    return {action: count / total for action, count in counts.items()}


def _subtract_vectors(
    minuend: Mapping[str, object],
    subtrahend: Mapping[str, object],
) -> dict[str, float]:
    if set(minuend) != set(subtrahend):
        raise RecurrentPairedAnalysisError("effect metric schema drifted")
    return {
        key: _rounded(float(minuend[key]) - float(subtrahend[key]))
        for key in sorted(minuend)
    }


def _mean_vectors(vectors: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not vectors:
        raise RecurrentPairedAnalysisError("cannot average empty metric vectors")
    keys = set(vectors[0])
    if any(set(vector) != keys for vector in vectors):
        raise RecurrentPairedAnalysisError(
            "effect metric schema drifted while averaging"
        )
    return {
        key: _rounded(
            math.fsum(float(vector[key]) for vector in vectors) / len(vectors)
        )
        for key in sorted(keys)
    }


def _candidate_run_map(
    report: Mapping[str, object],
    *,
    phase: str,
    mode: str,
) -> dict[RunIdentity, Mapping[str, object]]:
    evaluation = _evaluation_for(report, phase=phase, mode=mode)
    output: dict[RunIdentity, Mapping[str, object]] = {}
    for context in _evaluation_contexts(evaluation, label=f"{phase}.{mode}").values():
        policies = _mapping(context.get("policies"), field=f"{phase}.{mode}.policies")
        for identity, run in _run_map(policies, field=f"{phase}.{mode}").items():
            if identity in output:
                raise RecurrentPairedAnalysisError(
                    "candidate run identity duplicated across contexts"
                )
            output[identity] = run
    return output


def _evaluation_for(
    report: Mapping[str, object],
    *,
    phase: str,
    mode: str,
) -> Mapping[str, object]:
    evaluations = _mapping(report.get("evaluations"), field="evaluations")
    phase_payload = _mapping(evaluations.get(phase), field=f"evaluations.{phase}")
    return _mapping(phase_payload.get(mode), field=f"evaluations.{phase}.{mode}")


def _evaluation_contexts(
    evaluation: Mapping[str, object],
    *,
    label: str,
) -> dict[str, Mapping[str, object]]:
    contexts: dict[str, Mapping[str, object]] = {
        "broad_default": _mapping(evaluation.get("broad"), field=f"{label}.broad")
    }
    fixtures = evaluation.get("fixtures")
    if not isinstance(fixtures, list):
        raise RecurrentPairedAnalysisError(f"{label}.fixtures must be a list")
    for index, fixture in enumerate(fixtures):
        parsed = _mapping(fixture, field=f"{label}.fixtures[{index}]")
        name = parsed.get("fixture")
        if not isinstance(name, str) or not name:
            raise RecurrentPairedAnalysisError(f"{label} fixture name is invalid")
        context = f"fixture:{name}"
        if context in contexts:
            raise RecurrentPairedAnalysisError(f"{label} fixture is duplicated")
        contexts[context] = parsed
    return contexts


def _policy_runs(
    policies: Mapping[str, object],
    *,
    policy: str,
    field: str,
) -> tuple[Mapping[str, object], ...]:
    payload = _mapping(policies.get(policy), field=f"{field}.{policy}")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise RecurrentPairedAnalysisError(f"{field}.{policy}.runs must be a list")
    return tuple(
        _mapping(run, field=f"{field}.{policy}.runs[{index}]")
        for index, run in enumerate(runs)
    )


def _run_map(
    policies: Mapping[str, object],
    *,
    field: str,
) -> dict[RunIdentity, Mapping[str, object]]:
    output: dict[RunIdentity, Mapping[str, object]] = {}
    for run in _policy_runs(policies, policy="public_recurrent", field=field):
        identity = _run_identity(run, field=field)
        if identity in output:
            raise RecurrentPairedAnalysisError(
                f"{field} candidate run identity is duplicated"
            )
        output[identity] = run
    return output


def _run_identity(value: Mapping[str, object], *, field: str) -> RunIdentity:
    context = value.get("context")
    if not isinstance(context, str) or not context:
        raise RecurrentPairedAnalysisError(f"{field}.context is invalid")
    seed = _positive_int(value.get("seed"), field=f"{field}.seed")
    policy_seed = value.get("policy_sampling_seed")
    if policy_seed is not None:
        policy_seed = _positive_int(policy_seed, field=f"{field}.policy_sampling_seed")
    return context, seed, policy_seed


def _derive_training_policy_sampling_seed(identity: str) -> int:
    material = f"{TRAINING_POLICY_SAMPLING_NAMESPACE}|{identity}".encode("utf-8")
    digest_value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    span = _MAX_TRAINING_POLICY_SAMPLING_SEED - _MIN_TRAINING_POLICY_SAMPLING_SEED + 1
    return _MIN_TRAINING_POLICY_SAMPLING_SEED + digest_value % span


def _derive_evaluation_policy_sampling_seed(
    *,
    stream_id: str,
    replicate_index: int,
) -> int:
    if replicate_index < 0:
        raise RecurrentPairedAnalysisError(
            "evaluation sampling replicate index must be nonnegative"
        )
    material = (
        f"{EVALUATION_POLICY_SAMPLING_NAMESPACE}|{stream_id}|"
        f"replicate-{replicate_index:04d}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (2**63 - 1)


def _validate_learned_distribution_summary(
    distribution: Mapping[str, object],
    *,
    policy_decision_count: int,
    label: str,
) -> None:
    decision_count = _nonnegative_int(
        distribution.get("decision_count"),
        field=f"{label}.learned_distribution.decision_count",
    )
    if decision_count != policy_decision_count:
        raise RecurrentPairedAnalysisError(
            f"{label} learned-distribution decisions differ from policy decisions"
        )
    means = _mapping(
        distribution.get("metric_means"),
        field=f"{label}.learned_distribution.metric_means",
    )
    if not means:
        raise RecurrentPairedAnalysisError(
            f"{label} learned-distribution metric means are empty"
        )
    for metric, value in means.items():
        if not isinstance(metric, str) or not metric or value is None:
            raise RecurrentPairedAnalysisError(
                f"{label} learned-distribution metric is incomplete"
            )
        parsed = _finite_number(
            value,
            field=f"{label}.learned_distribution.metric_means.{metric}",
        )
        if metric != "entropy" and not 0.0 <= parsed <= 1.0 + 1.0e-9:
            raise RecurrentPairedAnalysisError(
                f"{label} learned-distribution metric {metric} is outside [0, 1]"
            )
    probabilities = _mapping(
        distribution.get("mean_action_probabilities"),
        field=f"{label}.learned_distribution.mean_action_probabilities",
    )
    if set(probabilities) != set(ACTION_NAMES):
        raise RecurrentPairedAnalysisError(
            f"{label} learned action-probability vocabulary drifted"
        )
    parsed_probabilities = [
        _finite_number(
            probabilities[action],
            field=f"{label}.learned_probability.{action}",
        )
        for action in ACTION_NAMES
    ]
    if any(not 0.0 <= value <= 1.0 for value in parsed_probabilities):
        raise RecurrentPairedAnalysisError(
            f"{label} learned action probability is outside [0, 1]"
        )
    if not math.isclose(
        math.fsum(parsed_probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=2.0e-3,
    ):
        raise RecurrentPairedAnalysisError(
            f"{label} learned mean action probabilities do not sum to one"
        )


def _validate_absent_control_distribution(
    distribution: Mapping[str, object],
    *,
    label: str,
) -> None:
    if distribution.get("decision_count") != 0:
        raise RecurrentPairedAnalysisError(
            f"{label} noncandidate control unexpectedly has learned decisions"
        )
    for field in ("metric_means", "metric_minima", "metric_maxima"):
        values = _mapping(
            distribution.get(field),
            field=f"{label}.learned_distribution.{field}",
        )
        if any(value is not None for value in values.values()):
            raise RecurrentPairedAnalysisError(
                f"{label} control unexpectedly has learned-distribution metrics"
            )
    probabilities = _mapping(
        distribution.get("mean_action_probabilities"),
        field=f"{label}.learned_distribution.mean_action_probabilities",
    )
    if set(probabilities) != set(ACTION_NAMES) or any(
        value is not None for value in probabilities.values()
    ):
        raise RecurrentPairedAnalysisError(
            f"{label} control unexpectedly has learned action probabilities"
        )


def _seed_vector(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RecurrentPairedAnalysisError(f"{field} must be a nonempty seed list")
    seeds = tuple(_positive_int(seed, field=field) for seed in value)
    if len(set(seeds)) != len(seeds):
        raise RecurrentPairedAnalysisError(f"{field} contains duplicate seeds")
    return seeds


def _sampling_seed_vector(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RecurrentPairedAnalysisError(f"{field} must be a seed list")
    seeds = tuple(_positive_int(seed, field=field) for seed in value)
    if len(set(seeds)) != len(seeds):
        raise RecurrentPairedAnalysisError(f"{field} contains duplicate seeds")
    return seeds


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentPairedAnalysisError(f"{field} must be a mapping")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentPairedAnalysisError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentPairedAnalysisError(f"{field} must be a nonnegative integer")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentPairedAnalysisError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentPairedAnalysisError(f"{field} must be finite")
    return parsed


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecurrentPairedAnalysisError(f"{field} must be lowercase SHA256")
    return value


def _canonical_clone(value: Mapping[str, object], *, field: str) -> dict[str, object]:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RecurrentPairedAnalysisError(f"{field} must be finite JSON data") from exc
    if not isinstance(decoded, dict):
        raise RecurrentPairedAnalysisError(f"{field} root must be a mapping")
    return decoded


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise RecurrentPairedAnalysisError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _reject_nonfinite_json_constant(value: str) -> object:
    raise RecurrentPairedAnalysisError(f"non-finite JSON constant: {value}")


def _identity_sort_key(identity: RunIdentity) -> tuple[str, int, int]:
    return identity[0], identity[1], -1 if identity[2] is None else identity[2]


def _sort_value(value: object) -> tuple[int, object]:
    return (0, -1) if value is None else (1, value)


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise RecurrentPairedAnalysisError("analysis produced a non-finite value")
    rounded = round(value, 12)
    return 0.0 if rounded == 0.0 else rounded


__all__ = [
    "PAIRED_ANALYSIS_SCHEMA_VERSION",
    "RecurrentPairedAnalysisError",
    "analyze_recurrent_causal_canary_pair",
]
