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
from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
    RECURRENT_COUNTERFACTUAL_AUXILIARY_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED,
    RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS,
    RecurrentCounterfactualAuxiliaryError,
    recurrent_counterfactual_auxiliary_bundle_digest,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RecurrentCounterfactualCollectionBundle,
    RecurrentCounterfactualCollectionConfig,
    recurrent_counterfactual_collection_config_payload,
    RecurrentCounterfactualCollectionError,
    RecurrentCounterfactualCollectionResult,
    RecurrentCounterfactualCollectionTask,
    validate_recurrent_counterfactual_collection_result,
)
from evolution_sim.mind.recurrent_evaluation import (
    RECURRENT_EVALUATION_SCHEMA_VERSION,
    RecurrentEvaluationError,
    validate_recurrent_evaluation_report,
)
from evolution_sim.mind.recurrent_experiment import (
    RECURRENT_EXPERIMENT_CONTRACT_VERSION,
)
from evolution_sim.mind.recurrent_paired_analysis import (
    CAUSAL_CANARY_POLICY_VERSION_V4,
    CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION_V4,
    CAUSAL_CANARY_SCHEMA_VERSION_V4,
)
from evolution_sim.mind.recurrent_policy import (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
)
from evolution_sim.mind.recurrent_rollout import (
    derive_recurrent_policy_sampling_seed,
)


RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_counterfactual_three_arm_comparison_v3"
)
RECURRENT_COUNTERFACTUAL_ANALYZER_PROVENANCE_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_counterfactual_analyzer_provenance_v1"
)
# Compatibility names for callers of this analyzer.  The values are sourced from
# the public paired-analysis contract rather than copied as independent literals.
CAUSAL_CANARY_SCHEMA_VERSION = CAUSAL_CANARY_SCHEMA_VERSION_V4
CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION = (
    CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION_V4
)
CAUSAL_CANARY_POLICY_VERSION = CAUSAL_CANARY_POLICY_VERSION_V4
CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_causal_canary_runtime_reproducibility_v1"
)
UNPINNED_LABEL_PREFIX = "unpinned-noncandidate-development-canary:"

BASE_ARM = "base_recurrent_ppo"
EXACT_ARM = "exact_counterfactual_auxiliary"
SHUFFLED_ARM = "deterministic_valid_action_label_shuffled_auxiliary"
ARM_ORDER = (BASE_ARM, EXACT_ARM, SHUFFLED_ARM)
SELECTION_MODES = (
    PUBLIC_RECURRENT_ARGMAX_SELECTION,
    PUBLIC_RECURRENT_SAMPLED_SELECTION,
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
_TOP_LEVEL_CLOSED_FLAGS = (
    "campaign_training_slice_consumed",
    "runtime_artifact_created",
    "runtime_action_selection_changed",
    "promotion_authorized",
    "validation_seeds_accessed",
    "lockbox_seeds_accessed",
)
_CAUSAL_METRICS = (
    "terminal_alive",
    "births",
    "deaths",
    "reward_total",
    "dominant_requested_action_share",
    "unsupported_requested_action_count",
    "heuristic_action_source_count",
)
_OUTCOME_METRICS = (
    "terminal_nonextinct",
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
    "learned_normalized_entropy_mean",
    "learned_eat_probability_mean",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_RE = re.compile(r"^[0-9a-f]{40,64}$")
_EVALUATION_POLICY_SAMPLING_NAMESPACE = (
    "evolution-sim|mind-v3-public-recurrent-evaluation|artifact-sampling-v1"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_SOURCE_MANIFEST_PATH_CONTRACT = (
    "repository_relative_sorted_runtime_python_plus_package"
)
_ANALYZER_RUNTIME_PATH = (
    "python/evolution_sim/mind/recurrent_counterfactual_comparison.py"
)
_FLOAT32_KL_ROUNDOFF_ABSOLUTE_TOLERANCE = 1.0e-6
_KL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "policy_improvement_kl",
        "behavior_kl",
        "mean_behavior_kl_old_to_post",
        "max_state_behavior_kl_old_to_post",
    }
)
_CANARY_MODEL_FINGERPRINT_HASH_CONTRACT = (
    "sorted_state_dict_name_dtype_shape_and_raw_cpu_bytes_v1"
)
_AUXILIARY_MODEL_STATE_HASH_CONTRACT = (
    "recurrent_policy.recurrent_model_state_sha256_length_prefixed_v1"
)


ReportSource: TypeAlias = Mapping[str, object] | str | Path
RunIdentity: TypeAlias = tuple[str, int, int | None]


class RecurrentCounterfactualComparisonError(ValueError):
    """Raised when three-arm causal evidence cannot be matched exactly."""


def analyze_recurrent_counterfactual_three_arm(
    base_report: ReportSource,
    exact_report: ReportSource,
    shuffled_report: ReportSource,
) -> dict[str, object]:
    """Validate and descriptively compare a preregistered three-arm canary.

    Environment seeds and policy-sampling streams are crossed repeated
    measurements within a learner run.  Consequently this report deliberately
    emits raw matched cells and two transparent stratifications, but no
    p-values, independence claim, or learner-level confidence interval.
    """

    reports = {
        BASE_ARM: _load_and_validate_report(base_report, arm=BASE_ARM),
        EXACT_ARM: _load_and_validate_report(exact_report, arm=EXACT_ARM),
        SHUFFLED_ARM: _load_and_validate_report(
            shuffled_report,
            arm=SHUFFLED_ARM,
        ),
    }
    matched = _validate_three_arm_match(reports)
    base_source = _mapping(reports[BASE_ARM].get("source"), field="base.source")
    training_source_manifest = _mapping(
        base_source.get("source_file_hash_manifest"),
        field="base.source.source_file_hash_manifest",
    )
    analyzer_provenance = _build_analyzer_provenance(training_source_manifest)

    selection_modes: dict[str, object] = {}
    for mode in SELECTION_MODES:
        effects = {
            arm: _arm_training_effects(report, mode=mode)
            for arm, report in reports.items()
        }
        selection_modes[mode] = {
            "run_identity_count": len(effects[BASE_ARM]),
            "exact_minus_base": _descriptive_contrast(
                effects[EXACT_ARM],
                effects[BASE_ARM],
                estimand=("(exact_after - exact_before) - (base_after - base_before)"),
                left_arm=EXACT_ARM,
                right_arm=BASE_ARM,
            ),
            "exact_minus_shuffled": _descriptive_contrast(
                effects[EXACT_ARM],
                effects[SHUFFLED_ARM],
                estimand=(
                    "(exact_after - exact_before) - "
                    "(label_shuffled_after - label_shuffled_before)"
                ),
                left_arm=EXACT_ARM,
                right_arm=SHUFFLED_ARM,
            ),
        }

    report: dict[str, object] = {
        "schema_version": RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION,
        "analysis_kind": ("matched_descriptive_three_arm_counterfactual_causal_canary"),
        "analyzer_provenance": analyzer_provenance,
        "evidence": {
            "base_report_exact_digest": reports[BASE_ARM]["exact_digest"],
            "exact_report_exact_digest": reports[EXACT_ARM]["exact_digest"],
            "shuffled_report_exact_digest": reports[SHUFFLED_ARM]["exact_digest"],
            "git_head_observed": matched["git_head_observed"],
            "source_manifest_aggregate_sha256": matched[
                "source_manifest_aggregate_sha256"
            ],
            "training_schedule_sha256": matched["training_schedule_sha256"],
            "before_model_state_sha256": matched["before_model_state_sha256"],
            "evaluation_sampling_stream_id": matched["evaluation_sampling_stream_id"],
            "label_shuffle_permutation_seed": matched["label_shuffle_permutation_seed"],
            "all_top_level_exact_digests_verified": True,
            "all_nested_v5_evaluations_publicly_validated": True,
            "counterfactual_collection_digests_verified": True,
            "counterfactual_bundle_one_use_semantics_verified": True,
            "source_head_manifest_and_initialization_exactly_matched": True,
            "schedule_and_evaluation_streams_exactly_matched": True,
            "base_before_evaluations_semantically_identical_across_arms": True,
            "validation_and_lockbox_remained_closed": True,
            "comparison_analyzer_provenance_verified": True,
            "non_analyzer_runtime_source_drift_rejected": True,
        },
        "comparison_contract": {
            "arms_in_order": list(ARM_ORDER),
            "base_counterfactual_auxiliary_enabled": False,
            "exact_branch_auxiliary_enabled": True,
            "label_shuffled_auxiliary_enabled": True,
            "label_shuffle_domain": "currently_valid_actions_only",
            "only_permitted_executed_configuration_difference": (
                "preregistered_counterfactual_arm_and_valid_action_label_permutation"
            ),
            "inactive_base_counterfactual_knobs_executed": False,
            "inactive_base_counterfactual_knobs_required_to_match_active_arms": False,
            "exact_and_shuffled_active_counterfactual_configuration_matched": True,
            "raw_analysis_unit": "environment_policy_sampling_stream_run_cell",
            "policy_sampling_streams_are_repeated_measurements": True,
            "environment_seeds_are_repeated_measurements": True,
            "learner_replication_count_per_arm": 1,
            "inferential_claim_authorized": False,
            "p_values_emitted": False,
            "independence_assumption_made": False,
            "interpretation": (
                "descriptive matched effects for one learner initialization; "
                "environment and stream stratifications expose crossed sensitivity"
            ),
            "end_state_digest_contracts": {
                "evaluation_fingerprint": _CANARY_MODEL_FINGERPRINT_HASH_CONTRACT,
                "counterfactual_auxiliary_transaction": (
                    _AUXILIARY_MODEL_STATE_HASH_CONTRACT
                ),
                "digest_values_directly_comparable": False,
                "same_state_cross_digest_link_present_in_input_reports": False,
                "limitation": (
                    "the v4 canary report records the final evaluated model and "
                    "the auxiliary transaction under different SHA256 encoding "
                    "contracts; both are validated independently, but their digest "
                    "strings cannot be equated"
                ),
            },
        },
        "selection_modes": selection_modes,
    }
    report["exact_digest"] = stable_payload_digest(report)
    return report


def write_recurrent_counterfactual_three_arm_comparison(
    base_report: ReportSource,
    exact_report: ReportSource,
    shuffled_report: ReportSource,
    *,
    output_path: str | Path,
) -> dict[str, object]:
    """Validate three causal reports and exclusively create one sealed report.

    The destination must not already exist.  Refusing replacement makes a CLI
    invocation auditable and prevents accidentally overwriting prior evidence.
    """

    output = Path(output_path)
    if output.suffix.lower() != ".json":
        raise RecurrentCounterfactualComparisonError(
            "comparison output path must have a .json suffix"
        )
    if not output.parent.is_dir():
        raise RecurrentCounterfactualComparisonError(
            f"comparison output parent does not exist: {output.parent}"
        )
    comparison = analyze_recurrent_counterfactual_three_arm(
        base_report,
        exact_report,
        shuffled_report,
    )
    encoded = (
        json.dumps(
            comparison,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise RecurrentCounterfactualComparisonError(
            f"comparison output already exists: {output}"
        ) from error
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for a sealed, read-only three-arm comparison.

    Example::

        python -m evolution_sim.mind.recurrent_counterfactual_comparison \
          --base-report output/base.json \
          --exact-report output/exact.json \
          --shuffled-report output/shuffled.json \
          --output output/counterfactual-three-arm-comparison.json
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate matched recurrent causal-canary v4 evidence and write a "
            "sealed descriptive three-arm comparison."
        )
    )
    parser.add_argument("--base-report", required=True, type=Path)
    parser.add_argument("--exact-report", required=True, type=Path)
    parser.add_argument("--shuffled-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        comparison = write_recurrent_counterfactual_three_arm_comparison(
            arguments.base_report,
            arguments.exact_report,
            arguments.shuffled_report,
            output_path=arguments.output,
        )
    except RecurrentCounterfactualComparisonError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "exact_digest": comparison["exact_digest"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_and_validate_report(
    source: ReportSource,
    *,
    arm: str,
) -> dict[str, object]:
    if isinstance(source, Mapping):
        report = _canonical_clone(source, field=f"{arm} report")
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() != ".json":
            raise RecurrentCounterfactualComparisonError(
                f"{arm} report path must have a .json suffix"
            )
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as error:
            raise RecurrentCounterfactualComparisonError(
                f"cannot read {arm} report: {path}"
            ) from error
        try:
            loaded = json.loads(
                raw,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, RecurrentCounterfactualComparisonError) as error:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} report is not canonical finite JSON"
            ) from error
        if not isinstance(loaded, dict):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} report root must be a mapping"
            )
        report = loaded
    else:
        raise TypeError(f"{arm} report must be a mapping or JSON path")

    observed = _sha256(report.get("exact_digest"), field=f"{arm}.exact_digest")
    payload = copy.deepcopy(report)
    payload.pop("exact_digest")
    if stable_payload_digest(payload) != observed:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} report exact digest does not match its canonical payload"
        )
    _validate_causal_report(report, arm=arm)
    return report


def _validate_causal_report(report: Mapping[str, object], *, arm: str) -> None:
    if report.get("schema_version") != CAUSAL_CANARY_SCHEMA_VERSION:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} causal canary schema must be v4"
        )
    if report.get("policy") != CAUSAL_CANARY_POLICY_VERSION:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} causal canary policy must be v4"
        )
    for field in (
        "development_run",
        "development_experiment",
        "noncandidate_development_canary",
        "source_unpinned",
        "artifact_output_refused_by_contract",
        "non_promoted",
    ):
        if report.get(field) is not True:
            raise RecurrentCounterfactualComparisonError(f"{arm}.{field} must be true")
    for field in ("source_pinned", "artifact_output_requested"):
        if report.get(field) is not False:
            raise RecurrentCounterfactualComparisonError(f"{arm}.{field} must be false")
    if report.get("artifact_path") is not None:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} development canary cannot name an artifact"
        )

    _validate_closed_lifecycle(report, arm=arm)
    _validate_source(report, arm=arm)
    counterfactual = _validate_preregistration(report, arm=arm)
    _validate_model_states(report, arm=arm)
    _validate_training(report, arm=arm, preregistered_counterfactual=counterfactual)
    _validate_evaluations(report, arm=arm)
    _validate_report_paired_deltas(report, arm=arm)


def _validate_closed_lifecycle(report: Mapping[str, object], *, arm: str) -> None:
    lifecycle = _mapping(report.get("lifecycle"), field=f"{arm}.lifecycle")
    if set(lifecycle) != set(_LIFECYCLE_FLAGS):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} lifecycle keys differ from the closed contract"
        )
    open_flags = sorted(key for key, value in lifecycle.items() if value is not False)
    if open_flags:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} lifecycle is open: {open_flags!r}"
        )
    for field in _TOP_LEVEL_CLOSED_FLAGS:
        if report.get(field) is not False:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} top-level lifecycle flag {field} must be false"
            )
    roles = _mapping(report.get("seed_roles"), field=f"{arm}.seed_roles")
    for role in ("validation", "lockbox"):
        value = _mapping(roles.get(role), field=f"{arm}.seed_roles.{role}")
        if (
            value.get("accessed") is not False
            or value.get("seeds_materialized_by_canary") is not False
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} {role} seed role was accessed or materialized"
            )


def _validate_source(report: Mapping[str, object], *, arm: str) -> None:
    source = _mapping(report.get("source"), field=f"{arm}.source")
    source_tree_dirty = _exact_bool(
        source.get("source_tree_dirty"),
        field=f"{arm}.source.source_tree_dirty",
    )
    if (
        source.get("source_pinned") is not False
        or source.get("unpinned") is not True
        or source.get("noncandidate") is not True
        or source.get("source_stable_during_run") is not True
        or source.get("git_state_available") is not True
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} source provenance is not stable, available, and unpinned"
        )
    if report.get("source_dirty") is not source_tree_dirty:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} top-level and source-tree dirty flags differ"
        )
    _sha256(
        source.get("git_status_porcelain_sha256"),
        field=f"{arm}.source.git_status_porcelain_sha256",
    )
    _nonnegative_int(
        source.get("git_status_entry_count"),
        field=f"{arm}.source.git_status_entry_count",
    )
    head = source.get("git_head_observed")
    if not isinstance(head, str) or _GIT_HEAD_RE.fullmatch(head) is None:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} source git head is invalid"
        )
    manifest = _mapping(
        source.get("source_file_hash_manifest"),
        field=f"{arm}.source.source_file_hash_manifest",
    )
    if (
        manifest.get("hash_algorithm") != "sha256"
        or manifest.get("path_contract") != _RUNTIME_SOURCE_MANIFEST_PATH_CONTRACT
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} source manifest contract drifted"
        )
    files = _mapping(manifest.get("files"), field=f"{arm}.source.manifest.files")
    if not files:
        raise RecurrentCounterfactualComparisonError(f"{arm} source manifest is empty")
    for path, digest in files.items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} source manifest path is invalid"
            )
        _sha256(digest, field=f"{arm}.source.manifest.files[{path!r}]")
    if _nonnegative_int(
        manifest.get("file_count"),
        field=f"{arm}.source.manifest.file_count",
    ) != len(files):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} source manifest file count is inconsistent"
        )
    aggregate = _sha256(
        manifest.get("aggregate_sha256"),
        field=f"{arm}.source.manifest.aggregate_sha256",
    )
    if stable_payload_digest(dict(files)) != aggregate:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} source manifest aggregate is invalid"
        )

    runtime = _mapping(
        report.get("runtime_reproducibility"),
        field=f"{arm}.runtime_reproducibility",
    )
    if (
        runtime.get("schema_version") != CAUSAL_CANARY_RUNTIME_SCHEMA_VERSION
        or runtime.get("captured_before_model_initialization") is not True
        or type(runtime.get("torch_deterministic_algorithms_enabled_after_runner"))
        is not bool
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} runtime reproducibility contract drifted"
        )


def _current_runtime_source_manifest() -> dict[str, object]:
    """Hash the runtime source surface under the canary's manifest contract."""

    source_root = _REPOSITORY_ROOT / "python" / "evolution_sim"
    paths = sorted(source_root.rglob("*.py"))
    package_path = _REPOSITORY_ROOT / "package.json"
    if package_path.is_file():
        paths.append(package_path)
    files = {
        path.relative_to(_REPOSITORY_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
        if path.is_file()
    }
    if not files:
        raise RecurrentCounterfactualComparisonError(
            "current runtime source manifest found no source files"
        )
    return {
        "hash_algorithm": "sha256",
        "path_contract": _RUNTIME_SOURCE_MANIFEST_PATH_CONTRACT,
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": stable_payload_digest(files),
    }


def _build_analyzer_provenance(
    training_source_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Bind analysis to current code while allowing only this analyzer to drift.

    A report produced before a comparison-analyzer repair must not pretend that
    the repaired analyzer was part of training.  Conversely, permitting an
    arbitrary current checkout would make validation depend on unrecorded code.
    The sole accepted post-training runtime-source difference is therefore this
    module itself; every other added, removed, or changed runtime path fails
    closed.
    """

    current_manifest = _current_runtime_source_manifest()
    training_files = _mapping(
        training_source_manifest.get("files"),
        field="training_source_manifest.files",
    )
    current_files = _mapping(
        current_manifest.get("files"),
        field="current_runtime_source_manifest.files",
    )
    if _ANALYZER_RUNTIME_PATH not in training_files:
        raise RecurrentCounterfactualComparisonError(
            "training source manifest does not contain the comparison analyzer"
        )
    if _ANALYZER_RUNTIME_PATH not in current_files:
        raise RecurrentCounterfactualComparisonError(
            "current source manifest does not contain the comparison analyzer"
        )

    changed_paths = sorted(
        path
        for path in set(training_files) | set(current_files)
        if training_files.get(path) != current_files.get(path)
    )
    forbidden_paths = [path for path in changed_paths if path != _ANALYZER_RUNTIME_PATH]
    if forbidden_paths:
        raise RecurrentCounterfactualComparisonError(
            "current runtime source drift extends beyond the comparison analyzer: "
            f"{forbidden_paths!r}"
        )

    training_module_sha256 = _sha256(
        training_files.get(_ANALYZER_RUNTIME_PATH),
        field="training comparison analyzer SHA256",
    )
    current_module_sha256 = _sha256(
        current_files.get(_ANALYZER_RUNTIME_PATH),
        field="current comparison analyzer SHA256",
    )
    analyzer_changed = changed_paths == [_ANALYZER_RUNTIME_PATH]
    source_exactly_matched = not changed_paths
    if not (analyzer_changed or source_exactly_matched):
        raise RecurrentCounterfactualComparisonError(
            "current runtime source drift is not an exact or analyzer-only match"
        )

    return {
        "schema_version": (RECURRENT_COUNTERFACTUAL_ANALYZER_PROVENANCE_SCHEMA_VERSION),
        "comparison_schema_version": (
            RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION
        ),
        "module_path": _ANALYZER_RUNTIME_PATH,
        "training_module_sha256": training_module_sha256,
        "current_module_sha256": current_module_sha256,
        "training_runtime_source_manifest_aggregate_sha256": _sha256(
            training_source_manifest.get("aggregate_sha256"),
            field="training source manifest aggregate",
        ),
        "current_runtime_source_manifest_aggregate_sha256": _sha256(
            current_manifest.get("aggregate_sha256"),
            field="current source manifest aggregate",
        ),
        "current_runtime_source_manifest_file_count": _nonnegative_int(
            current_manifest.get("file_count"),
            field="current source manifest file count",
        ),
        "runtime_source_manifest_path_contract": (
            _RUNTIME_SOURCE_MANIFEST_PATH_CONTRACT
        ),
        "runtime_source_changed_paths": changed_paths,
        "runtime_source_change_count": len(changed_paths),
        "runtime_source_exactly_matches_training": source_exactly_matched,
        "only_analyzer_module_changed_since_training": analyzer_changed,
        "non_analyzer_runtime_source_drift_detected": False,
        "post_training_analyzer_repair_transparent": analyzer_changed,
    }


def _validate_preregistration(
    report: Mapping[str, object],
    *,
    arm: str,
) -> Mapping[str, object]:
    prereg = _mapping(
        report.get("evaluation_preregistration"),
        field=f"{arm}.evaluation_preregistration",
    )
    if (
        prereg.get("schema_version") != CAUSAL_CANARY_PREREGISTRATION_SCHEMA_VERSION
        or prereg.get("created_before_model_initialization") is not True
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} causal preregistration must be v4 and pre-initialization"
        )
    exact = _sha256(
        prereg.get("exact_digest"),
        field=f"{arm}.evaluation_preregistration.exact_digest",
    )
    payload = copy.deepcopy(dict(prereg))
    payload.pop("exact_digest")
    label = payload.pop("synthetic_noncandidate_digest_label", None)
    if stable_payload_digest(payload) != exact:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} preregistration exact digest is invalid"
        )
    if label != UNPINNED_LABEL_PREFIX + exact:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} preregistration synthetic label is invalid"
        )
    if tuple(prereg.get("action_selection_modes_in_order", ())) != SELECTION_MODES:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} preregistration action-selection order drifted"
        )
    if prereg.get("before_after_sampling_stream_identical") is not True or prereg.get(
        "candidate_controls"
    ) != ["mind_v3_linear", "masked_random"]:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} preregistration pairing contract drifted"
        )
    configuration = _mapping(
        report.get("configuration"),
        field=f"{arm}.configuration",
    )
    if prereg.get("run_config") != configuration:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} configuration differs from its preregistration"
        )
    if prereg.get("seed_registry_sha256") != report.get(
        "seed_registry_sha256"
    ) or prereg.get("seed_roles") != report.get("seed_roles"):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} preregistered seed evidence drifted"
        )
    _sha256(report.get("seed_registry_sha256"), field=f"{arm}.seed_registry_sha256")
    source = _mapping(report.get("source"), field=f"{arm}.source")
    manifest = _mapping(
        source.get("source_file_hash_manifest"),
        field=f"{arm}.source.manifest",
    )
    if prereg.get("source_manifest_aggregate_sha256") != manifest.get(
        "aggregate_sha256"
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} preregistered source manifest drifted"
        )
    counterfactual = _mapping(
        configuration.get("counterfactual_auxiliary"),
        field=f"{arm}.configuration.counterfactual_auxiliary",
    )
    _validate_counterfactual_preregistration(counterfactual, arm=arm)
    return counterfactual


def _validate_counterfactual_preregistration(
    value: Mapping[str, object],
    *,
    arm: str,
) -> None:
    if (
        value.get("default_off") is not True
        or value.get("configuration_resolved_before_model_initialization") is not True
        or value.get("feed_forward_history_ablation_compatible") is not False
        or value.get("runtime_artifact_created") is not False
        or value.get("runtime_action_selection_changed") is not False
        or value.get("promotion_authorized") is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual preregistration closed contract drifted"
        )
    resolved = _mapping(
        value.get("resolved_configuration"),
        field=f"{arm}.counterfactual.resolved_configuration",
    )
    collection = _mapping(
        resolved.get("collection"),
        field=f"{arm}.counterfactual.collection",
    )
    horizons = _positive_int_sequence(
        collection.get("horizons"),
        field=f"{arm}.counterfactual.collection.horizons",
    )
    if tuple(sorted(horizons)) != horizons:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual horizons must be strictly increasing"
        )
    gamma = _finite_number(
        collection.get("gamma"),
        field=f"{arm}.counterfactual.collection.gamma",
    )
    if not 0.0 < gamma <= 1.0:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual collection gamma must be in (0, 1]"
        )
    auxiliary = _mapping(
        resolved.get("auxiliary"),
        field=f"{arm}.counterfactual.auxiliary",
    )
    if (
        auxiliary.get("schema_version")
        != RECURRENT_COUNTERFACTUAL_AUXILIARY_SCHEMA_VERSION
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual auxiliary schema drifted"
        )
    if (
        auxiliary.get("advantage_baseline")
        != "source_behavior_probability_weighted_action_value"
        or auxiliary.get("soft_target")
        != "normalize(pi_old * exp(clip(advantage) / temperature))"
        or auxiliary.get("policy_loss")
        != "forward_kl(soft_improvement_target || current_actor)"
        or auxiliary.get("behavior_regularizer")
        != "forward_kl(pi_old || current_actor)"
        or auxiliary.get(
            "metadata_seed_fixture_private_or_provenance_used_as_model_input"
        )
        is not False
        or auxiliary.get("outcome_labels_used_as_model_input") is not False
        or auxiliary.get("stored_source_hidden_used_as_model_input") is not False
        or auxiliary.get("current_hidden_state_policy")
        != "reconstruct_from_public_history_with_exact_current_model"
        or auxiliary.get("ppo_ratio_data_use") is not False
        or auxiliary.get("optimizer_step_performed") is not False
        or auxiliary.get("runtime_integrated") is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual public-input/soft-target contract drifted"
        )
    for field, strictly_positive in (
        ("temperature", True),
        ("advantage_clip", True),
        ("policy_improvement_coefficient", True),
        ("behavior_kl_coefficient", False),
        ("value_loss_coefficient", False),
    ):
        parsed = _finite_number(
            auxiliary.get(field),
            field=f"{arm}.counterfactual.auxiliary.{field}",
        )
        if parsed < 0.0 or (strictly_positive and parsed <= 0.0):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} counterfactual auxiliary {field} is invalid"
            )
    scalarization = _mapping(
        auxiliary.get("scalarization"),
        field=f"{arm}.counterfactual.auxiliary.scalarization",
    )
    horizon_weights = _sequence(
        scalarization.get("horizon_weights"),
        field=f"{arm}.counterfactual.horizon_weights",
    )
    parsed_horizon_weights = tuple(
        (
            _positive_int(
                _mapping(item, field=f"{arm}.horizon_weight").get("horizon_ticks"),
                field=f"{arm}.horizon_weight.horizon_ticks",
            ),
            _finite_number(
                _mapping(item, field=f"{arm}.horizon_weight").get("weight"),
                field=f"{arm}.horizon_weight.weight",
            ),
        )
        for item in horizon_weights
    )
    if tuple(item[0] for item in parsed_horizon_weights) != horizons:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} collection and scalarization horizons differ"
        )
    if any(
        weight <= 0.0 for _horizon, weight in parsed_horizon_weights
    ) or not math.isclose(
        math.fsum(weight for _horizon, weight in parsed_horizon_weights),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} scalarization horizon weights are invalid"
        )
    outcome_weights = _mapping(
        scalarization.get("outcome_weights"),
        field=f"{arm}.counterfactual.outcome_weights",
    )
    parsed_outcome_weights = tuple(
        _finite_number(weight, field=f"{arm}.counterfactual.outcome_weight")
        for weight in outcome_weights.values()
    )
    if not parsed_outcome_weights or not any(
        weight != 0.0 for weight in parsed_outcome_weights
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} scalarization outcome weights are all zero"
        )
    step = _mapping(
        resolved.get("step"),
        field=f"{arm}.counterfactual.step",
    )
    if (
        step.get("schema_version")
        != RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION
        or step.get("optimizer") != "shared_ppo_adam"
        or step.get("optimizer_steps") != 1
        or step.get("backward_passes") != 1
        or step.get("retry_on_rejection_or_error") is not False
        or step.get("runtime_integrated") is not False
        or step.get("promotion_authorized") is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual one-step transaction contract drifted"
        )
    if (
        _finite_number(
            step.get("learning_rate_multiplier"),
            field=f"{arm}.counterfactual.step.learning_rate_multiplier",
        )
        <= 0.0
        or _finite_number(
            step.get("max_gradient_norm"),
            field=f"{arm}.counterfactual.step.max_gradient_norm",
        )
        <= 0.0
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual step learning-rate or gradient limit is invalid"
        )
    audit = _mapping(
        step.get("post_step_public_policy_audit"),
        field=f"{arm}.counterfactual.step.post_step_public_policy_audit",
    )
    if any(
        _finite_number(audit.get(field), field=f"{arm}.counterfactual.step.{field}")
        < 0.0
        for field in (
            "mean_forward_kl_pi_old_to_pi_post_limit",
            "max_state_forward_kl_pi_old_to_pi_post_limit",
        )
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual step KL limits are invalid"
        )
    _positive_int(
        resolved.get("bundles_per_update"),
        field=f"{arm}.counterfactual.bundles_per_update",
    )
    branch_ticks = _nonnegative_int_sequence(
        resolved.get("branch_tick_candidates"),
        field=f"{arm}.counterfactual.branch_tick_candidates",
    )
    if tuple(sorted(branch_ticks)) != branch_ticks:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual branch ticks must be strictly increasing"
        )
    _positive_int(resolved.get("workers"), field=f"{arm}.counterfactual.workers")

    negative = _mapping(
        value.get("scientific_negative_control"),
        field=f"{arm}.counterfactual.scientific_negative_control",
    )
    auxiliary_negative = _mapping(
        auxiliary.get("scientific_negative_control"),
        field=f"{arm}.counterfactual.auxiliary.scientific_negative_control",
    )
    if arm == BASE_ARM:
        expected = (False, "disabled", False, False)
        expected_permutation = RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
    elif arm == EXACT_ARM:
        expected = (True, "exact_counterfactual_labels", True, False)
        expected_permutation = RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_DISABLED
    else:
        expected = (
            True,
            "deterministic_valid_action_label_shuffled_control",
            False,
            True,
        )
        expected_permutation = RECURRENT_COUNTERFACTUAL_TARGET_PERMUTATION_VALID_ACTIONS
    observed = (
        value.get("enabled"),
        value.get("mode"),
        value.get("exact_branch_labels_consumed"),
        negative.get("enabled"),
    )
    if observed != expected:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual arm role is inconsistent"
        )
    if value.get("one_auxiliary_step_per_ppo_update") is not expected[0]:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} one-step-per-update declaration drifted"
        )
    if (
        negative.get("mode") != expected_permutation
        or auxiliary_negative.get("mode") != expected_permutation
        or negative.get("permutation_domain") != "currently_valid_actions_only"
        or auxiliary_negative.get("applied_after_exact_row_validation") is not True
        or auxiliary_negative.get("branch_rows_mutated") is not False
        or auxiliary_negative.get("exact_branch_labels_claimed_after_permutation")
        is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} valid-action permutation contract drifted"
        )
    if arm == SHUFFLED_ARM:
        seed = _nonnegative_int(
            negative.get("permutation_seed"),
            field=f"{arm}.counterfactual.permutation_seed",
        )
        if (
            negative.get("permutation_seed_explicit") is not True
            or auxiliary_negative.get("enabled") is not True
            or auxiliary_negative.get("seed") != seed
        ):
            raise RecurrentCounterfactualComparisonError(
                "label-shuffled arm lacks one explicit matching permutation seed"
            )
    elif (
        negative.get("permutation_seed") is not None
        or negative.get("permutation_seed_explicit") is not False
        or auxiliary_negative.get("enabled") is not False
        or auxiliary_negative.get("seed") is not None
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} unexpectedly declares a permutation seed"
        )


def _validate_model_states(report: Mapping[str, object], *, arm: str) -> None:
    states = _mapping(report.get("model_states"), field=f"{arm}.model_states")
    if (
        states.get("changed") is not True
        or states.get("same_before_snapshot_used_for_both_selection_modes") is not True
        or states.get("same_after_model_used_for_both_selection_modes") is not True
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} model-state integrity contract drifted"
        )
    before = _mapping(states.get("before"), field=f"{arm}.model_states.before")
    after = _mapping(states.get("after"), field=f"{arm}.model_states.after")
    for label, fingerprint in (("before", before), ("after", after)):
        if fingerprint.get("hash_contract") != _CANARY_MODEL_FINGERPRINT_HASH_CONTRACT:
            raise RecurrentCounterfactualComparisonError(
                f"{arm}.model_states.{label} hash contract drifted"
            )
        state_value_count = _positive_int(
            fingerprint.get("state_value_count"),
            field=f"{arm}.model_states.{label}.state_value_count",
        )
        dtype_counts = _mapping(
            fingerprint.get("dtype_value_counts"),
            field=f"{arm}.model_states.{label}.dtype_value_counts",
        )
        if (
            not dtype_counts
            or sum(
                _nonnegative_int(
                    count,
                    field=f"{arm}.model_states.{label}.dtype_value_counts[{dtype!r}]",
                )
                for dtype, count in dtype_counts.items()
            )
            != state_value_count
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{arm}.model_states.{label} dtype counts are inconsistent"
            )
        for count_field in ("state_tensor_count", "parameter_count"):
            _positive_int(
                fingerprint.get(count_field),
                field=f"{arm}.model_states.{label}.{count_field}",
            )
        _nonnegative_int(
            fingerprint.get("trainable_parameter_count"),
            field=f"{arm}.model_states.{label}.trainable_parameter_count",
        )
    before_digest = _sha256(
        before.get("state_sha256"),
        field=f"{arm}.model_states.before.state_sha256",
    )
    after_digest = _sha256(
        after.get("state_sha256"),
        field=f"{arm}.model_states.after.state_sha256",
    )
    if before_digest == after_digest:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} training did not change model state"
        )


def _validate_training(
    report: Mapping[str, object],
    *,
    arm: str,
    preregistered_counterfactual: Mapping[str, object],
) -> None:
    config = _mapping(report.get("configuration"), field=f"{arm}.configuration")
    training = _mapping(report.get("training"), field=f"{arm}.training")
    if training.get("contract_version") != RECURRENT_EXPERIMENT_CONTRACT_VERSION:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} recurrent experiment contract drifted"
        )
    if config.get("model") != training.get("model_config"):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} training model configuration drifted"
        )
    ppo = _mapping(config.get("ppo"), field=f"{arm}.configuration.ppo")
    if ppo != training.get("ppo_config"):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} training PPO configuration drifted"
        )
    learner_seed = _positive_int(
        training.get("learner_seed"),
        field=f"{arm}.training.learner_seed",
    )
    learner_role = _mapping(
        _mapping(report.get("seed_roles"), field=f"{arm}.seed_roles").get("learner"),
        field=f"{arm}.seed_roles.learner",
    )
    if (
        ppo.get("learner_seed") != learner_seed
        or learner_role.get("seed") != learner_seed
        or training.get("deterministic_algorithms_enabled") is not True
        or training.get("device") != config.get("resolved_device")
        or training.get("training_scenarios") != config.get("training_scenarios")
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} learner, determinism, device, or scenario contract drifted"
        )
    if (
        config.get("feed_forward_history_ablation") is not False
        or ppo.get("feed_forward_history_ablation") is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} must be a recurrent-history arm, not a feed-forward ablation"
        )

    schedule = _sequence(
        config.get("training_schedule"),
        field=f"{arm}.configuration.training_schedule",
    )
    if not schedule or stable_payload_digest(schedule) != config.get(
        "training_schedule_sha256"
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} training schedule digest is invalid"
        )
    updates = _sequence(training.get("updates"), field=f"{arm}.training.updates")
    if len(updates) != len(schedule) or _positive_int(
        config.get("updates"), field=f"{arm}.configuration.updates"
    ) != len(schedule):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} training update count differs from the schedule"
        )
    worlds_per_update = _positive_int(
        config.get("worlds_per_update"),
        field=f"{arm}.configuration.worlds_per_update",
    )
    total_worlds = 0
    total_transitions = 0
    task_ids: set[str] = set()
    sampling_identities: set[str] = set()
    sampling_seeds: set[int] = set()
    for update_index, (scheduled_value, update_value) in enumerate(
        zip(schedule, updates, strict=True)
    ):
        scheduled = _sequence(
            scheduled_value,
            field=f"{arm}.configuration.training_schedule[{update_index}]",
        )
        update = _mapping(
            update_value,
            field=f"{arm}.training.updates[{update_index}]",
        )
        if len(scheduled) != worlds_per_update:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} worlds-per-update differs from the schedule"
            )
        if (
            update.get("update_index") != update_index
            or update.get("tasks") != scheduled
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} executed task schedule differs at update {update_index}"
            )
        for task_value in scheduled:
            task = _mapping(task_value, field=f"{arm}.training.task")
            task_id = _nonempty_string(task.get("task_id"), field=f"{arm}.task_id")
            identity = _nonempty_string(
                task.get("policy_sampling_identity"),
                field=f"{arm}.policy_sampling_identity",
            )
            seed = _positive_int(
                task.get("policy_sampling_seed"),
                field=f"{arm}.policy_sampling_seed",
            )
            if (
                task_id in task_ids
                or identity in sampling_identities
                or seed in sampling_seeds
            ):
                raise RecurrentCounterfactualComparisonError(
                    f"{arm} training task or sampling identity is duplicated"
                )
            if seed != derive_recurrent_policy_sampling_seed(task_identity=identity):
                raise RecurrentCounterfactualComparisonError(
                    f"{arm} training policy seed does not match its identity"
                )
            task_ids.add(task_id)
            sampling_identities.add(identity)
            sampling_seeds.add(seed)
            _positive_int(task.get("environment_seed"), field=f"{arm}.environment_seed")
            _positive_int(task.get("rollout_ticks"), field=f"{arm}.rollout_ticks")
        rollout = _mapping(
            update.get("rollout"),
            field=f"{arm}.training.updates[{update_index}].rollout",
        )
        if rollout.get("world_count") != len(scheduled):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} rollout world count differs at update {update_index}"
            )
        transitions = _positive_int(
            rollout.get("transition_count"),
            field=f"{arm}.training.updates[{update_index}].transition_count",
        )
        optimizer = _mapping(
            update.get("optimizer"),
            field=f"{arm}.training.updates[{update_index}].optimizer",
        )
        if optimizer.get("update_index") != update_index:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} PPO update index drifted"
            )
        total_worlds += len(scheduled)
        total_transitions += transitions
    if (
        training.get("total_worlds") != total_worlds
        or training.get("total_transitions") != total_transitions
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} training totals differ from update evidence"
        )
    _validate_counterfactual_training(
        report,
        arm=arm,
        preregistered=preregistered_counterfactual,
        updates=updates,
    )


def _validate_counterfactual_training(
    report: Mapping[str, object],
    *,
    arm: str,
    preregistered: Mapping[str, object],
    updates: Sequence[object],
) -> None:
    training = _mapping(report.get("training"), field=f"{arm}.training")
    experiment = training.get("counterfactual_experiment")
    enabled = preregistered.get("enabled") is True
    if not enabled:
        if experiment is not None:
            raise RecurrentCounterfactualComparisonError(
                "base arm unexpectedly contains a counterfactual experiment"
            )
        for index, value in enumerate(updates):
            update = _mapping(value, field=f"{arm}.training.updates[{index}]")
            if (
                update.get("counterfactual_collection") is not None
                or update.get("counterfactual_auxiliary") is not None
            ):
                raise RecurrentCounterfactualComparisonError(
                    "base arm unexpectedly attempted counterfactual evidence"
                )
        return

    experiment_mapping = _mapping(
        experiment,
        field=f"{arm}.training.counterfactual_experiment",
    )
    resolved = _mapping(
        preregistered.get("resolved_configuration"),
        field=f"{arm}.counterfactual.resolved_configuration",
    )
    for field in (
        "collection",
        "auxiliary",
        "step",
        "bundles_per_update",
        "branch_tick_candidates",
        "workers",
    ):
        if experiment_mapping.get(field) != resolved.get(field):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} executed counterfactual {field} differs from preregistration"
            )
    if (
        experiment_mapping.get("enabled") is not True
        or experiment_mapping.get("source_artifact_policy")
        != "ephemeral_in_memory_post_ppo_model_identity_only"
        or experiment_mapping.get("durable_artifact_created") is not False
        or experiment_mapping.get("runtime_integrated") is not False
        or experiment_mapping.get("promotion_authorized") is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} counterfactual experiment lifecycle drifted"
        )

    observed_bundle_digests: set[str] = set()
    accepted_count = 0
    final_auxiliary_model_digest: str | None = None
    for update_index, value in enumerate(updates):
        update = _mapping(value, field=f"{arm}.training.updates[{update_index}]")
        collection_payload = _mapping(
            update.get("counterfactual_collection"),
            field=f"{arm}.updates[{update_index}].counterfactual_collection",
        )
        auxiliary = _mapping(
            update.get("counterfactual_auxiliary"),
            field=f"{arm}.updates[{update_index}].counterfactual_auxiliary",
        )
        collection = _reconstruct_collection_result(
            collection_payload,
            field=f"{arm}.updates[{update_index}].counterfactual_collection",
        )
        try:
            validate_recurrent_counterfactual_collection_result(collection)
        except RecurrentCounterfactualCollectionError as error:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} update {update_index} counterfactual collection is invalid"
            ) from error
        if stable_payload_digest(
            recurrent_counterfactual_collection_config_payload(collection.config)
        ) != stable_payload_digest(resolved.get("collection")):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} update {update_index} collection config drifted"
            )
        if len(collection.bundles) != resolved.get("bundles_per_update"):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} update {update_index} bundle count drifted"
            )
        row_groups = tuple(bundle.rows for bundle in collection.bundles)
        try:
            expected_bundle_digest = recurrent_counterfactual_auxiliary_bundle_digest(
                row_groups,
                artifact_digest=collection.source_artifact_digest,
            )
        except RecurrentCounterfactualAuxiliaryError as error:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} update {update_index} auxiliary rows are invalid"
            ) from error
        observed_bundle_digest = _sha256(
            auxiliary.get("bundle_digest"),
            field=f"{arm}.updates[{update_index}].auxiliary.bundle_digest",
        )
        if expected_bundle_digest != observed_bundle_digest:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} update {update_index} auxiliary bundle digest drifted"
            )
        if observed_bundle_digest in observed_bundle_digests:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} reused a counterfactual bundle across updates"
            )
        observed_bundle_digests.add(observed_bundle_digest)
        accepted_count = _validate_auxiliary_diagnostics(
            auxiliary,
            arm=arm,
            update_index=update_index,
            collection=collection,
            step_contract=_mapping(
                resolved.get("step"),
                field=f"{arm}.counterfactual.step",
            ),
            accepted_count_before=accepted_count,
        )
        final_auxiliary_model_digest = str(auxiliary["final_model_state_sha256"])
    # `final_auxiliary_model_digest` is intentionally not compared to the v4
    # canary's `model_states.after.state_sha256`: the former is emitted by
    # `recurrent_model_state_sha256`, while the latter uses the canary's JSON
    # metadata fingerprint.  Equality across those byte encodings is not a
    # state-integrity invariant.  The loop above still verifies that the final
    # auxiliary digest is a valid, transaction-consistent SHA256 value.
    _sha256(
        final_auxiliary_model_digest,
        field=f"{arm}.final_auxiliary_model_digest",
    )


def _reconstruct_collection_result(
    value: Mapping[str, object],
    *,
    field: str,
) -> RecurrentCounterfactualCollectionResult:
    try:
        config_payload = _mapping(value.get("config"), field=f"{field}.config")
        config = RecurrentCounterfactualCollectionConfig(
            horizons=_positive_int_sequence(
                config_payload.get("horizons"),
                field=f"{field}.config.horizons",
            ),
            gamma=_finite_number(config_payload.get("gamma"), field=f"{field}.gamma"),
        )
        bundles = []
        for index, bundle_value in enumerate(
            _sequence(value.get("bundles"), field=f"{field}.bundles")
        ):
            bundle = _mapping(bundle_value, field=f"{field}.bundles[{index}]")
            task_payload = _mapping(
                bundle.get("task"),
                field=f"{field}.bundles[{index}].task",
            )
            task = RecurrentCounterfactualCollectionTask(
                task_id=_nonempty_string(
                    task_payload.get("task_id"),
                    field=f"{field}.bundles[{index}].task_id",
                ),
                seed_role=str(task_payload.get("seed_role")),
                scenario=str(task_payload.get("scenario")),
                environment_seed=_positive_int(
                    task_payload.get("environment_seed"),
                    field=f"{field}.bundles[{index}].environment_seed",
                ),
                branch_tick_candidates=_nonnegative_int_sequence(
                    task_payload.get("branch_tick_candidates"),
                    field=f"{field}.bundles[{index}].branch_tick_candidates",
                ),
                source_policy_sampling_identity=_nonempty_string(
                    task_payload.get("source_policy_sampling_identity"),
                    field=f"{field}.bundles[{index}].source_policy_sampling_identity",
                ),
                branch_selection_identity=_nonempty_string(
                    task_payload.get("branch_selection_identity"),
                    field=f"{field}.bundles[{index}].branch_selection_identity",
                ),
                source_policy_sampling_seed=_nonnegative_int(
                    task_payload.get("source_policy_sampling_seed"),
                    field=f"{field}.bundles[{index}].source_policy_sampling_seed",
                ),
                branch_selection_seed=_nonnegative_int(
                    task_payload.get("branch_selection_seed"),
                    field=f"{field}.bundles[{index}].branch_selection_seed",
                ),
            )
            rows = tuple(
                copy.deepcopy(_mapping(row, field=f"{field}.bundles[{index}].row"))
                for row in _sequence(
                    bundle.get("rows"),
                    field=f"{field}.bundles[{index}].rows",
                )
            )
            bundles.append(
                RecurrentCounterfactualCollectionBundle(
                    task=task,
                    rows=rows,
                    selection=copy.deepcopy(
                        dict(
                            _mapping(
                                bundle.get("selection"),
                                field=f"{field}.bundles[{index}].selection",
                            )
                        )
                    ),
                    prefix_proof=copy.deepcopy(
                        dict(
                            _mapping(
                                bundle.get("prefix_proof"),
                                field=f"{field}.bundles[{index}].prefix_proof",
                            )
                        )
                    ),
                    compute=copy.deepcopy(
                        dict(
                            _mapping(
                                bundle.get("compute"),
                                field=f"{field}.bundles[{index}].compute",
                            )
                        )
                    ),
                    source_model_state_sha256=_sha256(
                        bundle.get("source_model_state_sha256"),
                        field=f"{field}.bundles[{index}].source_model_state_sha256",
                    ),
                    source_artifact_digest=_sha256(
                        bundle.get("source_artifact_digest"),
                        field=f"{field}.bundles[{index}].source_artifact_digest",
                    ),
                    valid_actions=tuple(
                        _nonempty_string(action, field=f"{field}.valid_action")
                        for action in _sequence(
                            bundle.get("valid_actions"),
                            field=f"{field}.bundles[{index}].valid_actions",
                        )
                    ),
                    horizons=_positive_int_sequence(
                        bundle.get("horizons"),
                        field=f"{field}.bundles[{index}].horizons",
                    ),
                    exact_digest=_sha256(
                        bundle.get("exact_digest"),
                        field=f"{field}.bundles[{index}].exact_digest",
                    ),
                )
            )
        return RecurrentCounterfactualCollectionResult(
            contract_version=str(value.get("contract_version")),
            source_model_state_sha256=_sha256(
                value.get("source_model_state_sha256"),
                field=f"{field}.source_model_state_sha256",
            ),
            source_artifact_digest=_sha256(
                value.get("source_artifact_digest"),
                field=f"{field}.source_artifact_digest",
            ),
            config=config,
            bundles=tuple(bundles),
            workers_requested=_positive_int(
                value.get("workers_requested"),
                field=f"{field}.workers_requested",
            ),
            workers_resolved=_positive_int(
                value.get("workers_resolved"),
                field=f"{field}.workers_resolved",
            ),
            worker_start_method=str(value.get("worker_start_method")),
            torch_threads_per_worker=_positive_int(
                value.get("torch_threads_per_worker"),
                field=f"{field}.torch_threads_per_worker",
            ),
            ordered_merge=_exact_bool(
                value.get("ordered_merge"),
                field=f"{field}.ordered_merge",
            ),
            aggregate_compute={
                str(key): _nonnegative_int(item, field=f"{field}.aggregate_compute")
                for key, item in _mapping(
                    value.get("aggregate_compute"),
                    field=f"{field}.aggregate_compute",
                ).items()
            },
            training_ran=_exact_bool(
                value.get("training_ran"),
                field=f"{field}.training_ran",
            ),
            runtime_action_selection_changed=_exact_bool(
                value.get("runtime_action_selection_changed"),
                field=f"{field}.runtime_action_selection_changed",
            ),
            promotion_authorized=_exact_bool(
                value.get("promotion_authorized"),
                field=f"{field}.promotion_authorized",
            ),
            exact_digest=_sha256(
                value.get("exact_digest"),
                field=f"{field}.exact_digest",
            ),
        )
    except (TypeError, ValueError, RecurrentCounterfactualCollectionError) as error:
        if isinstance(error, RecurrentCounterfactualComparisonError):
            raise
        raise RecurrentCounterfactualComparisonError(
            f"{field} cannot be reconstructed under the collection contract"
        ) from error


def _validate_auxiliary_diagnostics(
    value: Mapping[str, object],
    *,
    arm: str,
    update_index: int,
    collection: RecurrentCounterfactualCollectionResult,
    step_contract: Mapping[str, object],
    accepted_count_before: int,
) -> int:
    field = f"{arm}.updates[{update_index}].counterfactual_auxiliary"
    if (
        value.get("schema_version")
        != RECURRENT_COUNTERFACTUAL_AUXILIARY_STEP_SCHEMA_VERSION
    ):
        raise RecurrentCounterfactualComparisonError(f"{field} schema drifted")
    if (
        value.get("retry_authorized") is not False
        or value.get("optimizer_step_count") != 1
        or value.get("backward_pass_count") != 1
        or value.get("ppo_update_index") != update_index + 1
        or value.get("runtime_artifact_created") is not False
        or value.get("runtime_action_selection_changed") is not False
        or value.get("promotion_authorized") is not False
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{field} violates the one-use closed transaction contract"
        )
    pre_digest = _sha256(
        value.get("pre_model_state_sha256"),
        field=f"{field}.pre_model_state_sha256",
    )
    attempted_digest = _sha256(
        value.get("attempted_post_model_state_sha256"),
        field=f"{field}.attempted_post_model_state_sha256",
    )
    final_digest = _sha256(
        value.get("final_model_state_sha256"),
        field=f"{field}.final_model_state_sha256",
    )
    if pre_digest != collection.source_model_state_sha256:
        raise RecurrentCounterfactualComparisonError(
            f"{field} did not consume the exact collected post-PPO model"
        )
    if value.get("group_count") != len(collection.bundles) or value.get(
        "row_count"
    ) != sum(len(bundle.rows) for bundle in collection.bundles):
        raise RecurrentCounterfactualComparisonError(
            f"{field} group or row count differs from the collection"
        )
    min_prefix = _nonnegative_int(
        value.get("min_public_prefix_length"),
        field=f"{field}.min_public_prefix_length",
    )
    max_prefix = _nonnegative_int(
        value.get("max_public_prefix_length"),
        field=f"{field}.max_public_prefix_length",
    )
    if min_prefix > max_prefix:
        raise RecurrentCounterfactualComparisonError(
            f"{field} public-prefix length range is invalid"
        )
    learning_rate_multiplier = _finite_number(
        value.get("learning_rate_multiplier"),
        field=f"{field}.learning_rate_multiplier",
    )
    if learning_rate_multiplier != _finite_number(
        step_contract.get("learning_rate_multiplier"),
        field=f"{field}.step.learning_rate_multiplier",
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{field} learning-rate multiplier drifted"
        )
    learning_rates = _sequence(
        value.get("parameter_group_learning_rates"),
        field=f"{field}.parameter_group_learning_rates",
    )
    if not learning_rates or any(
        _finite_number(rate, field=f"{field}.parameter_group_learning_rate") <= 0.0
        for rate in learning_rates
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{field} parameter-group learning rates are invalid"
        )
    for metric in (
        "total_loss",
        "policy_improvement_kl",
        "behavior_kl",
        "value_loss",
        "gradient_norm_before_clip",
        "gradient_norm_after_clip",
        "parameter_delta_l2",
        "mean_behavior_kl_old_to_post",
        "max_state_behavior_kl_old_to_post",
        "mean_behavior_kl_limit",
        "max_state_behavior_kl_limit",
    ):
        parsed = _finite_number(value.get(metric), field=f"{field}.{metric}")
        lower_bound = (
            -_FLOAT32_KL_ROUNDOFF_ABSOLUTE_TOLERANCE
            if metric in _KL_DIAGNOSTIC_FIELDS
            else 0.0
        )
        if metric != "total_loss" and parsed < lower_bound:
            raise RecurrentCounterfactualComparisonError(
                f"{field}.{metric} is below its numerical non-negative tolerance"
            )
    if (
        float(value["gradient_norm_after_clip"])
        > float(step_contract["max_gradient_norm"]) + 1.0e-6
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{field} gradient clipping limit was exceeded"
        )
    audit = _mapping(
        step_contract.get("post_step_public_policy_audit"),
        field=f"{field}.step.post_step_public_policy_audit",
    )
    if value.get("mean_behavior_kl_limit") != audit.get(
        "mean_forward_kl_pi_old_to_pi_post_limit"
    ) or value.get("max_state_behavior_kl_limit") != audit.get(
        "max_state_forward_kl_pi_old_to_pi_post_limit"
    ):
        raise RecurrentCounterfactualComparisonError(
            f"{field} post-step KL limits drifted"
        )

    accepted = value.get("accepted")
    if type(accepted) is not bool:
        raise RecurrentCounterfactualComparisonError(
            f"{field}.accepted must be an exact boolean"
        )
    mean_kl = float(value["mean_behavior_kl_old_to_post"])
    max_kl = float(value["max_state_behavior_kl_old_to_post"])
    mean_limit = float(value["mean_behavior_kl_limit"])
    max_limit = float(value["max_state_behavior_kl_limit"])
    if accepted:
        next_count = accepted_count_before + 1
        if (
            value.get("rejection_reason") is not None
            or value.get("rollback_performed") is not False
            or value.get("optimizer_state_restored") is not False
            or value.get("parameter_group_learning_rates_restored") is not True
            or value.get("update_counters_restored") is not False
            or value.get("rows_stale_after_step") is not True
            or value.get("rows_exact_model_valid_after_step") is not False
            or attempted_digest == pre_digest
            or final_digest != attempted_digest
            or mean_kl > mean_limit
            or max_kl > max_limit
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{field} accepted transaction diagnostics are inconsistent"
            )
    else:
        next_count = accepted_count_before
        reason = value.get("rejection_reason")
        valid_reason = (
            (
                reason == "optimizer_step_did_not_change_model_state"
                and attempted_digest == pre_digest
            )
            or (
                reason == "mean_public_policy_kl_limit_exceeded"
                and mean_kl > mean_limit
            )
            or (
                reason == "max_state_public_policy_kl_limit_exceeded"
                and mean_kl <= mean_limit
                and max_kl > max_limit
            )
        )
        if (
            not valid_reason
            or value.get("rollback_performed") is not True
            or value.get("optimizer_state_restored") is not True
            or value.get("parameter_group_learning_rates_restored") is not True
            or value.get("update_counters_restored") is not True
            or value.get("rows_stale_after_step") is not False
            or value.get("rows_exact_model_valid_after_step") is not True
            or final_digest != pre_digest
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{field} rejected transaction rollback diagnostics are inconsistent"
            )
    if value.get("auxiliary_update_count") != next_count:
        raise RecurrentCounterfactualComparisonError(
            f"{field} accepted auxiliary update count drifted"
        )
    return next_count


def _validate_evaluations(report: Mapping[str, object], *, arm: str) -> None:
    evaluations = _mapping(report.get("evaluations"), field=f"{arm}.evaluations")
    if set(evaluations) != {"before", "after"}:
        raise RecurrentCounterfactualComparisonError(
            f"{arm} evaluations must contain before and after"
        )
    prereg = _mapping(
        report.get("evaluation_preregistration"),
        field=f"{arm}.evaluation_preregistration",
    )
    expected_label = prereg.get("synthetic_noncandidate_digest_label")
    for phase in ("before", "after"):
        phase_payload = _mapping(
            evaluations.get(phase),
            field=f"{arm}.evaluations.{phase}",
        )
        if set(phase_payload) != set(SELECTION_MODES):
            raise RecurrentCounterfactualComparisonError(
                f"{arm}.{phase} evaluation modes drifted"
            )
        for mode in SELECTION_MODES:
            evaluation = _mapping(
                phase_payload.get(mode),
                field=f"{arm}.evaluations.{phase}.{mode}",
            )
            if evaluation.get("schema_version") != RECURRENT_EVALUATION_SCHEMA_VERSION:
                raise RecurrentCounterfactualComparisonError(
                    f"{arm}.{phase}.{mode} nested evaluation is not v5"
                )
            try:
                validate_recurrent_evaluation_report(evaluation)
            except RecurrentEvaluationError as error:
                raise RecurrentCounterfactualComparisonError(
                    f"{arm}.{phase}.{mode} failed the public v5 evaluator contract"
                ) from error
            contract = _mapping(
                evaluation.get("evaluation_contract"),
                field=f"{arm}.{phase}.{mode}.evaluation_contract",
            )
            if (
                contract.get("candidate_action_selection") != mode
                or contract.get("candidate_recurrent_state")
                != "one_hidden_state_per_agent"
                or contract.get("ticks") != 120
                or contract.get("candidate_replay_verification")
                != "every_run_exact_digest_repeat"
            ):
                raise RecurrentCounterfactualComparisonError(
                    f"{arm}.{phase}.{mode} evaluation mode/history drifted"
                )
            _validate_evaluation_sampling_contract(
                contract,
                mode=mode,
                field=f"{arm}.{phase}.{mode}.evaluation_contract",
            )
            provenance = _mapping(
                evaluation.get("candidate_provenance"),
                field=f"{arm}.{phase}.{mode}.candidate_provenance",
            )
            if provenance.get("synthetic_noncandidate_digest_label") != expected_label:
                raise RecurrentCounterfactualComparisonError(
                    f"{arm}.{phase}.{mode} synthetic model label drifted"
                )
    for mode in SELECTION_MODES:
        before = _evaluation_for(report, phase="before", mode=mode)
        after = _evaluation_for(report, phase="after", mode=mode)
        if before.get("evaluation_contract") != after.get(
            "evaluation_contract"
        ) or before.get("seed_plan") != after.get("seed_plan"):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} before/after evaluation stream drifted for {mode}"
            )
        before_contexts = _evaluation_contexts(before, field=f"{arm}.before.{mode}")
        after_contexts = _evaluation_contexts(after, field=f"{arm}.after.{mode}")
        if tuple(before_contexts) != tuple(after_contexts):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} before/after contexts drifted for {mode}"
            )
        for context in before_contexts:
            before_policies = _mapping(
                before_contexts[context].get("policies"),
                field=f"{arm}.before.{mode}.{context}.policies",
            )
            after_policies = _mapping(
                after_contexts[context].get("policies"),
                field=f"{arm}.after.{mode}.{context}.policies",
            )
            for control in ("mind_v3_linear", "masked_random"):
                if before_policies.get(control) != after_policies.get(control):
                    raise RecurrentCounterfactualComparisonError(
                        f"{arm} {control} drifted before/after in {context}"
                    )
        if set(_candidate_run_map(report, phase="before", mode=mode)) != set(
            _candidate_run_map(report, phase="after", mode=mode)
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{arm} before/after raw candidate grid drifted for {mode}"
            )


def _validate_report_paired_deltas(
    report: Mapping[str, object],
    *,
    arm: str,
) -> None:
    observed = _mapping(
        report.get("paired_outcome_deltas"),
        field=f"{arm}.paired_outcome_deltas",
    )
    if set(observed) != set(SELECTION_MODES):
        raise RecurrentCounterfactualComparisonError(
            f"{arm} paired outcome modes drifted"
        )
    for mode in SELECTION_MODES:
        expected = _causal_delta_payload(
            _evaluation_for(report, phase="before", mode=mode),
            _evaluation_for(report, phase="after", mode=mode),
        )
        if observed.get(mode) != expected:
            raise RecurrentCounterfactualComparisonError(
                f"{arm} paired outcome deltas differ from raw runs for {mode}"
            )


def _causal_delta_payload(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    before_contract = _mapping(
        before.get("evaluation_contract"), field="before.evaluation_contract"
    )
    after_contract = _mapping(
        after.get("evaluation_contract"), field="after.evaluation_contract"
    )
    sampling_fields = (
        "candidate_action_selection",
        "candidate_sampling_stream_id",
        "candidate_sampling_seeds",
        "candidate_sampling_seed_count",
    )
    if any(
        before_contract.get(field) != after_contract.get(field)
        for field in sampling_fields
    ):
        raise RecurrentCounterfactualComparisonError(
            "before/after evaluation sampling contract drifted"
        )
    before_contexts = _evaluation_contexts(before, field="before")
    after_contexts = _evaluation_contexts(after, field="after")
    if tuple(before_contexts) != tuple(after_contexts):
        raise RecurrentCounterfactualComparisonError(
            "before/after evaluation contexts drifted"
        )
    rows: list[dict[str, object]] = []
    candidate_vs_controls: dict[str, object] = {}
    for context, before_context in before_contexts.items():
        after_context = after_contexts[context]
        before_policies = _mapping(
            before_context.get("policies"), field="before.policies"
        )
        after_policies = _mapping(after_context.get("policies"), field="after.policies")
        for control in ("mind_v3_linear", "masked_random"):
            if before_policies.get(control) != after_policies.get(control):
                raise RecurrentCounterfactualComparisonError(
                    f"before/after {control} control drifted"
                )
        before_runs = _policy_runs(before_policies, policy="public_recurrent")
        after_runs = _policy_runs(after_policies, policy="public_recurrent")
        if len(before_runs) != len(after_runs) or not before_runs:
            raise RecurrentCounterfactualComparisonError(
                "before/after candidate run counts drifted"
            )
        for before_run, after_run in zip(before_runs, after_runs, strict=True):
            if _run_identity(before_run) != _run_identity(after_run):
                raise RecurrentCounterfactualComparisonError(
                    "before/after candidate run identity drifted"
                )
            row: dict[str, object] = {
                "context": before_run["context"],
                "seed": before_run["seed"],
                "policy_sampling_seed": before_run.get("policy_sampling_seed"),
            }
            for metric in _CAUSAL_METRICS:
                delta = _finite_number(
                    after_run.get(metric), field=metric
                ) - _finite_number(before_run.get(metric), field=metric)
                row[f"{metric}_delta"] = (
                    int(delta)
                    if metric
                    in {
                        "terminal_alive",
                        "births",
                        "deaths",
                        "unsupported_requested_action_count",
                        "heuristic_action_source_count",
                    }
                    else _rounded(delta)
                )
            rows.append(row)
        candidate_vs_controls[context] = {
            "before": before_context.get("paired_deltas"),
            "after": after_context.get("paired_deltas"),
        }
    return {
        "run_count": len(rows),
        "runs": rows,
        "mean_deltas": {
            f"{metric}_delta": _rounded(
                math.fsum(float(row[f"{metric}_delta"]) for row in rows) / len(rows)
            )
            for metric in _CAUSAL_METRICS
        },
        "candidate_minus_controls": candidate_vs_controls,
        "controls_exactly_stable_before_after": True,
        "same_selection_seed_stream_before_after": True,
        "candidate_sampling_stream_id": before_contract.get(
            "candidate_sampling_stream_id"
        ),
        "candidate_sampling_seeds": before_contract.get("candidate_sampling_seeds"),
    }


def _validate_three_arm_match(
    reports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    base = reports[BASE_ARM]
    shuffled = reports[SHUFFLED_ARM]
    base_source = _mapping(base.get("source"), field="base.source")
    for arm in (EXACT_ARM, SHUFFLED_ARM):
        if reports[arm].get("source") != base_source:
            raise RecurrentCounterfactualComparisonError(
                "three-arm source head, status, or manifest differs"
            )
        if reports[arm].get("seed_registry_sha256") != base.get("seed_registry_sha256"):
            raise RecurrentCounterfactualComparisonError(
                "three-arm seed registry differs"
            )

    base_config = _mapping(base.get("configuration"), field="base.configuration")
    normalized_config = _normalized_counterfactual_config(base_config)
    for arm in (EXACT_ARM, SHUFFLED_ARM):
        candidate = _mapping(
            reports[arm].get("configuration"),
            field=f"{arm}.configuration",
        )
        if _normalized_counterfactual_config(candidate) != normalized_config:
            raise RecurrentCounterfactualComparisonError(
                "three-arm configuration differs beyond the preregistered "
                "counterfactual role"
            )
    exact_counterfactual = _normalized_counterfactual_contract(
        _mapping(
            _mapping(
                reports[EXACT_ARM].get("configuration"),
                field="exact.configuration",
            ).get("counterfactual_auxiliary"),
            field="exact.configuration.counterfactual_auxiliary",
        )
    )
    shuffled_counterfactual = _normalized_counterfactual_contract(
        _mapping(
            _mapping(
                reports[SHUFFLED_ARM].get("configuration"),
                field="shuffled.configuration",
            ).get("counterfactual_auxiliary"),
            field="shuffled.configuration.counterfactual_auxiliary",
        )
    )
    if exact_counterfactual != shuffled_counterfactual:
        raise RecurrentCounterfactualComparisonError(
            "exact and shuffled active counterfactual configurations differ"
        )

    base_prereg = _normalized_preregistration(base)
    for arm in (EXACT_ARM, SHUFFLED_ARM):
        if _normalized_preregistration(reports[arm]) != base_prereg:
            raise RecurrentCounterfactualComparisonError(
                "three-arm preregistration differs beyond the counterfactual role"
            )

    base_training = _mapping(base.get("training"), field="base.training")
    common_training_fields = (
        "contract_version",
        "learner_seed",
        "device",
        "deterministic_algorithms_enabled",
        "model_config",
        "ppo_config",
        "rollout_execution",
        "training_scenarios",
        "total_worlds",
    )
    for arm in (EXACT_ARM, SHUFFLED_ARM):
        training = _mapping(reports[arm].get("training"), field=f"{arm}.training")
        for field in common_training_fields:
            if training.get(field) != base_training.get(field):
                raise RecurrentCounterfactualComparisonError(
                    f"three-arm training contract field {field} differs"
                )
    if any(
        _mapping(reports[arm].get("configuration"), field=f"{arm}.configuration").get(
            "training_schedule"
        )
        != base_config.get("training_schedule")
        for arm in (EXACT_ARM, SHUFFLED_ARM)
    ):
        raise RecurrentCounterfactualComparisonError(
            "three-arm task/sampling schedule differs"
        )

    base_states = _mapping(base.get("model_states"), field="base.model_states")
    for arm in (EXACT_ARM, SHUFFLED_ARM):
        states = _mapping(reports[arm].get("model_states"), field=f"{arm}.model_states")
        if states.get("before") != base_states.get("before"):
            raise RecurrentCounterfactualComparisonError(
                "three-arm initial model fingerprint differs"
            )

    for mode in SELECTION_MODES:
        normalized_before = _normalized_before_evaluation(
            _evaluation_for(base, phase="before", mode=mode)
        )
        for arm in (EXACT_ARM, SHUFFLED_ARM):
            candidate = _normalized_before_evaluation(
                _evaluation_for(reports[arm], phase="before", mode=mode)
            )
            if candidate != normalized_before:
                raise RecurrentCounterfactualComparisonError(
                    f"three-arm base-before evaluation differs for {mode}"
                )
        for phase in ("before", "after"):
            base_eval = _evaluation_for(base, phase=phase, mode=mode)
            base_contract = base_eval.get("evaluation_contract")
            base_seed_plan = base_eval.get("seed_plan")
            base_contexts = _evaluation_contexts(
                base_eval,
                field=f"base.{phase}.{mode}",
            )
            base_grid = set(_candidate_run_map(base, phase=phase, mode=mode))
            for arm in (EXACT_ARM, SHUFFLED_ARM):
                candidate_eval = _evaluation_for(reports[arm], phase=phase, mode=mode)
                if (
                    candidate_eval.get("evaluation_contract") != base_contract
                    or candidate_eval.get("seed_plan") != base_seed_plan
                ):
                    raise RecurrentCounterfactualComparisonError(
                        "three-arm evaluation stream or seed plan differs"
                    )
                if (
                    set(_candidate_run_map(reports[arm], phase=phase, mode=mode))
                    != base_grid
                ):
                    raise RecurrentCounterfactualComparisonError(
                        "three-arm raw environment/stream grid differs"
                    )
                candidate_contexts = _evaluation_contexts(
                    candidate_eval,
                    field=f"{arm}.{phase}.{mode}",
                )
                if tuple(candidate_contexts) != tuple(base_contexts):
                    raise RecurrentCounterfactualComparisonError(
                        "three-arm evaluation contexts differ"
                    )
                for context in base_contexts:
                    base_policies = _mapping(
                        base_contexts[context].get("policies"),
                        field=f"base.{context}.policies",
                    )
                    candidate_policies = _mapping(
                        candidate_contexts[context].get("policies"),
                        field=f"{arm}.{context}.policies",
                    )
                    for control in ("mind_v3_linear", "masked_random"):
                        if candidate_policies.get(control) != base_policies.get(
                            control
                        ):
                            raise RecurrentCounterfactualComparisonError(
                                f"three-arm {control} control differs"
                            )

    stream = _mapping(
        base_config.get("evaluation_sampling_stream"),
        field="base.configuration.evaluation_sampling_stream",
    )
    stream_id = _nonempty_string(
        stream.get("id"),
        field="base.configuration.evaluation_sampling_stream.id",
    )
    if stream.get("arm_independent") is not True:
        raise RecurrentCounterfactualComparisonError(
            "evaluation sampling stream is not arm-independent"
        )
    stream_contract = _mapping(
        stream.get("contract"),
        field="base.configuration.evaluation_sampling_stream.contract",
    )
    sampled = _mapping(
        _evaluation_for(
            base, phase="before", mode=PUBLIC_RECURRENT_SAMPLED_SELECTION
        ).get("evaluation_contract"),
        field="base.before.sampled.evaluation_contract",
    )
    if sampled.get("candidate_sampling_stream_id") != stream_id:
        raise RecurrentCounterfactualComparisonError(
            "executed sampled stream differs from preregistration"
        )
    sampled_eval = _evaluation_for(
        base,
        phase="before",
        mode=PUBLIC_RECURRENT_SAMPLED_SELECTION,
    )
    sampled_seed_plan = _mapping(
        sampled_eval.get("seed_plan"),
        field="base.before.sampled.seed_plan",
    )
    actual_fixtures = [
        context.removeprefix("fixture:")
        for context in _evaluation_contexts(
            sampled_eval,
            field="base.before.sampled",
        )
        if context.startswith("fixture:")
    ]
    sampled_count = _positive_int(
        sampled.get("candidate_sampling_seed_count"),
        field="base.before.sampled.candidate_sampling_seed_count",
    )
    if (
        stream_contract.get("seed_plan_digest") != sampled_seed_plan.get("digest")
        or stream_contract.get("fixtures") != actual_fixtures
        or stream_contract.get("candidate_sampling_seed_count") != sampled_count
        or base_config.get("evaluation_fixtures") != actual_fixtures
        or base_config.get("candidate_sampling_seed_count") != sampled_count
    ):
        raise RecurrentCounterfactualComparisonError(
            "preregistered evaluation stream contract differs from execution"
        )
    shuffled_cf = _mapping(
        _mapping(shuffled.get("configuration"), field="shuffled.configuration").get(
            "counterfactual_auxiliary"
        ),
        field="shuffled.counterfactual_auxiliary",
    )
    shuffled_negative = _mapping(
        shuffled_cf.get("scientific_negative_control"),
        field="shuffled.scientific_negative_control",
    )
    manifest = _mapping(
        base_source.get("source_file_hash_manifest"),
        field="base.source.manifest",
    )
    return {
        "git_head_observed": base_source["git_head_observed"],
        "source_manifest_aggregate_sha256": manifest["aggregate_sha256"],
        "training_schedule_sha256": base_config["training_schedule_sha256"],
        "before_model_state_sha256": _mapping(
            base_states.get("before"), field="base.model_states.before"
        )["state_sha256"],
        "evaluation_sampling_stream_id": stream_id,
        "label_shuffle_permutation_seed": shuffled_negative["permutation_seed"],
    }


def _normalized_counterfactual_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    normalized = copy.deepcopy(dict(config))
    # The disabled base arm does not execute counterfactual collection.  Its
    # dormant branch ticks, bundle count, and worker count therefore are not an
    # experimental treatment and need not equal the two active arms.  Active
    # exact/shuffled settings are compared separately after normalizing only
    # their preregistered label role.
    normalized.pop("counterfactual_auxiliary", None)
    return normalized


def _normalized_counterfactual_contract(
    value: Mapping[str, object],
) -> dict[str, object]:
    normalized = copy.deepcopy(dict(value))
    for field in (
        "enabled",
        "mode",
        "one_auxiliary_step_per_ppo_update",
        "exact_branch_labels_consumed",
    ):
        normalized[field] = "<COUNTERFACTUAL_ARM>"
    negative = _mapping(
        normalized.get("scientific_negative_control"),
        field="counterfactual.scientific_negative_control",
    )
    for field in (
        "enabled",
        "mode",
        "permutation_seed_explicit",
        "permutation_seed",
    ):
        negative[field] = "<COUNTERFACTUAL_ARM>"  # type: ignore[index]
    resolved = _mapping(
        normalized.get("resolved_configuration"),
        field="counterfactual.resolved_configuration",
    )
    auxiliary = _mapping(
        resolved.get("auxiliary"),
        field="counterfactual.resolved_configuration.auxiliary",
    )
    auxiliary_negative = _mapping(
        auxiliary.get("scientific_negative_control"),
        field="counterfactual.resolved.auxiliary.scientific_negative_control",
    )
    for field in ("enabled", "mode", "seed"):
        auxiliary_negative[field] = "<COUNTERFACTUAL_ARM>"  # type: ignore[index]
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
    prereg["run_config"] = _normalized_counterfactual_config(
        _mapping(prereg.get("run_config"), field="preregistration.run_config")
    )
    return prereg


def _normalized_before_evaluation(
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    normalized = copy.deepcopy(dict(evaluation))
    provenance = _mapping(
        normalized.get("candidate_provenance"),
        field="before.candidate_provenance",
    )
    provenance["synthetic_noncandidate_digest_label"] = "<ARM_LABEL>"  # type: ignore[index]
    replay = _mapping(
        normalized.get("replay_verification"),
        field="before.replay_verification",
    )
    for check_value in _sequence(replay.get("checks"), field="before.replay.checks"):
        check = _mapping(check_value, field="before.replay.check")
        check["digest"] = "<ARM_REPLAY_DIGEST>"  # type: ignore[index]
        check["outcome_evidence_sha256"] = "<ARM_OUTCOME_EVIDENCE>"  # type: ignore[index]
    for context in _evaluation_contexts(normalized, field="before").values():
        policies = _mapping(context.get("policies"), field="before.policies")
        for run in _policy_runs(policies, policy="public_recurrent"):
            run["replay_digest"] = "<ARM_REPLAY_DIGEST>"  # type: ignore[index]
            run["outcome_evidence_sha256"] = "<ARM_OUTCOME_EVIDENCE>"  # type: ignore[index]
    return normalized


def _arm_training_effects(
    report: Mapping[str, object],
    *,
    mode: str,
) -> dict[RunIdentity, dict[str, float]]:
    before = _candidate_run_map(report, phase="before", mode=mode)
    after = _candidate_run_map(report, phase="after", mode=mode)
    if set(before) != set(after):
        raise RecurrentCounterfactualComparisonError(
            f"before/after candidate identity grid differs for {mode}"
        )
    return {
        identity: _subtract_vectors(
            _run_metric_vector(after[identity]),
            _run_metric_vector(before[identity]),
        )
        for identity in sorted(before, key=_identity_sort_key)
    }


def _descriptive_contrast(
    left: Mapping[RunIdentity, Mapping[str, float]],
    right: Mapping[RunIdentity, Mapping[str, float]],
    *,
    estimand: str,
    left_arm: str,
    right_arm: str,
) -> dict[str, object]:
    if set(left) != set(right) or not left:
        raise RecurrentCounterfactualComparisonError(
            "contrast arms do not share one non-empty raw run grid"
        )
    rows: list[dict[str, object]] = []
    for identity in sorted(left, key=_identity_sort_key):
        left_effect = dict(left[identity])
        right_effect = dict(right[identity])
        contrast = _subtract_vectors(left_effect, right_effect)
        context, seed, sampling_seed = identity
        rows.append(
            {
                "context": context,
                "environment_seed": seed,
                "policy_sampling_seed": sampling_seed,
                f"{left_arm}_after_minus_before": left_effect,
                f"{right_arm}_after_minus_before": right_effect,
                "matched_contrast": contrast,
            }
        )
    return {
        "estimand": estimand,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "raw_unit": "environment_policy_sampling_stream_run_cell",
        "raw_environment_policy_stream_rows": rows,
        "raw_run_cell_count": len(rows),
        "overall_mean_contrast": _mean_vectors(
            [
                _mapping(row["matched_contrast"], field="matched_contrast")
                for row in rows
            ]
        ),
        "context_stratified": _grouped_contrast_rows(
            rows,
            key_fields=("context",),
        ),
        "environment_stratified": _grouped_contrast_rows(
            rows,
            key_fields=("context", "environment_seed"),
        ),
        "policy_sampling_stream_stratified": _grouped_contrast_rows(
            rows,
            key_fields=("context", "policy_sampling_seed"),
        ),
        "p_values_emitted": False,
        "independence_assumption_made": False,
    }


def _grouped_contrast_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in key_fields)].append(row)
    result = []
    for key in sorted(
        grouped, key=lambda item: tuple(_sort_value(value) for value in item)
    ):
        group = grouped[key]
        result.append(
            {
                **dict(zip(key_fields, key, strict=True)),
                "raw_run_cell_count": len(group),
                "mean_matched_contrast": _mean_vectors(
                    [
                        _mapping(row.get("matched_contrast"), field="matched_contrast")
                        for row in group
                    ]
                ),
            }
        )
    return result


def _run_metric_vector(run: Mapping[str, object]) -> dict[str, float]:
    vector = {
        "terminal_nonextinct": float(
            _nonnegative_int(run.get("terminal_alive"), field="terminal_alive") > 0
        ),
        "terminal_alive": float(run["terminal_alive"]),
    }
    for metric in _OUTCOME_METRICS:
        if metric in {"terminal_nonextinct", "terminal_alive"}:
            continue
        if metric == "learned_normalized_entropy_mean":
            distribution = _mapping(
                run.get("learned_masked_distribution"),
                field="learned_masked_distribution",
            )
            means = _mapping(
                distribution.get("metric_means"),
                field="learned_masked_distribution.metric_means",
            )
            value = means.get("normalized_entropy")
        elif metric == "learned_eat_probability_mean":
            distribution = _mapping(
                run.get("learned_masked_distribution"),
                field="learned_masked_distribution",
            )
            means = _mapping(
                distribution.get("metric_means"),
                field="learned_masked_distribution.metric_means",
            )
            value = means.get("eat_probability")
        else:
            value = run.get(metric)
        vector[metric] = _finite_number(value, field=metric)
    action_counts = _mapping(
        run.get("requested_action_counts"),
        field="requested_action_counts",
    )
    decision_count = _nonnegative_int(
        run.get("policy_decision_record_count"),
        field="policy_decision_record_count",
    )
    if decision_count <= 0:
        raise RecurrentCounterfactualComparisonError(
            "candidate run has no policy decisions"
        )
    for action in ACTION_NAMES:
        count = _nonnegative_int(
            action_counts.get(action, 0),
            field=f"requested_action_counts.{action}",
        )
        vector[f"requested_action_count:{action}"] = float(count)
        vector[f"requested_action_share:{action}"] = count / decision_count
    components = _mapping(
        run.get("reward_component_totals"),
        field="reward_component_totals",
    )
    for component, value in sorted(components.items()):
        vector[f"reward_component:{component}"] = _finite_number(
            value,
            field=f"reward_component_totals.{component}",
        )
    return vector


def _subtract_vectors(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, float]:
    if set(left) != set(right):
        raise RecurrentCounterfactualComparisonError(
            "matched metric vector schemas differ"
        )
    return {key: _rounded(float(left[key]) - float(right[key])) for key in sorted(left)}


def _mean_vectors(vectors: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not vectors:
        raise RecurrentCounterfactualComparisonError(
            "cannot summarize an empty metric vector set"
        )
    keys = set(vectors[0])
    if any(set(vector) != keys for vector in vectors):
        raise RecurrentCounterfactualComparisonError(
            "summary metric vector schemas differ"
        )
    return {
        key: _rounded(
            math.fsum(_finite_number(vector[key], field=key) for vector in vectors)
            / len(vectors)
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
    result: dict[RunIdentity, Mapping[str, object]] = {}
    for context in _evaluation_contexts(
        evaluation,
        field=f"{phase}.{mode}",
    ).values():
        policies = _mapping(context.get("policies"), field="policies")
        for run in _policy_runs(policies, policy="public_recurrent"):
            identity = _run_identity(run)
            if identity in result:
                raise RecurrentCounterfactualComparisonError(
                    "candidate raw run identity is duplicated"
                )
            result[identity] = run
    if not result:
        raise RecurrentCounterfactualComparisonError("candidate raw run grid is empty")
    return result


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
    field: str,
) -> dict[str, Mapping[str, object]]:
    contexts = {
        # `broad` is the evaluation report's structural container key.  Raw
        # runs inside that container intentionally retain context
        # `broad_default`; conflating the two makes reconstruction disagree
        # with the causal-canary producer even when every numeric delta agrees.
        "broad": _mapping(
            evaluation.get("broad"),
            field=f"{field}.broad",
        )
    }
    for index, value in enumerate(
        _sequence(evaluation.get("fixtures"), field=f"{field}.fixtures")
    ):
        fixture = _mapping(value, field=f"{field}.fixtures[{index}]")
        name = _nonempty_string(
            fixture.get("fixture"),
            field=f"{field}.fixtures[{index}].fixture",
        )
        context = f"fixture:{name}"
        if context in contexts:
            raise RecurrentCounterfactualComparisonError(
                "evaluation fixture names are duplicated"
            )
        contexts[context] = fixture
    return contexts


def _policy_runs(
    policies: Mapping[str, object],
    *,
    policy: str,
) -> tuple[Mapping[str, object], ...]:
    payload = _mapping(policies.get(policy), field=f"policies.{policy}")
    return tuple(
        _mapping(run, field=f"policies.{policy}.run")
        for run in _sequence(payload.get("runs"), field=f"policies.{policy}.runs")
    )


def _run_identity(run: Mapping[str, object]) -> RunIdentity:
    context = _nonempty_string(run.get("context"), field="run.context")
    seed = _positive_int(run.get("seed"), field="run.seed")
    sampling_seed = run.get("policy_sampling_seed")
    if sampling_seed is not None:
        sampling_seed = _nonnegative_int(
            sampling_seed,
            field="run.policy_sampling_seed",
        )
    return context, seed, sampling_seed


def _validate_evaluation_sampling_contract(
    contract: Mapping[str, object],
    *,
    mode: str,
    field: str,
) -> None:
    stream_id = _nonempty_string(
        contract.get("candidate_sampling_stream_id"),
        field=f"{field}.candidate_sampling_stream_id",
    )
    count = _positive_int(
        contract.get("candidate_sampling_seed_count"),
        field=f"{field}.candidate_sampling_seed_count",
    )
    raw_seeds = _sequence(
        contract.get("candidate_sampling_seeds"),
        field=f"{field}.candidate_sampling_seeds",
    )
    observed = tuple(
        _nonnegative_int(seed, field=f"{field}.candidate_sampling_seeds")
        for seed in raw_seeds
    )
    if mode == PUBLIC_RECURRENT_ARGMAX_SELECTION:
        if (
            count != 1
            or observed
            or contract.get("candidate_sampling_seed_contract") != "none_argmax"
        ):
            raise RecurrentCounterfactualComparisonError(
                f"{field} argmax sampling contract drifted"
            )
        return
    if not 1 <= count <= 32 or len(observed) != count:
        raise RecurrentCounterfactualComparisonError(
            f"{field} sampled stream count drifted"
        )
    expected = tuple(
        int.from_bytes(
            hashlib.sha256(
                (
                    f"{_EVALUATION_POLICY_SAMPLING_NAMESPACE}|"
                    f"{stream_id}|replicate-{index:04d}"
                ).encode("ascii")
            ).digest()[:8],
            byteorder="big",
        )
        & (2**63 - 1)
        for index in range(count)
    )
    if observed != expected or len(set(observed)) != len(observed):
        raise RecurrentCounterfactualComparisonError(
            f"{field} sampled seeds do not match their namespaced stream"
        )


def _exact_bool(value: object, *, field: str) -> bool:
    if type(value) is not bool:
        raise RecurrentCounterfactualComparisonError(
            f"{field} must be an exact boolean"
        )
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentCounterfactualComparisonError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RecurrentCounterfactualComparisonError(f"{field} must be a sequence")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RecurrentCounterfactualComparisonError(
            f"{field} must be a non-empty trimmed string"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentCounterfactualComparisonError(
            f"{field} must be a positive integer"
        )
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentCounterfactualComparisonError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_int_sequence(value: object, *, field: str) -> tuple[int, ...]:
    sequence = _sequence(value, field=field)
    if not sequence:
        raise RecurrentCounterfactualComparisonError(f"{field} must be non-empty")
    result = tuple(
        _positive_int(item, field=f"{field}[{index}]")
        for index, item in enumerate(sequence)
    )
    if len(set(result)) != len(result):
        raise RecurrentCounterfactualComparisonError(f"{field} must be unique")
    return result


def _nonnegative_int_sequence(value: object, *, field: str) -> tuple[int, ...]:
    sequence = _sequence(value, field=field)
    if not sequence:
        raise RecurrentCounterfactualComparisonError(f"{field} must be non-empty")
    result = tuple(
        _nonnegative_int(item, field=f"{field}[{index}]")
        for index, item in enumerate(sequence)
    )
    if len(set(result)) != len(result):
        raise RecurrentCounterfactualComparisonError(f"{field} must be unique")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecurrentCounterfactualComparisonError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise RecurrentCounterfactualComparisonError(f"{field} must be a finite number")
    return parsed


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecurrentCounterfactualComparisonError(
            f"{field} must be a lowercase SHA256 digest"
        )
    return value


def _canonical_clone(value: Mapping[str, object], *, field: str) -> dict[str, object]:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        loaded = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (TypeError, ValueError) as error:
        raise RecurrentCounterfactualComparisonError(
            f"{field} must be finite JSON-compatible data"
        ) from error
    if not isinstance(loaded, dict):
        raise RecurrentCounterfactualComparisonError(f"{field} must be a mapping")
    return loaded


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecurrentCounterfactualComparisonError(
                f"duplicate JSON key is forbidden: {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise RecurrentCounterfactualComparisonError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _identity_sort_key(identity: RunIdentity) -> tuple[str, int, int]:
    return identity[0], identity[1], -1 if identity[2] is None else identity[2]


def _sort_value(value: object) -> tuple[int, object]:
    if value is None:
        return 0, ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1, value
    return 2, str(value)


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise RecurrentCounterfactualComparisonError(
            "comparison produced a non-finite value"
        )
    rounded = round(value, 12)
    return 0.0 if rounded == 0.0 else rounded


__all__ = [
    "BASE_ARM",
    "EXACT_ARM",
    "SHUFFLED_ARM",
    "RECURRENT_COUNTERFACTUAL_ANALYZER_PROVENANCE_SCHEMA_VERSION",
    "RECURRENT_COUNTERFACTUAL_COMPARISON_SCHEMA_VERSION",
    "RecurrentCounterfactualComparisonError",
    "analyze_recurrent_counterfactual_three_arm",
    "write_recurrent_counterfactual_three_arm_comparison",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
