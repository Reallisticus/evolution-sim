from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import TrajectoryJsonlDataset
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _resolve_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import (
    MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
    RolloutContextConfig,
    RolloutContextState,
    rollout_context_feature_contract,
)
from evolution_sim.mind.rollout_sequence_strict_seed_support_recheck import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V138_REPORT_PATH,
    MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION,
    _source_role_summary,
)
from evolution_sim.mind.rollout_sequence_support_audit import (
    ACTION_ONLY_BASELINE_ID,
    ACTION_ORDER_BASELINE_ID,
    DOMINANT_PREDICTED_ACTION_SHARE_MAX,
    SEQUENCE_MODEL_ID,
    STRICT_HELDOUT_SEEDS,
    _evaluate,
    _examples_from_records,
    _leakage_scan,
    _action_mask_token,
    _predict_action_only,
    _predict_action_order,
    _recent_actions,
    _records_from_datasets,
    _resolve_trajectories,
    _sequence_counts,
)

MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION = (
    "mind_v3_v139_sequence_history_shadow_scorer_v1"
)
MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_POLICY = (
    "diagnostics_only_v139_public_sequence_history_shadow_scorer_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v139-sequence-history-shadow-scorer.json"
)

MIN_STRICT_ACCURACY = 0.372481
MIN_STRICT_DELTA = 0.189889
MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.309851
READY_V138_CLASSIFICATION = (
    "rollout_sequence_strict_seed_support_ready_for_future_sequence_world_model_scorer_diagnostic"
)
FORBIDDEN_SCORE_FEATURE_TOKENS = (
    "seed",
    "path",
    "provenance",
    "fixture",
    "future",
    "private",
    "world",
    "outcome",
)


class SequenceHistoryShadowScorerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SequenceHistoryShadowScorerBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class SequenceHistoryShadowScorer:
    artifact: Mapping[str, object]

    def score(
        self,
        *,
        sequence_keys: Sequence[str],
        valid_actions: Sequence[str],
    ) -> dict[str, object]:
        valid = tuple(action for action in ACTION_NAMES if action in set(valid_actions))
        valid = valid or ACTION_NAMES
        lookup = _mapping(self.artifact.get("sequence_lookup"))
        sequence_counts = _mapping(lookup.get("sequence_counts"))
        for key_index, key in enumerate(sequence_keys):
            counts = _action_counts_from_mapping(sequence_counts.get(str(key)))
            action = _best_counted_action(counts, set(valid))
            if action is not None:
                return _score_result(
                    predicted_action=action,
                    scores=counts,
                    source="sequence_key",
                    source_key=str(key),
                    source_key_index=key_index,
                    valid_actions=valid,
                )
        backoff = _mapping(self.artifact.get("backoff"))
        mask_counts = _mapping(backoff.get("mask_action_counts"))
        mask_token = _action_mask_token(valid)
        counts = _action_counts_from_mapping(mask_counts.get(mask_token))
        action = _best_counted_action(counts, set(valid))
        if action is not None:
            return _score_result(
                predicted_action=action,
                scores=counts,
                source="train_mask_action_counts",
                source_key=mask_token,
                source_key_index=None,
                valid_actions=valid,
            )
        action_counts = _action_counts_from_mapping(backoff.get("action_counts"))
        action = _best_counted_action(action_counts, set(valid))
        if action is not None:
            return _score_result(
                predicted_action=action,
                scores=action_counts,
                source="train_action_counts",
                source_key=None,
                source_key_index=None,
                valid_actions=valid,
            )
        return _score_result(
            predicted_action=None,
            scores=Counter(),
            source="no_data_supported_action",
            source_key=None,
            source_key_index=None,
            valid_actions=valid,
        )

    def predict(
        self,
        *,
        sequence_keys: Sequence[str],
        valid_actions: Sequence[str],
    ) -> object:
        return self.score(
            sequence_keys=sequence_keys,
            valid_actions=valid_actions,
        )["predicted_action"]


def sequence_history_shadow_sequence_keys(
    *,
    state: RolloutContextState,
    valid_action_mask: Mapping[str, bool] | Sequence[str],
) -> tuple[str, ...]:
    valid_actions = _valid_actions_from_mask(valid_action_mask)
    mask = _action_mask_token(valid_actions)
    snapshot = state.snapshot()
    recent_resolved = _recent_actions(snapshot.get("recent_resolved_actions"))
    recent_requested = _recent_actions(snapshot.get("recent_requested_actions"))
    last_resolved = recent_resolved[-1] if recent_resolved else "none"
    last_requested = recent_requested[-1] if recent_requested else "none"
    return (
        f"mask={mask}|ctx={state.context_key()}",
        f"mask={mask}|coarse={state.coarse_context_key()}",
        f"mask={mask}|last_resolved={last_resolved}|last_requested={last_requested}",
        f"mask={mask}|last_resolved={last_resolved}",
        f"mask={mask}|history_any={bool(recent_resolved or recent_requested)}",
    )


def build_sequence_history_shadow_scorer_report(
    *,
    v138_report: Mapping[str, object] | None = None,
    v138_report_path: str | Path | None = DEFAULT_V138_REPORT_PATH,
    train_trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    strict_heldout_trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    train_trajectory_paths: Sequence[str | Path] = (),
    strict_heldout_trajectory_paths: Sequence[str | Path] = (),
    strict_seed_values: Sequence[int] = STRICT_HELDOUT_SEEDS,
    context_config: RolloutContextConfig | None = None,
) -> SequenceHistoryShadowScorerBuild:
    strict_seeds = tuple(sorted({int(seed) for seed in strict_seed_values}))
    if not strict_seeds:
        raise SequenceHistoryShadowScorerError("strict_seed_values must not be empty")
    config = context_config or RolloutContextConfig()
    v138_payload, v138_evidence = _resolve_json_report(
        v138_report,
        v138_report_path,
        expected_schema=MIND_V3_ROLLOUT_SEQUENCE_STRICT_SEED_SUPPORT_RECHECK_SCHEMA_VERSION,
    )
    train = _resolve_trajectories(
        trajectory_datasets=train_trajectory_datasets,
        trajectory_paths=train_trajectory_paths,
        report_paths=(),
    )
    strict_heldout = _resolve_trajectories(
        trajectory_datasets=strict_heldout_trajectory_datasets,
        trajectory_paths=strict_heldout_trajectory_paths,
        report_paths=(),
    )
    datasets = (*train.datasets, *strict_heldout.datasets)
    source_roles = _source_role_summary(
        train.datasets,
        strict_heldout.datasets,
        strict_seed_values=strict_seeds,
    )
    records = _records_from_datasets(datasets)
    leakage_scan = _leakage_scan(records)
    examples = _examples_from_records(
        records,
        config=config,
        heldout_seed_values=strict_seeds,
        heldout_source_patterns=(),
        heldout_fraction=0.0,
    )
    train_examples = tuple(example for example in examples if example.split == "train")
    strict_examples = _strict_heldout_examples(
        examples,
        source_seed_sets=_mapping(source_roles.get("source_seed_sets_by_path")),
        strict_seed_values=strict_seeds,
    )
    artifact = _build_artifact(
        train_examples=train_examples,
        config=config,
        strict_seed_values=strict_seeds,
    )
    artifact_feature_leakage_scan = _artifact_feature_leakage_scan(artifact)
    pre_scorer = SequenceHistoryShadowScorer(artifact=artifact)
    loaded_artifact = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    loaded_scorer = load_sequence_history_shadow_scorer_artifact(loaded_artifact)
    roundtrip = _roundtrip_check(
        pre_scorer=pre_scorer,
        loaded_scorer=loaded_scorer,
        examples=strict_examples,
    )
    evaluation = _evaluate_shadow_scorer(
        scorer=loaded_scorer,
        train_examples=train_examples,
        strict_examples=strict_examples,
        source_seed_sets=_mapping(source_roles.get("source_seed_sets_by_path")),
        strict_seed_values=strict_seeds,
    )
    source_integrity = _source_integrity(
        v138_report=v138_payload,
        v138_evidence=v138_evidence,
        train_evidence=train.evidence,
        strict_heldout_evidence=strict_heldout.evidence,
        source_roles=source_roles,
        leakage_scan=leakage_scan,
        artifact_feature_leakage_scan=artifact_feature_leakage_scan,
    )
    support_floors = _support_floors(
        source_integrity=source_integrity,
        evaluation=evaluation,
        roundtrip=roundtrip,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
    )
    contract = _contract(config=config, strict_seed_values=strict_seeds)
    report = {
        "schema_version": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
        "audit_policy": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_POLICY,
        "contract": contract,
        "provenance": {
            "contract_digest": stable_payload_digest(contract),
            "artifact_digest": stable_payload_digest(artifact),
        },
        "source_integrity": source_integrity,
        "source_evidence": {
            "v138_report": v138_evidence,
            "train_trajectories": train.evidence,
            "strict_heldout_trajectories": strict_heldout.evidence,
        },
        "strict_seed_source_integrity": source_roles,
        "leakage_scan": leakage_scan,
        "artifact_feature_leakage_scan": artifact_feature_leakage_scan,
        "artifact": artifact,
        "diagnostic_artifact_created": True,
        "artifact_roundtrip": roundtrip,
        "strict_heldout_metrics": evaluation,
        "support_floors": support_floors,
        "classification": classification,
        "authorization_block": _authorization_block(),
        "next_step": _next_step(classification),
        "non_promoted": True,
    }
    return SequenceHistoryShadowScorerBuild(report=report)


def write_sequence_history_shadow_scorer_report(
    build: SequenceHistoryShadowScorerBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_sequence_history_shadow_scorer_artifact(
    payload_or_path: Mapping[str, object] | str | Path,
) -> SequenceHistoryShadowScorer:
    if isinstance(payload_or_path, (str, Path)):
        with Path(payload_or_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = dict(payload_or_path)
    if not isinstance(payload, Mapping):
        raise SequenceHistoryShadowScorerError("artifact payload must be a JSON object")
    artifact = payload.get("artifact") if "artifact" in payload else payload
    if not isinstance(artifact, Mapping):
        raise SequenceHistoryShadowScorerError("missing artifact object")
    if artifact.get("schema_version") != MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION:
        raise SequenceHistoryShadowScorerError("sequence history artifact schema mismatch")
    return SequenceHistoryShadowScorer(artifact=dict(artifact))


def _contract(
    *,
    config: RolloutContextConfig,
    strict_seed_values: Sequence[int],
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "report_only": True,
        "runtime_policy_effect": "none",
        "trainer_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "training_authorized": False,
        "training_executed": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "runtime_promotion_authorized": False,
        "runtime_action_selection_changed": False,
        "scorer_runtime_selection_authorized": False,
        "strict_seed_values": list(strict_seed_values),
        "train_source_role": "v98_numeric_train_bank_only",
        "heldout_source_role": "strict_seed_evaluation_only",
        "sequence_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
        "rollout_context": rollout_context_feature_contract(config),
        "current_decision_inputs": ["action_mask"],
        "prior_history_inputs": [
            "same-agent finalized public trajectory rows before current decision"
        ],
        "excluded_score_features": [
            "seed",
            "path",
            "provenance",
            "fixture identity",
            "current-row outcome",
            "future rows",
            "private world state",
            "replay result fields",
        ],
    }


def _build_artifact(
    *,
    train_examples: Sequence[object],
    config: RolloutContextConfig,
    strict_seed_values: Sequence[int],
) -> dict[str, object]:
    sequence_counts = _sequence_counts(train_examples)
    action_counts = Counter(example.label for example in train_examples)
    mask_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in train_examples:
        mask_counts[_action_mask_token(example.valid_actions)][example.label] += 1
    return {
        "schema_version": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
        "artifact_policy": MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_POLICY,
        "model_id": SEQUENCE_MODEL_ID,
        "diagnostics_only": True,
        "runtime_action_selection_authorized": False,
        "runtime_policy_change_authorized": False,
        "training_authorized": False,
        "built_from": {
            "source_role": "v98_numeric_train_bank_only",
            "train_example_count": len(train_examples),
            "strict_seed_values_excluded_from_training": list(strict_seed_values),
        },
        "feature_policy": {
            "sequence_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "rollout_context": rollout_context_feature_contract(config),
            "current_decision_inputs": ["action_mask"],
            "prior_history_inputs": [
                "same-agent finalized public trajectory rows before current decision"
            ],
            "excluded_score_features": [
                "seed",
                "path",
                "provenance",
                "fixture identity",
                "current-row outcome",
                "future rows",
                "private world state",
                "replay result fields",
            ],
        },
        "sequence_lookup": {
            "key_order_policy": "v137_sequence_key_order_v1",
            "count_policy": "train_label_counts_by_public_sequence_key_v1",
            "sequence_counts": {
                key: _ordered_counts(Counter(counts))
                for key, counts in sorted(sequence_counts.items())
            },
        },
        "backoff": {
            "policy": "train_mask_counts_then_train_action_counts_no_action_order_fallback_v1",
            "mask_action_counts": {
                key: _ordered_counts(Counter(counts))
                for key, counts in sorted(mask_counts.items())
            },
            "action_counts": _ordered_counts(action_counts),
        },
    }


def _valid_actions_from_mask(
    valid_action_mask: Mapping[str, bool] | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(valid_action_mask, Mapping):
        return tuple(
            action for action in ACTION_NAMES if bool(valid_action_mask.get(action))
        )
    if isinstance(valid_action_mask, (str, bytes)):
        return ()
    valid = set(valid_action_mask)
    return tuple(action for action in ACTION_NAMES if action in valid)


def _strict_heldout_examples(
    examples: Sequence[object],
    *,
    source_seed_sets: Mapping[str, object],
    strict_seed_values: Sequence[int],
) -> tuple[object, ...]:
    strict = set(int(seed) for seed in strict_seed_values)
    return tuple(
        example
        for example in examples
        if example.split == "heldout"
        and set(_seed_values(source_seed_sets.get(example.source_path))).intersection(
            strict
        )
    )


def _evaluate_shadow_scorer(
    *,
    scorer: SequenceHistoryShadowScorer,
    train_examples: Sequence[object],
    strict_examples: Sequence[object],
    source_seed_sets: Mapping[str, object],
    strict_seed_values: Sequence[int],
) -> dict[str, object]:
    action_counts = Counter(example.label for example in train_examples)
    sequence_eval = _evaluate(
        strict_examples,
        predictor=lambda example: scorer.predict(
            sequence_keys=example.sequence_keys,
            valid_actions=example.valid_actions,
        ),
    )
    action_only_eval = _evaluate(
        strict_examples,
        predictor=lambda example: _predict_action_only(
            example,
            action_counts=action_counts,
        ),
    )
    action_order_eval = _evaluate(strict_examples, predictor=_predict_action_order)
    per_seed: dict[str, object] = {}
    counts_by_seed: dict[int, int] = {}
    for seed in sorted({int(seed) for seed in strict_seed_values}):
        seed_examples = tuple(
            example
            for example in strict_examples
            if seed in set(_seed_values(source_seed_sets.get(example.source_path)))
        )
        seed_eval = _evaluate(
            seed_examples,
            predictor=lambda example: scorer.predict(
                sequence_keys=example.sequence_keys,
                valid_actions=example.valid_actions,
            ),
        )
        seed_action_only = _evaluate(
            seed_examples,
            predictor=lambda example: _predict_action_only(
                example,
                action_counts=action_counts,
            ),
        )
        seed_action_order = _evaluate(seed_examples, predictor=_predict_action_order)
        counts_by_seed[seed] = _int(seed_eval.get("heldout_record_count"))
        per_seed[str(seed)] = {
            "heldout_record_count": seed_eval.get("heldout_record_count"),
            "accuracy": seed_eval.get("accuracy"),
            "correct_count": seed_eval.get("correct_count"),
            "action_only_accuracy": seed_action_only.get("accuracy"),
            "action_order_accuracy": seed_action_order.get("accuracy"),
            "accuracy_delta_vs_action_only": _round(
                _float(seed_eval.get("accuracy"))
                - _float(seed_action_only.get("accuracy"))
            ),
            "accuracy_delta_vs_action_order": _round(
                _float(seed_eval.get("accuracy"))
                - _float(seed_action_order.get("accuracy"))
            ),
            "dominant_predicted_action": seed_eval.get("dominant_predicted_action"),
            "dominant_predicted_action_share": seed_eval.get(
                "dominant_predicted_action_share"
            ),
            "unsupported_action_count": seed_eval.get("unsupported_action_count"),
        }
    missing = [seed for seed, count in counts_by_seed.items() if count <= 0]
    return {
        "policy": "v139_loaded_shadow_scorer_strict_seed_metrics_v1",
        "model_summaries": {
            SEQUENCE_MODEL_ID: sequence_eval,
            ACTION_ONLY_BASELINE_ID: action_only_eval,
            ACTION_ORDER_BASELINE_ID: action_order_eval,
        },
        "baseline_comparisons": {
            "strict_heldout_accuracy_delta_vs_action_only": _round(
                _float(sequence_eval.get("accuracy"))
                - _float(action_only_eval.get("accuracy"))
            ),
            "strict_heldout_accuracy_delta_vs_action_order": _round(
                _float(sequence_eval.get("accuracy"))
                - _float(action_order_eval.get("accuracy"))
            ),
            "beats_action_only_baseline": _float(sequence_eval.get("accuracy"))
            > _float(action_only_eval.get("accuracy")),
            "beats_action_order_baseline": _float(sequence_eval.get("accuracy"))
            > _float(action_order_eval.get("accuracy")),
        },
        "per_strict_seed": per_seed,
        "strict_seed_evaluated_record_counts": {
            str(seed): count for seed, count in counts_by_seed.items()
        },
        "strict_seed_missing_evaluated_record_seeds": missing,
        "all_strict_seeds_have_evaluated_records": not missing,
    }


def _roundtrip_check(
    *,
    pre_scorer: SequenceHistoryShadowScorer,
    loaded_scorer: SequenceHistoryShadowScorer,
    examples: Sequence[object],
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    for example in examples:
        pre = pre_scorer.score(
            sequence_keys=example.sequence_keys,
            valid_actions=example.valid_actions,
        )
        loaded = loaded_scorer.score(
            sequence_keys=example.sequence_keys,
            valid_actions=example.valid_actions,
        )
        if pre != loaded:
            mismatches.append(
                {
                    "row_id": example.row_id,
                    "pre_serialization": pre,
                    "loaded": loaded,
                }
            )
    return {
        "policy": "json_serialization_roundtrip_exact_score_match_v1",
        "loaded_artifact_scores_match_pre_serialization": not mismatches,
        "checked_example_count": len(examples),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:12],
    }


def _source_integrity(
    *,
    v138_report: Mapping[str, object] | None,
    v138_evidence: Mapping[str, object],
    train_evidence: Mapping[str, object],
    strict_heldout_evidence: Mapping[str, object],
    source_roles: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    artifact_feature_leakage_scan: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if v138_evidence.get("loaded") is not True:
        failures.append("missing_v138_report")
    elif v138_evidence.get("schema_matches") is not True:
        failures.append("v138_schema_mismatch")
    if _mapping(_mapping(v138_report).get("source_integrity")).get("passed") is not True:
        failures.append("v138_source_integrity_not_passed")
    if _mapping(_mapping(v138_report).get("support_floors")).get("passed") is not True:
        failures.append("v138_support_floors_not_passed")
    if _mapping(_mapping(v138_report).get("classification")).get("primary") != (
        READY_V138_CLASSIFICATION
    ):
        failures.append("v138_classification_not_ready")
    if _int(train_evidence.get("loaded_path_count")) <= 0:
        failures.append("no_train_trajectory_inputs_loaded")
    if _int(strict_heldout_evidence.get("loaded_path_count")) <= 0:
        failures.append("no_strict_heldout_trajectory_inputs_loaded")
    if _int(train_evidence.get("load_failure_count")) > 0:
        failures.append("train_trajectory_load_failures")
    if _int(strict_heldout_evidence.get("load_failure_count")) > 0:
        failures.append("strict_heldout_trajectory_load_failures")
    if _int(source_roles.get("strict_seed_train_source_leakage_count")) > 0:
        failures.append("train_source_seed_set_intersects_strict_seeds")
    if _int(source_roles.get("source_path_overlap_count")) > 0:
        failures.append("train_strict_heldout_source_path_overlap")
    if source_roles.get("strict_seed_presence_only_heldout") is not True:
        failures.append("strict_seeds_not_present_only_in_heldout")
    if leakage_scan.get("passed") is not True:
        failures.append("leakage_scan_failed")
    if artifact_feature_leakage_scan.get("passed") is not True:
        failures.append("artifact_feature_leakage_scan_failed")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v138_source_integrity_passed": _mapping(
            _mapping(v138_report).get("source_integrity")
        ).get("passed"),
        "v138_support_floors_passed": _mapping(
            _mapping(v138_report).get("support_floors")
        ).get("passed"),
        "v138_classification": _mapping(
            _mapping(v138_report).get("classification")
        ).get("primary"),
        "required_v138_classification": READY_V138_CLASSIFICATION,
        "train_loaded_path_count": train_evidence.get("loaded_path_count"),
        "strict_heldout_loaded_path_count": strict_heldout_evidence.get(
            "loaded_path_count"
        ),
        "strict_seed_train_source_leakage_count": source_roles.get(
            "strict_seed_train_source_leakage_count"
        ),
        "source_path_overlap_count": source_roles.get("source_path_overlap_count"),
        "source_path_overlap_examples": source_roles.get(
            "source_path_overlap_examples"
        ),
        "strict_seed_presence_only_heldout": source_roles.get(
            "strict_seed_presence_only_heldout"
        ),
        "leakage_scan_passed": leakage_scan.get("passed"),
        "artifact_feature_leakage_scan_passed": artifact_feature_leakage_scan.get(
            "passed"
        ),
    }


def _support_floors(
    *,
    source_integrity: Mapping[str, object],
    evaluation: Mapping[str, object],
    roundtrip: Mapping[str, object],
) -> dict[str, object]:
    summaries = _mapping(evaluation.get("model_summaries"))
    sequence = _mapping(summaries.get(SEQUENCE_MODEL_ID))
    comparisons = _mapping(evaluation.get("baseline_comparisons"))
    floors = [
        _floor(
            "v138_source_integrity_passed",
            source_integrity.get("v138_source_integrity_passed") is True,
            observed=source_integrity.get("v138_source_integrity_passed"),
            required=True,
        ),
        _floor(
            "v138_support_floors_passed",
            source_integrity.get("v138_support_floors_passed") is True,
            observed=source_integrity.get("v138_support_floors_passed"),
            required=True,
        ),
        _floor(
            "v138_classification_ready",
            source_integrity.get("v138_classification") == READY_V138_CLASSIFICATION,
            observed=source_integrity.get("v138_classification"),
            required=READY_V138_CLASSIFICATION,
        ),
        _floor(
            "source_integrity_passed",
            source_integrity.get("passed") is True,
            observed=source_integrity.get("passed"),
            required=True,
        ),
        _floor(
            "train_source_seed_set_intersection_count_eq_0",
            _int(source_integrity.get("strict_seed_train_source_leakage_count")) == 0,
            observed=source_integrity.get("strict_seed_train_source_leakage_count"),
            required=0,
        ),
        _floor(
            "train_strict_heldout_source_path_overlap_count_eq_0",
            _int(source_integrity.get("source_path_overlap_count")) == 0,
            observed=source_integrity.get("source_path_overlap_count"),
            required=0,
        ),
        _floor(
            "all_strict_seeds_have_evaluated_records",
            evaluation.get("all_strict_seeds_have_evaluated_records") is True,
            observed=evaluation.get("strict_seed_evaluated_record_counts"),
            required="heldout_record_count>0 for every configured strict seed",
        ),
        _floor(
            "loaded_artifact_scores_match_pre_serialization",
            roundtrip.get("loaded_artifact_scores_match_pre_serialization") is True,
            observed=roundtrip.get("mismatch_count"),
            required=0,
        ),
        _floor(
            "artifact_feature_leakage_scan_passed",
            source_integrity.get("artifact_feature_leakage_scan_passed") is True,
            observed=source_integrity.get("artifact_feature_leakage_scan_passed"),
            required=True,
        ),
        _floor(
            "strict_heldout_accuracy_gte_v138",
            _float(sequence.get("accuracy")) >= MIN_STRICT_ACCURACY,
            observed=sequence.get("accuracy"),
            required=MIN_STRICT_ACCURACY,
        ),
        _floor(
            "strict_delta_vs_action_only_gte_v138",
            _float(comparisons.get("strict_heldout_accuracy_delta_vs_action_only"))
            >= MIN_STRICT_DELTA,
            observed=comparisons.get("strict_heldout_accuracy_delta_vs_action_only"),
            required=MIN_STRICT_DELTA,
        ),
        _floor(
            "strict_delta_vs_action_order_gte_v138",
            _float(comparisons.get("strict_heldout_accuracy_delta_vs_action_order"))
            >= MIN_STRICT_DELTA,
            observed=comparisons.get("strict_heldout_accuracy_delta_vs_action_order"),
            required=MIN_STRICT_DELTA,
        ),
        _floor(
            "dominant_predicted_action_share_lte_v138",
            _float(sequence.get("dominant_predicted_action_share"))
            <= MAX_DOMINANT_PREDICTED_ACTION_SHARE,
            observed=sequence.get("dominant_predicted_action_share"),
            required=MAX_DOMINANT_PREDICTED_ACTION_SHARE,
        ),
        _floor(
            "dominant_predicted_action_share_lte_0_50",
            _float(sequence.get("dominant_predicted_action_share"))
            <= DOMINANT_PREDICTED_ACTION_SHARE_MAX,
            observed=sequence.get("dominant_predicted_action_share"),
            required=DOMINANT_PREDICTED_ACTION_SHARE_MAX,
        ),
        _floor(
            "unsupported_action_count_eq_0",
            _int(sequence.get("unsupported_action_count")) == 0,
            observed=sequence.get("unsupported_action_count"),
            required=0,
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "v139_sequence_history_shadow_scorer_floors_v1",
        "passed": first_failed is None,
        "first_failed_floor": None if first_failed is None else first_failed["name"],
        "floors": floors,
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    support_floors: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        primary = "sequence_history_shadow_scorer_source_integrity_failed"
    elif support_floors.get("passed") is True:
        primary = "sequence_history_shadow_scorer_passed_diagnostics_only_ready_for_shadow_runtime_logging"
    else:
        primary = "sequence_history_shadow_scorer_blocked"
    return {
        "primary": primary,
        "labels": [primary],
        "first_failed_floor": support_floors.get("first_failed_floor"),
        "allowed_classifications": [
            "sequence_history_shadow_scorer_source_integrity_failed",
            "sequence_history_shadow_scorer_blocked",
            "sequence_history_shadow_scorer_passed_diagnostics_only_ready_for_shadow_runtime_logging",
        ],
    }


def _next_step(classification: Mapping[str, object]) -> str:
    if (
        classification.get("primary")
        == "sequence_history_shadow_scorer_passed_diagnostics_only_ready_for_shadow_runtime_logging"
    ):
        return "shadow_runtime_integration_logging_only_no_action_selection_change"
    return "stop_do_not_add_another_diagnostic_until_blocker_is_resolved"


def _authorization_block() -> dict[str, object]:
    return {
        "training_authorized": False,
        "runtime_policy_change_authorized": False,
        "shadow_scorer_execution_authorized": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "runtime_promotion_authorized": False,
        "runtime_action_selection_authorized": False,
        "gate_change_authorized": False,
        "viewer_change_authorized": False,
        "replay_golden_change_authorized": False,
        "foundation_change_authorized": False,
    }


def _score_result(
    *,
    predicted_action: str | None,
    scores: Counter[str],
    source: str,
    source_key: str | None,
    source_key_index: int | None,
    valid_actions: Sequence[str],
) -> dict[str, object]:
    valid_set = set(valid_actions)
    valid_action_scores = {
        action: int(scores.get(action, 0))
        for action in ACTION_NAMES
        if action in valid_set
    }
    return {
        "predicted_action": predicted_action,
        "score_source": source,
        "source_key": source_key,
        "source_key_index": source_key_index,
        "valid_actions": list(valid_actions),
        "scores": valid_action_scores,
        "valid_action_scores": valid_action_scores,
        "raw_train_counts": _ordered_counts(scores),
        "supported_prediction": predicted_action is not None,
    }


def _best_counted_action(
    counts: Counter[str],
    valid_actions: set[str],
) -> str | None:
    best_action: str | None = None
    best_count = -1
    for action in ACTION_NAMES:
        if action not in valid_actions:
            continue
        count = int(counts.get(action, 0))
        if count > best_count:
            best_count = count
            best_action = action
    if best_action is None or best_count <= 0:
        return None
    return best_action


def _action_counts_from_mapping(value: object) -> Counter[str]:
    mapping = _mapping(value)
    return Counter({action: _int(mapping.get(action)) for action in ACTION_NAMES})


def _ordered_counts(counts: Counter[str]) -> dict[str, int]:
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


def _artifact_feature_leakage_scan(artifact: Mapping[str, object]) -> dict[str, object]:
    leaks: list[dict[str, object]] = []
    sequence_counts = _mapping(_mapping(artifact.get("sequence_lookup")).get("sequence_counts"))
    for key in sorted(sequence_counts):
        lowered = str(key).lower()
        for token in FORBIDDEN_SCORE_FEATURE_TOKENS:
            if token in lowered:
                leaks.append(
                    {
                        "feature": "sequence_lookup.sequence_counts",
                        "key": str(key),
                        "token": token,
                    }
                )
    return {
        "policy": "v139_serialized_score_feature_leakage_scan_v1",
        "passed": not leaks,
        "forbidden_score_feature_tokens": list(FORBIDDEN_SCORE_FEATURE_TOKENS),
        "leakage_count": len(leaks),
        "leaks": leaks[:24],
    }


def _seed_values(value: object) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seeds: list[int] = []
        for item in value:
            if isinstance(item, int) and not isinstance(item, bool):
                seeds.append(int(item))
        return tuple(seeds)
    return ()


def _floor(
    name: str,
    passed: bool,
    *,
    observed: object,
    required: object,
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _round(value: float) -> float:
    return round(float(value), 6)
