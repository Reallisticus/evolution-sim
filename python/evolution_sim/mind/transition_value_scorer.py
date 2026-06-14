from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)
from evolution_sim.mind.dataset import (
    TRAJECTORY_DATASET_RECORD_INDEX_FIELD,
    TRAJECTORY_EPISODE_ID_FIELD,
    TRAJECTORY_SOURCE_PATH_FIELD,
    TrajectoryJsonlDataset,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.rollout_context import (
    MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
    RolloutContextConfig,
    RolloutContextState,
    rollout_context_feature_contract,
)
from evolution_sim.mind.rollout_sequence_strict_seed_support_recheck import (
    _source_role_summary,
)
from evolution_sim.mind.rollout_sequence_support_audit import (
    STRICT_HELDOUT_SEEDS,
    _action_mask_token,
    _records_from_datasets,
    _resolve_trajectories,
)

MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION = (
    "mind_v3_v142_public_transition_value_scorer_v1"
)
MIND_V3_TRANSITION_VALUE_SCORER_POLICY = (
    "opt_in_v142_public_transition_value_world_model_scorer_v1"
)
MIND_V3_TRANSITION_VALUE_MODEL_ID = "public_transition_value_utility_lookup"
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v142-transition-value-scorer.json"
)

UTILITY_COMPONENT_WEIGHTS = {
    "alive_continuation": 0.35,
    "birth_reproduction": 1.25,
    "reproduction_readiness": 0.25,
    "death_risk": 1.4,
    "energy_delta": 0.5,
    "hydration_delta": 0.55,
    "health_delta": 0.65,
    "unsupported_action_rejection": 1.0,
}
FORBIDDEN_SCORE_FEATURE_TOKENS = (
    "seed",
    "fixture",
    "private",
    "future",
    "heuristic",
    "world",
    "provenance",
)
REQUIRED_RECORD_FIELDS = (
    "observation_input",
    "action_mask",
    "requested_action",
    "resolved_action",
    "action_source",
    "action_valid",
    "resolution_action_valid",
    "before",
    "after",
    "outcome",
    "reward",
)
TRANSITION_VALUE_LOW_SPECIFICITY_SOURCE_KEY_CATEGORIES = frozenset(
    {
        "mask_only_hit",
        "coarse_feature_hit",
        "self_feature_hit",
        "self_coarse_feature_hit",
    }
)
TRANSITION_VALUE_HIGH_SPECIFICITY_SOURCE_KEY_CATEGORIES = frozenset(
    {
        "exact_context_feature_hit",
        "self_nav_coarse_feature_hit",
        "self_nav_feature_hit",
        "nav_feature_hit",
    }
)
TRANSITION_VALUE_DEFAULT_ALLOWED_SOURCE_KEY_CATEGORIES = (
    TRANSITION_VALUE_HIGH_SPECIFICITY_SOURCE_KEY_CATEGORIES
)


def transition_value_source_key_category(
    score_source: str,
    source_key: object,
) -> str:
    if score_source != "feature_action_utility":
        return "miss"
    key = str(source_key or "")
    if "|ctx=" in key:
        return "exact_context_feature_hit"
    if "|coarse=" in key and "|self=" in key and "|nav=" in key:
        return "self_nav_coarse_feature_hit"
    if "|self=" in key and "|nav=" in key:
        return "self_nav_feature_hit"
    if "|self=" in key and "|coarse=" in key:
        return "self_coarse_feature_hit"
    if "|self=" in key:
        return "self_feature_hit"
    if "|nav=" in key:
        return "nav_feature_hit"
    if "|coarse=" in key:
        return "coarse_feature_hit"
    if key == "global":
        return "global_hit"
    if key:
        return "mask_only_hit"
    return "missing_source_key"


class TransitionValueScorerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransitionValueScorerBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _UtilityAccumulator:
    count: int = 0
    utility_sum: float = 0.0
    component_sums: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class _TransitionValueExample:
    row_id: str
    action: str
    valid_actions: tuple[str, ...]
    feature_keys: tuple[str, ...]
    utility: float
    utility_components: Mapping[str, float]
    action_source: str
    source_path: str
    split: str


@dataclass(frozen=True, slots=True)
class TransitionValueScorer:
    artifact: Mapping[str, object]

    def score(
        self,
        *,
        observation_input: Mapping[str, object],
        valid_action_mask: Mapping[str, bool] | Sequence[str],
        state: RolloutContextState,
        min_observed_support_count: int = 1,
    ) -> dict[str, object]:
        valid_actions = _valid_actions_from_mask(valid_action_mask)
        if not valid_actions:
            return _score_result(
                predicted_action=None,
                scores={},
                score_counts={},
                score_imputed={},
                source="no_valid_actions",
                source_key=None,
                valid_actions=(),
                min_observed_support_count=min_observed_support_count,
            )
        keys = transition_value_feature_keys(
            observation_input=observation_input,
            valid_action_mask=valid_action_mask,
            state=state,
        )
        tables = _mapping(self.artifact.get("utility_tables"))
        feature_table = _mapping(tables.get("feature_action_utility"))
        for key in keys:
            scored = _scores_from_action_stats(
                feature_table.get(str(key)),
                valid_actions=valid_actions,
            )
            if scored is not None:
                return _score_result(
                    predicted_action=scored["predicted_action"],
                    scores=scored["scores"],
                    score_counts=scored["score_counts"],
                    score_imputed=scored["score_imputed"],
                    source="feature_action_utility",
                    source_key=str(key),
                    valid_actions=valid_actions,
                    min_observed_support_count=min_observed_support_count,
                )
        return _score_result(
            predicted_action=None,
            scores={},
            score_counts={},
            score_imputed={},
            source="missing_supported_scores_for_valid_actions",
            source_key=None,
            valid_actions=valid_actions,
            min_observed_support_count=min_observed_support_count,
        )

    def candidate_key_coverage(
        self,
        *,
        observation_input: Mapping[str, object],
        valid_action_mask: Mapping[str, bool] | Sequence[str],
        state: RolloutContextState,
        min_observed_support_count: int = 1,
    ) -> list[dict[str, object]]:
        valid_actions = _valid_actions_from_mask(valid_action_mask)
        if not valid_actions:
            return []
        keys = transition_value_feature_keys(
            observation_input=observation_input,
            valid_action_mask=valid_action_mask,
            state=state,
        )
        tables = _mapping(self.artifact.get("utility_tables"))
        feature_table = _mapping(tables.get("feature_action_utility"))
        coverage: list[dict[str, object]] = []
        for index, key in enumerate(keys):
            key_text = str(key)
            coverage.append(
                _candidate_key_coverage_result(
                    rank=index,
                    source_key=key_text,
                    key_present=key_text in feature_table,
                    value=feature_table.get(key_text),
                    valid_actions=valid_actions,
                    min_observed_support_count=min_observed_support_count,
                )
            )
        return coverage

    def predict(
        self,
        *,
        observation_input: Mapping[str, object],
        valid_action_mask: Mapping[str, bool] | Sequence[str],
        state: RolloutContextState,
    ) -> object:
        return self.score(
            observation_input=observation_input,
            valid_action_mask=valid_action_mask,
            state=state,
        )["predicted_action"]


def transition_value_feature_keys(
    *,
    observation_input: Mapping[str, object],
    valid_action_mask: Mapping[str, bool] | Sequence[str],
    state: RolloutContextState,
) -> tuple[str, ...]:
    valid_actions = _valid_actions_from_mask(valid_action_mask)
    mask = _action_mask_token(valid_actions)
    values = _observation_values(observation_input)
    self_key = _self_feature_key(values)
    nav_key = _navigation_feature_key(values)
    return (
        f"mask={mask}|self={self_key}|nav={nav_key}|ctx={state.context_key()}",
        f"mask={mask}|self={self_key}|nav={nav_key}|coarse={state.coarse_context_key()}",
        f"mask={mask}|self={self_key}|nav={nav_key}",
        f"mask={mask}|self={self_key}|coarse={state.coarse_context_key()}",
        f"mask={mask}|self={self_key}",
        f"mask={mask}|nav={nav_key}",
        f"mask={mask}|coarse={state.coarse_context_key()}",
        f"mask={mask}",
        "global",
    )


def build_transition_value_scorer_report(
    *,
    train_trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    strict_heldout_trajectory_datasets: Sequence[TrajectoryJsonlDataset] | None = None,
    train_trajectory_paths: Sequence[str | Path] = (),
    strict_heldout_trajectory_paths: Sequence[str | Path] = (),
    strict_seed_values: Sequence[int] = STRICT_HELDOUT_SEEDS,
    context_config: RolloutContextConfig | None = None,
) -> TransitionValueScorerBuild:
    strict_seeds = tuple(sorted({int(seed) for seed in strict_seed_values}))
    if not strict_seeds:
        raise TransitionValueScorerError("strict_seed_values must not be empty")
    config = context_config or RolloutContextConfig()
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
    schema_scan = _schema_scan(records)
    examples, finite_feature_scan = _examples_from_records(
        records,
        config=config,
        source_seed_sets=_mapping(source_roles.get("source_seed_sets_by_path")),
        strict_seed_values=strict_seeds,
    )
    train_examples = tuple(example for example in examples if example.split == "train")
    strict_examples = tuple(
        example for example in examples if example.split == "heldout"
    )
    artifact = _build_artifact(
        train_examples=train_examples,
        config=config,
        strict_seed_values=strict_seeds,
    )
    artifact_feature_leakage_scan = _artifact_feature_leakage_scan(artifact)
    pre_scorer = TransitionValueScorer(artifact=artifact)
    loaded_artifact = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    loaded_scorer = load_transition_value_scorer_artifact(loaded_artifact)
    roundtrip = _roundtrip_check(
        pre_scorer=pre_scorer,
        loaded_scorer=loaded_scorer,
        examples=strict_examples,
    )
    evaluation = _evaluate_scorer(
        scorer=loaded_scorer,
        train_examples=train_examples,
        strict_examples=strict_examples,
    )
    action_support = _action_support_summary(train_examples)
    source_integrity = _source_integrity(
        train_evidence=train.evidence,
        strict_heldout_evidence=strict_heldout.evidence,
        source_roles=source_roles,
        schema_scan=schema_scan,
        finite_feature_scan=finite_feature_scan,
        artifact_feature_leakage_scan=artifact_feature_leakage_scan,
        action_support=action_support,
    )
    support_floors = _support_floors(
        source_integrity=source_integrity,
        finite_feature_scan=finite_feature_scan,
        artifact_feature_leakage_scan=artifact_feature_leakage_scan,
        action_support=action_support,
        roundtrip=roundtrip,
        evaluation=evaluation,
    )
    classification = _classification(
        source_integrity=source_integrity,
        support_floors=support_floors,
    )
    contract = _contract(config=config, strict_seed_values=strict_seeds)
    report = {
        "schema_version": MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        "audit_policy": MIND_V3_TRANSITION_VALUE_SCORER_POLICY,
        "contract": contract,
        "provenance": {
            "contract_digest": stable_payload_digest(contract),
            "artifact_digest": stable_payload_digest(artifact),
        },
        "source_integrity": source_integrity,
        "source_evidence": {
            "train_trajectories": train.evidence,
            "strict_heldout_trajectories": strict_heldout.evidence,
        },
        "strict_seed_source_integrity": source_roles,
        "schema_scan": schema_scan,
        "finite_feature_scan": finite_feature_scan,
        "artifact_feature_leakage_scan": artifact_feature_leakage_scan,
        "action_support": action_support,
        "artifact": artifact,
        "artifact_roundtrip": roundtrip,
        "strict_heldout_metrics": evaluation,
        "support_floors": support_floors,
        "classification": classification,
        "non_default_runtime": True,
        "non_promoted": True,
    }
    return TransitionValueScorerBuild(report=report)


def write_transition_value_scorer_report(
    build: TransitionValueScorerBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_transition_value_scorer_artifact(
    payload_or_path: Mapping[str, object] | str | Path,
) -> TransitionValueScorer:
    if isinstance(payload_or_path, (str, Path)):
        with Path(payload_or_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = dict(payload_or_path)
    if not isinstance(payload, Mapping):
        raise TransitionValueScorerError("transition-value payload must be an object")
    artifact = payload.get("artifact") if "artifact" in payload else payload
    if not isinstance(artifact, Mapping):
        raise TransitionValueScorerError("missing transition-value artifact object")
    if artifact.get("schema_version") != MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION:
        raise TransitionValueScorerError("transition-value artifact schema mismatch")
    tables = _mapping(artifact.get("utility_tables"))
    if not _mapping(tables.get("feature_action_utility")):
        raise TransitionValueScorerError("transition-value artifact has no utility table")
    return TransitionValueScorer(artifact=dict(artifact))


def _contract(
    *,
    config: RolloutContextConfig,
    strict_seed_values: Sequence[int],
) -> dict[str, object]:
    return {
        "default_runtime_behavior_changed": False,
        "explicit_opt_in_required": True,
        "explicit_opt_in_live_ab_eligible": True,
        "runtime_policy_effect": "none unless --transition-value-action-override",
        "runtime_policy_change_requires_explicit_flag": True,
        "trainer_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "model_artifact_created": True,
        "runtime_loadable_artifact_created": True,
        "runtime_promotion_authorized": False,
        "promotion_authorized": False,
        "strict_seed_values": list(strict_seed_values),
        "train_source_role": "public_non_strict_seed_trajectories",
        "heldout_source_role": "strict_seed_evaluation_only",
        "model_id": MIND_V3_TRANSITION_VALUE_MODEL_ID,
        "utility_policy": {
            "target": "observed_public_transition_outcome_utility",
            "component_weights": dict(sorted(UTILITY_COMPONENT_WEIGHTS.items())),
            "estimates_action_utility_not_action_frequency": True,
        },
        "feature_policy": {
            "sequence_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "rollout_context": rollout_context_feature_contract(config),
            "current_decision_inputs": [
                "public observation_input",
                "public action_mask",
            ],
            "prior_history_inputs": [
                "same-agent finalized public trajectory rows before current decision"
            ],
            "excluded_score_features": [
                "seed identity",
                "fixture identity",
                "private world state",
                "future rows at decision time",
                "heuristic recommendations",
                "source path",
                "provenance",
            ],
        },
    }


def _examples_from_records(
    records: Sequence[Mapping[str, object]],
    *,
    config: RolloutContextConfig,
    source_seed_sets: Mapping[str, object],
    strict_seed_values: Sequence[int],
) -> tuple[tuple[_TransitionValueExample, ...], dict[str, object]]:
    states: dict[tuple[str, int], RolloutContextState] = {}
    examples: list[_TransitionValueExample] = []
    failures: list[dict[str, object]] = []
    strict_seeds = {int(seed) for seed in strict_seed_values}
    for index, record in enumerate(records):
        source_path = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
        source_seeds = {
            int(seed)
            for seed in _seed_values(source_seed_sets.get(source_path))
        }
        split = "heldout" if source_seeds.intersection(strict_seeds) else "train"
        agent_id = _agent_id(record)
        episode_id = _episode_id(record, fallback_index=index)
        state = states.setdefault((episode_id, agent_id), RolloutContextState(config))
        if _action_source(record) == "passive":
            state.update_from_record(record)
            continue
        try:
            observation_input = _observation_input(record)
            valid_actions = _valid_actions(record)
            action = _label_action(record)
            if action not in ACTION_NAMES:
                raise TransitionValueScorerError("row action is not in ACTION_NAMES")
            utility_components = _utility_components(record)
            utility = _utility_from_components(utility_components)
            feature_keys = transition_value_feature_keys(
                observation_input=observation_input,
                valid_action_mask=valid_actions,
                state=state,
            )
        except (ValueError, TransitionValueScorerError) as exc:
            failures.append(
                {
                    "row_id": _row_id(record, fallback_index=index),
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            state.update_from_record(record)
            continue
        examples.append(
            _TransitionValueExample(
                row_id=_row_id(record, fallback_index=index),
                action=action,
                valid_actions=valid_actions,
                feature_keys=feature_keys,
                utility=utility,
                utility_components=utility_components,
                action_source=_action_source(record),
                source_path=source_path,
                split=split,
            )
        )
        state.update_from_record(record)
    return tuple(examples), {
        "policy": "v142_public_feature_finiteness_scan_v1",
        "passed": not failures,
        "checked_record_count": len(records),
        "example_count": len(examples),
        "failure_count": len(failures),
        "failures": failures[:24],
    }


def _build_artifact(
    *,
    train_examples: Sequence[_TransitionValueExample],
    config: RolloutContextConfig,
    strict_seed_values: Sequence[int],
) -> dict[str, object]:
    accumulators: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(lambda: {"count": 0, "utility_sum": 0.0, "components": defaultdict(float)})
    )
    for example in train_examples:
        for key in example.feature_keys:
            bucket = accumulators[str(key)][example.action]
            bucket["count"] = int(bucket["count"]) + 1
            bucket["utility_sum"] = float(bucket["utility_sum"]) + float(example.utility)
            components = bucket["components"]
            if isinstance(components, defaultdict):
                for name, value in example.utility_components.items():
                    components[str(name)] += float(value)
    table: dict[str, object] = {}
    for key, action_buckets in sorted(accumulators.items()):
        table[key] = {
            action: _finalize_action_bucket(bucket)
            for action, bucket in sorted(action_buckets.items())
            if action in ACTION_NAMES
        }
    return {
        "schema_version": MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
        "artifact_policy": MIND_V3_TRANSITION_VALUE_SCORER_POLICY,
        "model_id": MIND_V3_TRANSITION_VALUE_MODEL_ID,
        "diagnostics_only": True,
        "explicit_opt_in_live_ab_eligible": True,
        "runtime_action_selection_authorized": False,
        "runtime_policy_change_requires_explicit_flag": True,
        "promotion_authorized": False,
        "built_from": {
            "source_role": "public_non_strict_seed_trajectories",
            "train_example_count": len(train_examples),
            "strict_seed_values_excluded_from_training": list(strict_seed_values),
        },
        "utility_policy": {
            "target": "observed_public_transition_outcome_utility",
            "component_weights": dict(sorted(UTILITY_COMPONENT_WEIGHTS.items())),
            "estimates_action_utility_not_action_frequency": True,
        },
        "feature_policy": {
            "sequence_policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "rollout_context": rollout_context_feature_contract(config),
            "current_decision_inputs": ["public observation_input", "public action_mask"],
            "prior_history_inputs": [
                "same-agent finalized public trajectory rows before current decision"
            ],
            "excluded_score_features": [
                "seed identity",
                "fixture identity",
                "private world state",
                "future rows at decision time",
                "heuristic recommendations",
                "source path",
                "provenance",
            ],
        },
        "utility_tables": {
            "policy": "mean_observed_transition_utility_by_public_feature_key_and_action_v1",
            "feature_action_utility": table,
        },
        "action_support_counts": _ordered_counts(
            Counter(example.action for example in train_examples)
        ),
        "unsupported_action_rejection": {
            "policy": "runtime_scores_only_currently_valid_actions_v1",
            "invalid_actions_are_never_selected": True,
        },
    }


def _finalize_action_bucket(bucket: Mapping[str, object]) -> dict[str, object]:
    count = int(bucket.get("count", 0))
    utility_sum = float(bucket.get("utility_sum", 0.0))
    components = _mapping(bucket.get("components"))
    return {
        "count": count,
        "utility_mean": _round(utility_sum / float(count) if count else 0.0),
        "component_means": {
            name: _round(float(value) / float(count) if count else 0.0)
            for name, value in sorted(components.items())
        },
    }


def _score_result(
    *,
    predicted_action: str | None,
    scores: Mapping[str, float],
    score_counts: Mapping[str, int],
    score_imputed: Mapping[str, bool],
    source: str,
    source_key: str | None,
    valid_actions: Sequence[str],
    min_observed_support_count: int = 1,
) -> dict[str, object]:
    valid_action_set = set(valid_actions)
    observed_support_floor = max(1, int(min_observed_support_count))
    source_key_category = transition_value_source_key_category(source, source_key)
    valid_scores = {
        action: _round(float(scores[action]))
        for action in ACTION_NAMES
        if action in valid_action_set and action in scores
    }
    valid_support_counts = {
        action: int(score_counts.get(action, 0))
        for action in ACTION_NAMES
        if action in valid_action_set
    }
    valid_imputed_scores = {
        action: bool(score_imputed.get(action, False))
        for action in ACTION_NAMES
        if action in valid_action_set and action in scores
    }
    observed_support_counts = {
        action: (
            0
            if valid_imputed_scores.get(action, False)
            else int(score_counts.get(action, 0))
        )
        for action in ACTION_NAMES
        if action in valid_action_set
    }
    observed_score_actions = {
        action
        for action in valid_action_set
        if action in scores and not valid_imputed_scores.get(action, False)
    }
    low_observed_support_actions = [
        action
        for action in ACTION_NAMES
        if action in observed_score_actions
        and observed_support_counts.get(action, 0) < observed_support_floor
    ]
    observed_scores_for_all_valid_actions = observed_score_actions == valid_action_set
    observed_support_floor_satisfied = (
        observed_scores_for_all_valid_actions and not low_observed_support_actions
    )
    ranked = sorted(
        valid_scores.items(),
        key=lambda item: (-float(item[1]), ACTION_NAMES.index(item[0])),
    )
    best = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    margin = (
        _round(float(best[1]) - float(second[1]))
        if best is not None and second is not None
        else 0.0
    )
    clear_best = best is not None and (second is None or margin > 0.0)
    return {
        "predicted_action": predicted_action if clear_best else None,
        "raw_predicted_action": predicted_action,
        "score_source": source,
        "source_key": source_key,
        "source_key_category": source_key_category,
        "valid_actions": list(valid_actions),
        "valid_action_scores": valid_scores,
        "valid_action_support_counts": valid_support_counts,
        "valid_action_observed_support_counts": observed_support_counts,
        "valid_action_imputed_scores": valid_imputed_scores,
        "imputed_valid_action_score_count": sum(
            1 for imputed in valid_imputed_scores.values() if imputed
        ),
        "observed_valid_action_score_count": len(observed_score_actions),
        "missing_valid_action_score_count": len(valid_action_set - set(valid_scores)),
        "has_imputed_valid_action_score": any(valid_imputed_scores.values()),
        "observed_scores_for_all_valid_actions": observed_scores_for_all_valid_actions,
        "min_observed_support_count": observed_support_floor,
        "valid_actions_below_observed_support_floor": low_observed_support_actions,
        "low_observed_support_valid_action_score_count": len(
            low_observed_support_actions
        ),
        "observed_support_floor_satisfied_for_all_valid_actions": (
            observed_support_floor_satisfied
        ),
        "supported_scores_for_all_valid_actions": (
            set(valid_scores) == valid_action_set
        ),
        "clear_best_valid_action": clear_best,
        "utility_margin": margin,
        "selected_utility": _round(float(best[1])) if best is not None else None,
        "supported_prediction": (
            best is not None
            and clear_best
            and set(valid_scores) == set(valid_actions)
        ),
    }


def _candidate_key_coverage_result(
    *,
    rank: int,
    source_key: str,
    key_present: bool,
    value: object,
    valid_actions: Sequence[str],
    min_observed_support_count: int,
) -> dict[str, object]:
    scored = _partial_scores_from_action_stats(value, valid_actions=valid_actions)
    score = _score_result(
        predicted_action=scored["predicted_action"],
        scores=scored["scores"],
        score_counts=scored["score_counts"],
        score_imputed=scored["score_imputed"],
        source="feature_action_utility",
        source_key=source_key,
        valid_actions=valid_actions,
        min_observed_support_count=min_observed_support_count,
    )
    return {
        "rank": int(rank),
        "source_key": source_key,
        "source_key_category": score["source_key_category"],
        "key_present": bool(key_present),
        "valid_actions": list(valid_actions),
        "complete_for_current_valid_actions": score[
            "supported_scores_for_all_valid_actions"
        ],
        "observed_scores_for_all_valid_actions": score[
            "observed_scores_for_all_valid_actions"
        ],
        "observed_support_floor_satisfied_for_all_valid_actions": score[
            "observed_support_floor_satisfied_for_all_valid_actions"
        ],
        "has_imputed_valid_action_score": score["has_imputed_valid_action_score"],
        "imputed_valid_action_score_count": score[
            "imputed_valid_action_score_count"
        ],
        "observed_valid_action_score_count": score[
            "observed_valid_action_score_count"
        ],
        "missing_valid_action_score_count": score[
            "missing_valid_action_score_count"
        ],
        "valid_action_support_counts": score["valid_action_support_counts"],
        "valid_action_observed_support_counts": score[
            "valid_action_observed_support_counts"
        ],
        "valid_actions_below_observed_support_floor": score[
            "valid_actions_below_observed_support_floor"
        ],
        "low_observed_support_valid_action_score_count": score[
            "low_observed_support_valid_action_score_count"
        ],
        "clear_best_valid_action": score["clear_best_valid_action"],
        "predicted_action": score["predicted_action"],
        "raw_predicted_action": score["raw_predicted_action"],
        "utility_margin": score["utility_margin"],
        "selected_utility": score["selected_utility"],
    }


def _partial_scores_from_action_stats(
    value: object,
    *,
    valid_actions: Sequence[str],
) -> dict[str, object]:
    mapping = _mapping(value)
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    imputed: dict[str, bool] = {}
    for action in valid_actions:
        action_stats = _mapping(mapping.get(action))
        count = _int(action_stats.get("count"))
        counts[action] = max(0, count)
        try:
            utility = float(action_stats.get("utility_mean"))
        except (TypeError, ValueError):
            continue
        if count <= 0 or not math.isfinite(utility):
            continue
        component_means = _mapping(action_stats.get("component_means"))
        scores[action] = utility
        imputed[action] = "imputed_unobserved_action" in component_means
    predicted = (
        max(scores, key=lambda action: (scores[action], -ACTION_NAMES.index(action)))
        if scores
        else None
    )
    return {
        "predicted_action": predicted,
        "scores": scores,
        "score_counts": counts,
        "score_imputed": imputed,
    }


def _scores_from_action_stats(
    value: object,
    *,
    valid_actions: Sequence[str],
) -> dict[str, object] | None:
    mapping = _mapping(value)
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    imputed: dict[str, bool] = {}
    for action in valid_actions:
        action_stats = _mapping(mapping.get(action))
        count = _int(action_stats.get("count"))
        utility = _float(action_stats.get("utility_mean"))
        if count <= 0 or not math.isfinite(utility):
            return None
        component_means = _mapping(action_stats.get("component_means"))
        scores[action] = utility
        counts[action] = count
        imputed[action] = "imputed_unobserved_action" in component_means
    if not scores:
        return None
    predicted = max(
        scores,
        key=lambda action: (scores[action], -ACTION_NAMES.index(action)),
    )
    return {
        "predicted_action": predicted,
        "scores": scores,
        "score_counts": counts,
        "score_imputed": imputed,
    }


def _roundtrip_check(
    *,
    pre_scorer: TransitionValueScorer,
    loaded_scorer: TransitionValueScorer,
    examples: Sequence[_TransitionValueExample],
) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    for example in examples:
        pre = _score_example_from_keys(pre_scorer, example)
        loaded = _score_example_from_keys(loaded_scorer, example)
        if pre != loaded:
            mismatches.append(
                {
                    "row_id": example.row_id,
                    "pre_serialization": pre,
                    "loaded": loaded,
                }
            )
    return {
        "policy": "json_serialization_roundtrip_exact_transition_value_scores_v1",
        "loaded_artifact_scores_match_pre_serialization": not mismatches,
        "checked_example_count": len(examples),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:12],
    }


def _evaluate_scorer(
    *,
    scorer: TransitionValueScorer,
    train_examples: Sequence[_TransitionValueExample],
    strict_examples: Sequence[_TransitionValueExample],
) -> dict[str, object]:
    predicted_counts: Counter[str] = Counter()
    unsupported = 0
    clear_best = 0
    supported = 0
    matched = 0
    utility_sum = 0.0
    for example in strict_examples:
        score = _score_example_from_keys(scorer, example)
        predicted = score.get("predicted_action")
        if isinstance(predicted, str):
            predicted_counts[predicted] += 1
            if predicted == example.action:
                matched += 1
        if score.get("supported_scores_for_all_valid_actions") is True:
            supported += 1
        else:
            unsupported += 1
        if score.get("clear_best_valid_action") is True:
            clear_best += 1
        utility_sum += example.utility
    total = len(strict_examples)
    dominant_action, dominant_count = _dominant_count(predicted_counts)
    return {
        "policy": "v142_transition_value_strict_heldout_diagnostics_v1",
        "train_example_count": len(train_examples),
        "strict_heldout_example_count": total,
        "supported_score_count": supported,
        "supported_score_share": _share(supported, total),
        "missing_supported_score_count": unsupported,
        "clear_best_count": clear_best,
        "clear_best_share": _share(clear_best, total),
        "observed_action_match_count": matched,
        "observed_action_match_share": _share(matched, total),
        "observed_transition_utility_mean": _round(_share_float(utility_sum, total)),
        "predicted_action_counts": _ordered_counts(predicted_counts),
        "dominant_predicted_action": dominant_action,
        "dominant_predicted_action_count": dominant_count,
        "dominant_predicted_action_share": _share(dominant_count, total),
    }


def _score_example_from_keys(
    scorer: TransitionValueScorer,
    example: _TransitionValueExample,
) -> dict[str, object]:
    tables = _mapping(scorer.artifact.get("utility_tables"))
    feature_table = _mapping(tables.get("feature_action_utility"))
    for key in example.feature_keys:
        scored = _scores_from_action_stats(
            feature_table.get(str(key)),
            valid_actions=example.valid_actions,
        )
        if scored is not None:
            return _score_result(
                predicted_action=scored["predicted_action"],
                scores=scored["scores"],
                score_counts=scored["score_counts"],
                score_imputed=scored["score_imputed"],
                source="feature_action_utility",
                source_key=str(key),
                valid_actions=example.valid_actions,
            )
    return _score_result(
        predicted_action=None,
        scores={},
        score_counts={},
        score_imputed={},
        source="missing_supported_scores_for_valid_actions",
        source_key=None,
        valid_actions=example.valid_actions,
    )


def _source_integrity(
    *,
    train_evidence: Mapping[str, object],
    strict_heldout_evidence: Mapping[str, object],
    source_roles: Mapping[str, object],
    schema_scan: Mapping[str, object],
    finite_feature_scan: Mapping[str, object],
    artifact_feature_leakage_scan: Mapping[str, object],
    action_support: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
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
    if schema_scan.get("passed") is not True:
        failures.append("schema_scan_failed")
    if finite_feature_scan.get("passed") is not True:
        failures.append("finite_feature_scan_failed")
    if artifact_feature_leakage_scan.get("passed") is not True:
        failures.append("artifact_feature_leakage_scan_failed")
    if _int(action_support.get("supported_action_count")) <= 0:
        failures.append("no_action_support_counts")
    if _int(action_support.get("heuristic_action_source_count")) > 0:
        failures.append("heuristic_action_source_rows_present")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "train_loaded_path_count": train_evidence.get("loaded_path_count"),
        "strict_heldout_loaded_path_count": strict_heldout_evidence.get(
            "loaded_path_count"
        ),
        "strict_seed_train_source_leakage_count": source_roles.get(
            "strict_seed_train_source_leakage_count"
        ),
        "source_path_overlap_count": source_roles.get("source_path_overlap_count"),
        "strict_seed_presence_only_heldout": source_roles.get(
            "strict_seed_presence_only_heldout"
        ),
        "schema_scan_passed": schema_scan.get("passed"),
        "finite_feature_scan_passed": finite_feature_scan.get("passed"),
        "artifact_feature_leakage_scan_passed": artifact_feature_leakage_scan.get(
            "passed"
        ),
        "heuristic_action_source_count": action_support.get(
            "heuristic_action_source_count"
        ),
    }


def _support_floors(
    *,
    source_integrity: Mapping[str, object],
    finite_feature_scan: Mapping[str, object],
    artifact_feature_leakage_scan: Mapping[str, object],
    action_support: Mapping[str, object],
    roundtrip: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    floors = [
        _floor(
            "source_integrity_passed",
            source_integrity.get("passed") is True,
            observed=source_integrity.get("passed"),
            required=True,
        ),
        _floor(
            "strict_seed_separation_passed",
            source_integrity.get("strict_seed_presence_only_heldout") is True
            and _int(source_integrity.get("strict_seed_train_source_leakage_count")) == 0,
            observed={
                "strict_seed_presence_only_heldout": source_integrity.get(
                    "strict_seed_presence_only_heldout"
                ),
                "strict_seed_train_source_leakage_count": source_integrity.get(
                    "strict_seed_train_source_leakage_count"
                ),
            },
            required="strict seeds only in heldout",
        ),
        _floor(
            "finite_features",
            finite_feature_scan.get("passed") is True,
            observed=finite_feature_scan.get("failure_count"),
            required=0,
        ),
        _floor(
            "no_forbidden_feature_tokens",
            artifact_feature_leakage_scan.get("passed") is True,
            observed=artifact_feature_leakage_scan.get("leakage_count"),
            required=0,
        ),
        _floor(
            "action_support_counts_recorded",
            _int(action_support.get("supported_action_count")) > 0,
            observed=action_support.get("action_support_counts"),
            required="at least one supported train action",
        ),
        _floor(
            "zero_heuristic_action_source_rows",
            _int(action_support.get("heuristic_action_source_count")) == 0,
            observed=action_support.get("heuristic_action_source_count"),
            required=0,
        ),
        _floor(
            "loaded_artifact_scores_match_pre_serialization",
            roundtrip.get("loaded_artifact_scores_match_pre_serialization") is True,
            observed=roundtrip.get("mismatch_count"),
            required=0,
        ),
        _floor(
            "strict_heldout_supported_score_count_nonzero",
            _int(evaluation.get("supported_score_count")) > 0,
            observed=evaluation.get("supported_score_count"),
            required=">0",
        ),
    ]
    first_failed = next((floor for floor in floors if floor["passed"] is not True), None)
    return {
        "policy": "v142_transition_value_scorer_source_floors_v1",
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
        primary = "transition_value_scorer_source_integrity_failed"
    elif support_floors.get("passed") is True:
        primary = "transition_value_scorer_ready_for_opt_in_live_ab"
    else:
        primary = "transition_value_scorer_blocked"
    return {
        "primary": primary,
        "labels": [primary],
        "first_failed_floor": support_floors.get("first_failed_floor"),
        "allowed_classifications": [
            "transition_value_scorer_source_integrity_failed",
            "transition_value_scorer_blocked",
            "transition_value_scorer_ready_for_opt_in_live_ab",
        ],
    }


def _schema_scan(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            failures.append(
                {
                    "record_index": index,
                    "reason": "missing_required_public_transition_fields",
                    "fields": missing,
                }
            )
    return {
        "policy": "v142_public_transition_row_schema_scan_v1",
        "passed": not failures,
        "checked_record_count": len(records),
        "failure_count": len(failures),
        "failures": failures[:24],
        "required_record_fields": list(REQUIRED_RECORD_FIELDS),
    }


def _artifact_feature_leakage_scan(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    leaks: list[dict[str, object]] = []
    feature_table = _mapping(
        _mapping(artifact.get("utility_tables")).get("feature_action_utility")
    )
    for key in sorted(feature_table):
        lowered = str(key).lower()
        for token in FORBIDDEN_SCORE_FEATURE_TOKENS:
            if token in lowered:
                leaks.append(
                    {
                        "feature": "utility_tables.feature_action_utility",
                        "key": str(key),
                        "token": token,
                    }
                )
    return {
        "policy": "v142_serialized_score_feature_leakage_scan_v1",
        "passed": not leaks,
        "forbidden_score_feature_tokens": list(FORBIDDEN_SCORE_FEATURE_TOKENS),
        "leakage_count": len(leaks),
        "leaks": leaks[:24],
    }


def _action_support_summary(
    train_examples: Sequence[_TransitionValueExample],
) -> dict[str, object]:
    counts = Counter(example.action for example in train_examples)
    source_counts = Counter(example.action_source for example in train_examples)
    heuristic_count = sum(
        count for source, count in source_counts.items() if "heuristic" in source
    )
    return {
        "policy": "v142_train_action_support_counts_v1",
        "train_example_count": len(train_examples),
        "action_support_counts": _ordered_counts(counts),
        "supported_action_count": sum(1 for action in ACTION_NAMES if counts[action] > 0),
        "zero_support_actions": [action for action in ACTION_NAMES if counts[action] <= 0],
        "action_source_counts": dict(sorted(source_counts.items())),
        "heuristic_action_source_count": int(heuristic_count),
    }


def _utility_components(record: Mapping[str, object]) -> dict[str, float]:
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    alive_after = after.get("alive") is not False and outcome.get("died") is not True
    reproduced = outcome.get("reproduced") is True
    ready_after = outcome.get("reproduction_ready_after") is True
    action_valid = record.get("action_valid") is True
    resolution_valid = record.get("resolution_action_valid") is True
    return {
        "alive_continuation": 1.0 if alive_after else 0.0,
        "birth_reproduction": 1.0 if reproduced else 0.0,
        "reproduction_readiness": 1.0 if ready_after else 0.0,
        "death_risk": -1.0 if not alive_after else 0.0,
        "energy_delta": _float(after.get("energy_ratio"))
        - _float(before.get("energy_ratio")),
        "hydration_delta": _float(after.get("hydration_ratio"))
        - _float(before.get("hydration_ratio")),
        "health_delta": _float(after.get("health_ratio"))
        - _float(before.get("health_ratio")),
        "unsupported_action_rejection": (
            -1.0 if not action_valid or not resolution_valid else 0.0
        ),
    }


def _utility_from_components(components: Mapping[str, float]) -> float:
    total = 0.0
    for name, weight in UTILITY_COMPONENT_WEIGHTS.items():
        value = float(components.get(name, 0.0))
        if not math.isfinite(value):
            raise TransitionValueScorerError("utility component must be finite")
        total += float(weight) * value
    return _round(total)


def _self_feature_key(values: Sequence[float]) -> str:
    fields = {
        "e": _self_field(values, "energy_ratio"),
        "h": _self_field(values, "hydration_ratio"),
        "hp": _self_field(values, "health_ratio"),
        "inj": _self_field(values, "injury_load"),
        "age": _self_field(values, "age_norm"),
        "repr": _self_field(values, "reproduction_ready"),
        "diet": _self_field(values, "matched_diet_ratio"),
        "role": _self_field(values, "trophic_role_code"),
        "meat": _self_field(values, "meat_mode_code"),
        "water": _self_field(values, "water_access_reason_code"),
        "haz": _self_field(values, "hazard_level"),
        "veg": _self_field(values, "tile_vegetation"),
    }
    return ",".join(f"{name}{_bin(value)}" for name, value in sorted(fields.items()))


def _navigation_feature_key(values: Sequence[float]) -> str:
    base = len(SELF_INPUT_FIELDS) + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    chunks: list[str] = []
    for target_index, target in enumerate(NAVIGATION_TARGETS):
        start = base + target_index * len(NAVIGATION_INPUT_FIELDS)
        dx = _value_at(values, start + NAVIGATION_INPUT_FIELDS.index("dx"))
        dy = _value_at(values, start + NAVIGATION_INPUT_FIELDS.index("dy"))
        distance = _value_at(values, start + NAVIGATION_INPUT_FIELDS.index("distance"))
        strength = _value_at(values, start + NAVIGATION_INPUT_FIELDS.index("strength"))
        chunks.append(
            f"{target}=x{_sign_bin(dx)}y{_sign_bin(dy)}d{_bin(distance)}s{_bin(strength)}"
        )
    return ",".join(chunks)


def _observation_values(payload: Mapping[str, object]) -> tuple[float, ...]:
    values = payload.get("values")
    if isinstance(values, list):
        parsed = tuple(_finite(value) for value in values)
    else:
        parsed = tuple(float(value) for value in decode_observation_input(dict(payload)))
    if len(parsed) != _observation_vector_size():
        raise TransitionValueScorerError(
            f"observation vector has {len(parsed)} values; "
            f"expected {_observation_vector_size()}"
        )
    if any(not math.isfinite(value) for value in parsed):
        raise TransitionValueScorerError("observation features must be finite")
    return parsed


def _observation_vector_size() -> int:
    return len(SELF_INPUT_FIELDS) + (
        PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    ) + (len(NAVIGATION_TARGETS) * len(NAVIGATION_INPUT_FIELDS))


def _self_field(values: Sequence[float], field: str) -> float:
    try:
        return _value_at(values, SELF_INPUT_FIELDS.index(field))
    except ValueError:
        return 0.0


def _value_at(values: Sequence[float], index: int) -> float:
    if index < 0 or index >= len(values):
        return 0.0
    return _finite(values[index])


def _bin(value: float) -> str:
    parsed = _finite(value)
    if parsed < 0.20:
        return "0"
    if parsed < 0.45:
        return "1"
    if parsed < 0.70:
        return "2"
    return "3"


def _sign_bin(value: float) -> str:
    parsed = _finite(value)
    if parsed < -0.05:
        return "n"
    if parsed > 0.05:
        return "p"
    return "z"


def _observation_input(record: Mapping[str, object]) -> Mapping[str, object]:
    payload = record.get("observation_input")
    if not isinstance(payload, Mapping):
        raise TransitionValueScorerError("missing public observation_input")
    return payload


def _valid_actions(record: Mapping[str, object]) -> tuple[str, ...]:
    mask = record.get("action_mask")
    if not isinstance(mask, Mapping):
        raise TransitionValueScorerError("missing public action_mask")
    actions = tuple(action for action in ACTION_NAMES if bool(mask.get(action)))
    if not actions:
        raise TransitionValueScorerError("action_mask has no valid actions")
    return actions


def _valid_actions_from_mask(
    valid_action_mask: Mapping[str, bool] | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(valid_action_mask, Mapping):
        return tuple(action for action in ACTION_NAMES if bool(valid_action_mask.get(action)))
    if isinstance(valid_action_mask, (str, bytes)):
        return ()
    valid = set(valid_action_mask)
    return tuple(action for action in ACTION_NAMES if action in valid)


def _label_action(record: Mapping[str, object]) -> str:
    requested = record.get("requested_action")
    resolved = record.get("resolved_action")
    if record.get("resolution_action_valid", True) is True and requested in ACTION_NAMES:
        return str(requested)
    if resolved in ACTION_NAMES:
        return str(resolved)
    if requested in ACTION_NAMES:
        return str(requested)
    return "stay"


def _action_source(record: Mapping[str, object]) -> str:
    source = record.get("action_source")
    return source if isinstance(source, str) and source else "unknown"


def _row_id(record: Mapping[str, object], *, fallback_index: int) -> str:
    source = str(record.get(TRAJECTORY_SOURCE_PATH_FIELD, "unknown"))
    record_index = record.get(TRAJECTORY_DATASET_RECORD_INDEX_FIELD)
    if isinstance(record_index, int) and not isinstance(record_index, bool):
        return f"{source}:record={record_index}"
    return f"{source}:record={fallback_index}"


def _episode_id(record: Mapping[str, object], *, fallback_index: int) -> str:
    value = record.get(TRAJECTORY_EPISODE_ID_FIELD)
    return value if isinstance(value, str) and value else f"inferred:{fallback_index}"


def _agent_id(record: Mapping[str, object]) -> int:
    value = record.get("agent_id")
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return int(value)


def _seed_values(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(int(item) for item in value if isinstance(item, int) and not isinstance(item, bool))


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


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    action = max(ACTION_NAMES, key=lambda item: (int(counts.get(item, 0)), item))
    return action, int(counts.get(action, 0))


def _ordered_counts(counts: Counter[str]) -> dict[str, int]:
    return {action: int(counts.get(action, 0)) for action in ACTION_NAMES}


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransitionValueScorerError("feature value must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise TransitionValueScorerError("feature value must be finite")
    return parsed


def _share(numerator: int, denominator: int) -> float:
    return _round(float(numerator) / float(denominator)) if denominator else 0.0


def _share_float(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)
