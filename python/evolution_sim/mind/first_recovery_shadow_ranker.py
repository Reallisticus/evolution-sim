from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.dataset import TrajectoryDatasetError, load_trajectory_jsonl
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    FORBIDDEN_TRAINABLE_KEYS as ARCHIVE_FORBIDDEN_TRAINABLE_KEYS,
    trainable_public_input_leakage,
)
from evolution_sim.mind.policy_inputs import (
    PolicyInputError,
    ecological_policy_input_contract,
    ecological_policy_input_values,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION = (
    "mind_v3_first_recovery_shadow_ranker_v1"
)
MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_POLICY = (
    "shadow_only_first_recovery_pairwise_linear_ranker_v1"
)
CURRENT_ROW_FEATURE_POLICY = (
    "current_row_trainable_public_input_linear_pairwise_ranker_v1"
)
OBSERVATION_INPUT_FEATURE_POLICY = (
    "optional_joined_public_ecological_observation_input_v1"
)
PUBLIC_HISTORY_FEATURE_POLICY = "optional_same_agent_public_history_prefix_v1"

DEFAULT_ARCHIVE_REPORT_PATH = Path(
    "output/mind/mind-v3-v109-first-recovery-branch-archive.json"
)
DEFAULT_ARCHIVE_ROWS_PATH = Path(
    "output/mind/mind-v3-v109-first-recovery-branch-archive.jsonl.gz"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v110-first-recovery-shadow-ranker.json"
)
DEFAULT_HISTORY_WINDOW = 8
DOMINANT_SELECTED_ACTION_SHARE_MAX = 0.5
MIN_MEANINGFUL_MATERIAL_GAIN_RECALL = 0.2

ALLOWED_CLASSIFICATION_LABELS: tuple[str, ...] = (
    "shadow_ranker_current_row_learns_signal",
    "shadow_ranker_current_row_no_signal",
    "shadow_ranker_observation_input_improves",
    "shadow_ranker_observation_input_no_improvement",
    "shadow_ranker_history_improves",
    "shadow_ranker_history_no_improvement",
    "shadow_ranker_fixture_open_generalizes",
    "shadow_ranker_fixture_open_fails",
    "shadow_ranker_seed29_passes",
    "shadow_ranker_seed29_fails",
    "shadow_ranker_action_distribution_clean",
    "shadow_ranker_action_distribution_collapsed",
    "shadow_ranker_unsupported_action_free",
    "shadow_ranker_unsupported_action_selected",
    "shadow_ranker_leakage_free",
    "shadow_ranker_leakage_detected",
    "shadow_ranker_ready_for_runtime_planning",
    "shadow_ranker_not_ready_for_runtime_planning",
    "missing_evidence_inconclusive",
)
CLASSIFICATION_LABEL_SET = frozenset(ALLOWED_CLASSIFICATION_LABELS)

FORBIDDEN_FEATURE_KEY_FRAGMENTS = frozenset(
    {
        *ARCHIVE_FORBIDDEN_TRAINABLE_KEYS,
        "provenance",
        "private",
        "logged",
        "source_path",
        "source_kind",
        "fixture",
        "seed",
        "branch_id",
        "record_index",
        "agent_id",
    }
)

CURRENT_NUMERIC_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("target_public_state_before.alive", ("target_public_state_before", "alive")),
    (
        "target_public_state_before.energy_ratio",
        ("target_public_state_before", "energy_ratio"),
    ),
    (
        "target_public_state_before.hydration_ratio",
        ("target_public_state_before", "hydration_ratio"),
    ),
    (
        "target_public_state_before.health_ratio",
        ("target_public_state_before", "health_ratio"),
    ),
    ("target_public_state_before.age", ("target_public_state_before", "age")),
    (
        "public_transition_context.ticks_after_animal_resource_gain",
        ("public_transition_context", "ticks_after_animal_resource_gain"),
    ),
    (
        "public_transition_context.records_after_animal_resource_gain",
        ("public_transition_context", "records_after_animal_resource_gain"),
    ),
)


class FirstRecoveryShadowRankerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BranchTargetGroup:
    group_key: str
    rows: tuple[dict[str, object], ...]
    seed: int | None
    source_kind: str | None


@dataclass(frozen=True, slots=True)
class LinearRankerModel:
    feature_policy: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    pairwise_example_count: int
    train_group_count: int
    train_row_count: int


@dataclass(frozen=True, slots=True)
class FirstRecoveryShadowRankerBuild:
    report: dict[str, object]


FeatureFunction = Callable[[Mapping[str, object]], tuple[float, ...]]


def build_first_recovery_shadow_ranker(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    trajectory_paths: Sequence[str | Path] = (),
    trajectory_glob_patterns: Sequence[str] = (),
    enable_observation_input: bool = False,
    enable_public_history: bool = False,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> FirstRecoveryShadowRankerBuild:
    history_limit = _positive_int(history_window, field="history_window")
    contract = _contract(
        enable_observation_input=enable_observation_input,
        enable_public_history=enable_public_history,
        history_window=history_limit,
    )
    report_payload, report_evidence = _resolve_archive_report(
        archive_report,
        archive_report_path,
    )
    rows_payload, rows_evidence = _resolve_archive_rows(
        archive_rows,
        archive_rows_path,
    )
    rows = [dict(row) for row in rows_payload]
    groups = group_archive_rows_by_branch_target(rows)
    archive_summary = _archive_summary(
        archive_report=report_payload,
        rows=rows,
        groups=groups,
        rows_evidence=rows_evidence,
    )
    split_metadata = _split_metadata(groups)
    leakage_audit = _leakage_audit(rows)
    unsupported_action_audit = _unsupported_action_audit(rows)
    feature_sets = _feature_sets(
        rows=rows,
        leakage_audit=leakage_audit,
        enable_observation_input=enable_observation_input,
        enable_public_history=enable_public_history,
        history_window=history_limit,
    )
    evidence = {
        "archive_report": report_evidence,
        "archive_rows": rows_evidence,
    }
    missing_evidence = _missing_evidence(
        evidence=evidence,
        archive_summary=archive_summary,
        groups=groups,
        leakage_audit=leakage_audit,
        unsupported_action_audit=unsupported_action_audit,
    )
    if missing_evidence:
        empty_eval = _empty_evaluation("missing_evidence_inconclusive")
        baselines = _baseline_report(groups)
        current_row_ranker = _missing_current_ranker(missing_evidence)
        leave_one_seed = _empty_split_evaluation(missing_evidence)
        leave_one_source = _empty_split_evaluation(missing_evidence)
        fixture_open = _empty_fixture_open_evaluation(missing_evidence)
        seed29_evaluation = _empty_seed29_evaluation(missing_evidence)
        action_distribution = _selected_action_distribution(empty_eval)
        material_gain = _material_gain_recall_section(empty_eval)
        observation_input_ranker = _optional_ranker_not_run(
            enabled=enable_observation_input,
            answer=(
                "missing_evidence_inconclusive"
                if enable_observation_input
                else "shadow_ranker_observation_input_no_improvement"
            ),
            reason="missing_required_archive_evidence",
        )
        public_history_ranker = _optional_ranker_not_run(
            enabled=enable_public_history,
            answer=(
                "missing_evidence_inconclusive"
                if enable_public_history
                else "shadow_ranker_history_no_improvement"
            ),
            reason="missing_required_archive_evidence",
        )
    else:
        current_feature_names = current_row_feature_names()
        current_feature = current_row_feature_vector
        current_all = _train_and_evaluate(
            groups,
            groups,
            feature_names=current_feature_names,
            feature_func=current_feature,
            feature_policy=CURRENT_ROW_FEATURE_POLICY,
        )
        baselines = _baseline_report(groups)
        leave_one_seed = _leave_one_seed_evaluation(
            groups,
            feature_names=current_feature_names,
            feature_func=current_feature,
            feature_policy=CURRENT_ROW_FEATURE_POLICY,
        )
        leave_one_source = _leave_one_source_evaluation(
            groups,
            feature_names=current_feature_names,
            feature_func=current_feature,
            feature_policy=CURRENT_ROW_FEATURE_POLICY,
        )
        fixture_open = _fixture_open_evaluation(
            groups,
            feature_names=current_feature_names,
            feature_func=current_feature,
            feature_policy=CURRENT_ROW_FEATURE_POLICY,
        )
        seed29_evaluation = _seed29_evaluation(
            groups,
            feature_names=current_feature_names,
            feature_func=current_feature,
            feature_policy=CURRENT_ROW_FEATURE_POLICY,
        )
        current_row_ranker = _current_row_ranker_section(
            all_data=current_all,
            leave_one_seed=leave_one_seed,
            leave_one_source=leave_one_source,
            fixture_open=fixture_open,
            seed29_evaluation=seed29_evaluation,
        )
        action_distribution = _selected_action_distribution(
            _mapping(leave_one_seed.get("aggregate")).get("current_row_linear_ranker")
            or current_all["metrics"]
        )
        material_gain = _material_gain_recall_section(
            _mapping(leave_one_seed.get("aggregate")).get("current_row_linear_ranker")
            or current_all["metrics"]
        )
        observation_input_ranker = _optional_observation_input_ranker(
            groups=groups,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=trajectory_glob_patterns,
            enabled=enable_observation_input,
            current_reference=leave_one_seed,
        )
        public_history_ranker = _optional_public_history_ranker(
            groups=groups,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=trajectory_glob_patterns,
            enabled=enable_public_history,
            history_window=history_limit,
            current_reference=leave_one_seed,
        )
    sections = {
        "current_row_linear_ranker": current_row_ranker,
        "observation_input_ranker": observation_input_ranker,
        "public_history_ranker": public_history_ranker,
        "seed29_evaluation": seed29_evaluation,
        "fixture_open_evaluation": fixture_open,
        "leave_one_seed_evaluation": leave_one_seed,
        "leave_one_source_evaluation": leave_one_source,
        "action_distribution": action_distribution,
        "leakage_audit": leakage_audit,
        "unsupported_action_audit": unsupported_action_audit,
    }
    classification = _classification(
        missing_evidence=missing_evidence,
        sections=sections,
    )
    recommendation = _research_recommendation(
        classification=classification,
        current_row_ranker=current_row_ranker,
        seed29_evaluation=seed29_evaluation,
        fixture_open=fixture_open,
        action_distribution=action_distribution,
        unsupported_action_audit=unsupported_action_audit,
        leakage_audit=leakage_audit,
        material_gain=material_gain,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
        "scorer_policy": MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_POLICY,
        "contract": contract,
        "provenance": _provenance(
            contract=contract,
            archive_report_path=archive_report_path,
            archive_rows_path=archive_rows_path,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=trajectory_glob_patterns,
            report_evidence=report_evidence,
            rows_evidence=rows_evidence,
        ),
        "archive_summary": archive_summary,
        "feature_sets": feature_sets,
        "split_metadata": split_metadata,
        "baselines": baselines,
        "current_row_linear_ranker": current_row_ranker,
        "observation_input_ranker": observation_input_ranker,
        "public_history_ranker": public_history_ranker,
        "seed29_evaluation": seed29_evaluation,
        "fixture_open_evaluation": fixture_open,
        "leave_one_seed_evaluation": leave_one_seed,
        "leave_one_source_evaluation": leave_one_source,
        "action_distribution": action_distribution,
        "leakage_audit": leakage_audit,
        "unsupported_action_audit": unsupported_action_audit,
        "material_gain_recall": material_gain,
        "classification": classification,
        "research_recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryShadowRankerBuild(report=report)


def write_first_recovery_shadow_ranker_report(
    build: FirstRecoveryShadowRankerBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def load_first_recovery_archive_rows(
    path: str | Path,
) -> tuple[dict[str, object], ...]:
    rows, evidence = _load_archive_rows_path(Path(path))
    if evidence.get("loaded") is not True:
        raise FirstRecoveryShadowRankerError(
            f"archive rows could not be loaded: {evidence.get('error')}"
        )
    return tuple(rows)


def group_archive_rows_by_branch_target(
    archive_rows: Sequence[Mapping[str, object]],
) -> tuple[BranchTargetGroup, ...]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in archive_rows:
        provenance = _mapping(row.get("provenance"))
        branch_id = provenance.get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            continue
        grouped[branch_id].append(dict(row))
    groups: list[BranchTargetGroup] = []
    for branch_id, rows in grouped.items():
        sorted_rows = tuple(sorted(rows, key=_row_sort_key))
        provenance = _mapping(sorted_rows[0].get("provenance")) if sorted_rows else {}
        groups.append(
            BranchTargetGroup(
                group_key=branch_id,
                rows=sorted_rows,
                seed=_int_or_none(provenance.get("seed")),
                source_kind=_optional_string(provenance.get("source_kind")),
            )
        )
    return tuple(sorted(groups, key=_group_sort_key))


def build_pairwise_examples(
    groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str] | None = None,
    feature_func: FeatureFunction | None = None,
) -> tuple[dict[str, object], ...]:
    names = tuple(feature_names) if feature_names is not None else current_row_feature_names()
    func = feature_func or current_row_feature_vector
    examples: list[dict[str, object]] = []
    for group in groups:
        rows = tuple(_legal_candidate_rows(group.rows))
        for better in rows:
            better_rank = _oracle_rank(better)
            for worse in rows:
                worse_rank = _oracle_rank(worse)
                if better_rank >= worse_rank:
                    continue
                examples.append(
                    {
                        "target_key": group.group_key,
                        "better_archive_row_id": better.get("archive_row_id"),
                        "worse_archive_row_id": worse.get("archive_row_id"),
                        "better_oracle_rank": better_rank,
                        "worse_oracle_rank": worse_rank,
                        "feature_delta": [
                            _round(left - right)
                            for left, right in zip(func(better), func(worse), strict=True)
                        ],
                        "feature_count": len(names),
                    }
                )
    return tuple(examples)


def current_row_feature_names() -> tuple[str, ...]:
    names = ["post_carrion_first_recovery"]
    names.extend(f"candidate_action.{action}" for action in ACTION_NAMES)
    names.extend(f"action_mask.{action}" for action in ACTION_NAMES)
    names.extend(name for name, _path in CURRENT_NUMERIC_PATHS)
    return tuple(names)


def current_row_feature_vector(row: Mapping[str, object]) -> tuple[float, ...]:
    trainable = _mapping(row.get("trainable_public_input"))
    action = _optional_string(trainable.get("candidate_action")) or ""
    mask = _mapping(trainable.get("action_mask"))
    values: list[float] = [
        1.0 if trainable.get("post_carrion_first_recovery") is True else 0.0
    ]
    values.extend(1.0 if action == candidate else 0.0 for candidate in ACTION_NAMES)
    values.extend(1.0 if mask.get(candidate) is True else 0.0 for candidate in ACTION_NAMES)
    for _name, path in CURRENT_NUMERIC_PATHS:
        values.append(_nested_number(trainable, path))
    return tuple(_round(value) for value in values)


def train_pairwise_linear_ranker(
    groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str] | None = None,
    feature_func: FeatureFunction | None = None,
    feature_policy: str = CURRENT_ROW_FEATURE_POLICY,
    epochs: int = 400,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> LinearRankerModel:
    names = tuple(feature_names) if feature_names is not None else current_row_feature_names()
    func = feature_func or current_row_feature_vector
    train_rows = [row for group in groups for row in _legal_candidate_rows(group.rows)]
    means, scales = _fit_scaler(train_rows, func, len(names))
    features_by_id = {
        str(row.get("archive_row_id")): _scale_vector(func(row), means, scales)
        for row in train_rows
    }
    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for group in groups:
        rows = tuple(_legal_candidate_rows(group.rows))
        for better in rows:
            better_rank = _oracle_rank(better)
            better_features = features_by_id.get(str(better.get("archive_row_id")))
            if better_features is None:
                continue
            for worse in rows:
                if better_rank >= _oracle_rank(worse):
                    continue
                worse_features = features_by_id.get(str(worse.get("archive_row_id")))
                if worse_features is None:
                    continue
                pairs.append((better_features, worse_features))
    weights = [0.0 for _ in names]
    if pairs:
        for _epoch in range(int(epochs)):
            gradient = [l2 * weight for weight in weights]
            for better, worse in pairs:
                delta = [left - right for left, right in zip(better, worse, strict=True)]
                score = _dot(weights, delta)
                coeff = -1.0 / (1.0 + math.exp(_clamp(score, -50.0, 50.0)))
                for index, value in enumerate(delta):
                    gradient[index] += coeff * value
            inv_count = 1.0 / float(len(pairs))
            for index in range(len(weights)):
                weights[index] -= learning_rate * gradient[index] * inv_count
    return LinearRankerModel(
        feature_policy=feature_policy,
        feature_names=names,
        means=tuple(_round(value) for value in means),
        scales=tuple(_round(value) for value in scales),
        weights=tuple(_round(value) for value in weights),
        pairwise_example_count=len(pairs),
        train_group_count=len(groups),
        train_row_count=len(train_rows),
    )


def model_scores_for_group(
    model: LinearRankerModel,
    group: BranchTargetGroup,
    *,
    feature_func: FeatureFunction | None = None,
) -> list[dict[str, object]]:
    func = feature_func or current_row_feature_vector
    scored: list[dict[str, object]] = []
    for row in _legal_candidate_rows(group.rows):
        vector = _scale_vector(func(row), model.means, model.scales)
        scored.append(
            {
                "row": row,
                "score": _round(_dot(model.weights, vector)),
                "candidate_action": row.get("candidate_action"),
                "oracle_rank": _oracle_rank(row),
            }
        )
    scored.sort(
        key=lambda item: (
            -_number(item.get("score")),
            _action_index(str(item.get("candidate_action"))),
            str(_mapping(item.get("row")).get("archive_row_id")),
        )
    )
    return scored


def selected_action_distribution_answer(
    selected_action_counts: Mapping[str, object],
) -> dict[str, object]:
    counts = Counter({str(key): _int(value) for key, value in selected_action_counts.items()})
    dominant = _dominant_count_share(counts)
    answer = (
        "shadow_ranker_action_distribution_collapsed"
        if _number(dominant.get("share")) > DOMINANT_SELECTED_ACTION_SHARE_MAX
        else "shadow_ranker_action_distribution_clean"
    )
    return {
        "answer": answer,
        "selected_action_counts": _counter_to_ordered_dict(counts),
        "dominant_selected_action": dominant["key"],
        "dominant_selected_action_count": dominant["count"],
        "dominant_selected_action_share": dominant["share"],
        "dominant_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
    }


def history_usefulness_answer(
    *,
    current_metrics: Mapping[str, object],
    history_metrics: Mapping[str, object],
) -> str:
    current_mrr = _number(current_metrics.get("mrr"))
    history_mrr = _number(history_metrics.get("mrr"))
    current_top1 = _number(current_metrics.get("top1_oracle_match_rate"))
    history_top1 = _number(history_metrics.get("top1_oracle_match_rate"))
    if history_mrr > current_mrr and history_top1 >= current_top1:
        return "shadow_ranker_history_improves"
    return "shadow_ranker_history_no_improvement"


def _train_and_evaluate(
    train_groups: Sequence[BranchTargetGroup],
    test_groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str],
    feature_func: FeatureFunction,
    feature_policy: str,
) -> dict[str, object]:
    model = train_pairwise_linear_ranker(
        train_groups,
        feature_names=feature_names,
        feature_func=feature_func,
        feature_policy=feature_policy,
    )
    metrics = _evaluate_model(
        model,
        test_groups,
        feature_func=feature_func,
    )
    return {
        "feature_policy": feature_policy,
        "model": _model_summary(model),
        "metrics": metrics,
    }


def _evaluate_model(
    model: LinearRankerModel,
    groups: Sequence[BranchTargetGroup],
    *,
    feature_func: FeatureFunction,
) -> dict[str, object]:
    rankings = {
        group.group_key: model_scores_for_group(
            model,
            group,
            feature_func=feature_func,
        )
        for group in groups
    }
    return _metrics_from_rankings(groups, rankings)


def _metrics_from_rankings(
    groups: Sequence[BranchTargetGroup],
    rankings: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    selected_rows: list[Mapping[str, object]] = []
    pairwise_correct = 0.0
    pairwise_total = 0
    top1 = 0
    top2 = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    regrets: list[float] = []
    selected_oracle_ranks: list[float] = []
    logged_gains: list[float] = []
    stay_gains: list[float] = []
    material_groups = 0
    selected_material_groups = 0
    unsupported = 0
    evaluable = 0
    missing_selection = 0
    for group in groups:
        ranking = list(rankings.get(group.group_key, ()))
        rows = [item["row"] for item in ranking if isinstance(item.get("row"), Mapping)]
        if not rows:
            missing_selection += 1
            continue
        evaluable += 1
        selected = rows[0]
        selected_rows.append(selected)
        best_rows = _oracle_best_rows(rows)
        best_rank = min(_oracle_rank(row) for row in rows)
        selected_rank = _oracle_rank(selected)
        selected_oracle_ranks.append(float(selected_rank))
        regrets.append(float(selected_rank - best_rank))
        if any(_same_archive_row(selected, best) for best in best_rows):
            top1 += 1
        top_two_rows = rows[:2]
        if any(
            any(_same_archive_row(candidate, best) for candidate in top_two_rows)
            for best in best_rows
        ):
            top2 += 1
        oracle_position = next(
            (
                index + 1
                for index, row in enumerate(rows)
                if any(_same_archive_row(row, best) for best in best_rows)
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if oracle_position is None else 1.0 / oracle_position)
        ndcgs.append(_ndcg(rows))
        selected_score_by_id = {
            str(_mapping(item.get("row")).get("archive_row_id")): _number(item.get("score"))
            for item in ranking
        }
        for better in rows:
            for worse in rows:
                if _oracle_rank(better) >= _oracle_rank(worse):
                    continue
                pairwise_total += 1
                better_score = selected_score_by_id.get(str(better.get("archive_row_id")), 0.0)
                worse_score = selected_score_by_id.get(str(worse.get("archive_row_id")), 0.0)
                if better_score > worse_score:
                    pairwise_correct += 1.0
                elif better_score == worse_score:
                    pairwise_correct += 0.5
        logged = _logged_action_row(group)
        if logged is not None:
            logged_gains.append(float(_oracle_rank(logged) - selected_rank))
        stay = _row_for_action(group.rows, "stay")
        if stay is not None:
            stay_gains.append(float(_oracle_rank(stay) - selected_rank))
        if any(row.get("material_gain_label") is True for row in rows):
            material_groups += 1
            if selected.get("material_gain_label") is True:
                selected_material_groups += 1
        if not _row_observation_supported(selected):
            unsupported += 1
    counts = Counter(str(row.get("candidate_action")) for row in selected_rows)
    dominant = _dominant_count_share(counts)
    return {
        "group_count": len(groups),
        "evaluable_group_count": evaluable,
        "missing_selection_count": missing_selection,
        "pairwise_accuracy": _share_float(pairwise_correct, pairwise_total),
        "pairwise_comparison_count": pairwise_total,
        "top1_oracle_match_rate": _share(top1, evaluable),
        "top2_oracle_inclusion_rate": _share(top2, evaluable),
        "mrr": _round(_mean(reciprocal_ranks)),
        "ndcg": _round(_mean(ndcgs)),
        "mean_selected_oracle_rank": _round(_mean(selected_oracle_ranks)),
        "mean_rank_regret_vs_oracle": _round(_mean(regrets)),
        "mean_rank_gain_vs_logged_action": _round(_mean(logged_gains)),
        "mean_rank_gain_vs_stay_baseline": _round(_mean(stay_gains)),
        "material_gain_group_count": material_groups,
        "selected_material_gain_group_count": selected_material_groups,
        "material_gain_recall": _share(selected_material_groups, material_groups),
        "selected_action_counts": _counter_to_ordered_dict(counts),
        "dominant_selected_action": dominant["key"],
        "dominant_selected_action_count": dominant["count"],
        "dominant_selected_action_share": dominant["share"],
        "unsupported_action_selection_count": unsupported,
        "unsupported_action_rate": _share(unsupported, evaluable),
    }


def _baseline_report(groups: Sequence[BranchTargetGroup]) -> dict[str, object]:
    if not groups:
        return {
            "baseline_policy": "evaluation_only_no_runtime_policy_effect_v1",
            "all_data": {},
        }
    train_counts = _oracle_best_action_counts(groups)
    return {
        "baseline_policy": "evaluation_only_no_runtime_policy_effect_v1",
        "policies": {
            "logged_action": "candidate matching provenance.logged_action; evaluation only",
            "stay": "candidate stay action when present; evaluation only",
            "immediate_homeostatic_delta": (
                "best public first-action recovery vitals delta; evaluation only"
            ),
            "weak_best_action_frequency": (
                "most frequent training oracle action; weak descriptive baseline"
            ),
        },
        "all_data": {
            "logged_action": _evaluate_baseline(groups, _select_logged_baseline),
            "stay": _evaluate_baseline(groups, _select_stay_baseline),
            "immediate_homeostatic_delta": _evaluate_baseline(
                groups,
                _select_homeostatic_baseline,
            ),
            "weak_best_action_frequency": _evaluate_baseline(
                groups,
                lambda group: _select_frequency_baseline(group, train_counts),
            ),
        },
    }


def _leave_one_seed_evaluation(
    groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str],
    feature_func: FeatureFunction,
    feature_policy: str,
) -> dict[str, object]:
    seeds = sorted({group.seed for group in groups if group.seed is not None})
    splits: list[dict[str, object]] = []
    for seed in seeds:
        train = [group for group in groups if group.seed != seed]
        test = [group for group in groups if group.seed == seed]
        splits.append(
            _split_evaluation(
                holdout_field="holdout_seed",
                holdout_value=seed,
                train_groups=train,
                test_groups=test,
                feature_names=feature_names,
                feature_func=feature_func,
                feature_policy=feature_policy,
            )
        )
    aggregate = _aggregate_split_evaluations(splits)
    return {
        "answer": _signal_answer(aggregate),
        "split_policy": "leave_one_seed_grouped_branch_target_v1",
        "split_count": len(splits),
        "splits": splits,
        "aggregate": aggregate,
    }


def _leave_one_source_evaluation(
    groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str],
    feature_func: FeatureFunction,
    feature_policy: str,
) -> dict[str, object]:
    sources = sorted({group.source_kind for group in groups if group.source_kind})
    splits: list[dict[str, object]] = []
    for source in sources:
        train = [group for group in groups if group.source_kind != source]
        test = [group for group in groups if group.source_kind == source]
        splits.append(
            _split_evaluation(
                holdout_field="holdout_source_kind",
                holdout_value=source,
                train_groups=train,
                test_groups=test,
                feature_names=feature_names,
                feature_func=feature_func,
                feature_policy=feature_policy,
            )
        )
    aggregate = _aggregate_split_evaluations(splits)
    return {
        "answer": _source_generalization_answer(aggregate),
        "split_policy": "leave_one_source_grouped_branch_target_v1",
        "split_count": len(splits),
        "splits": splits,
        "aggregate": aggregate,
    }


def _fixture_open_evaluation(
    groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str],
    feature_func: FeatureFunction,
    feature_policy: str,
) -> dict[str, object]:
    sources = sorted({group.source_kind for group in groups if group.source_kind})
    splits: list[dict[str, object]] = []
    for source in sources:
        train = [group for group in groups if group.source_kind != source]
        test = [group for group in groups if group.source_kind == source]
        splits.append(
            _split_evaluation(
                holdout_field="holdout_source_kind",
                holdout_value=source,
                train_groups=train,
                test_groups=test,
                feature_names=feature_names,
                feature_func=feature_func,
                feature_policy=feature_policy,
            )
        )
    aggregate = _aggregate_split_evaluations(splits)
    return {
        "answer": _source_generalization_answer(aggregate),
        "split_policy": "fixture_open_bidirectional_group_holdout_v1",
        "split_count": len(splits),
        "splits": splits,
        "aggregate": aggregate,
    }


def _seed29_evaluation(
    groups: Sequence[BranchTargetGroup],
    *,
    feature_names: Sequence[str],
    feature_func: FeatureFunction,
    feature_policy: str,
) -> dict[str, object]:
    train = [group for group in groups if group.seed != 29]
    test = [group for group in groups if group.seed == 29]
    split = _split_evaluation(
        holdout_field="holdout_seed",
        holdout_value=29,
        train_groups=train,
        test_groups=test,
        feature_names=feature_names,
        feature_func=feature_func,
        feature_policy=feature_policy,
    )
    metrics = _mapping(split.get("current_row_linear_ranker"))
    answer = (
        "shadow_ranker_seed29_passes"
        if _int(metrics.get("evaluable_group_count")) > 0
        and _number(metrics.get("unsupported_action_rate")) == 0.0
        and (
            _number(metrics.get("top1_oracle_match_rate")) > 0.0
            or _number(metrics.get("material_gain_recall")) >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
        )
        else "shadow_ranker_seed29_fails"
    )
    return {
        "answer": answer,
        "split_policy": "seed29_holdout_grouped_branch_target_v1",
        "holdout_seed": 29,
        "evaluation": split,
    }


def _split_evaluation(
    *,
    holdout_field: str,
    holdout_value: object,
    train_groups: Sequence[BranchTargetGroup],
    test_groups: Sequence[BranchTargetGroup],
    feature_names: Sequence[str],
    feature_func: FeatureFunction,
    feature_policy: str,
) -> dict[str, object]:
    current = _train_and_evaluate(
        train_groups,
        test_groups,
        feature_names=feature_names,
        feature_func=feature_func,
        feature_policy=feature_policy,
    )
    train_counts = _oracle_best_action_counts(train_groups)
    return {
        holdout_field: holdout_value,
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "current_row_linear_ranker": current["metrics"],
        "baselines": {
            "logged_action": _evaluate_baseline(test_groups, _select_logged_baseline),
            "stay": _evaluate_baseline(test_groups, _select_stay_baseline),
            "immediate_homeostatic_delta": _evaluate_baseline(
                test_groups,
                _select_homeostatic_baseline,
            ),
            "weak_best_action_frequency": _evaluate_baseline(
                test_groups,
                lambda group: _select_frequency_baseline(group, train_counts),
            ),
        },
        "model": current["model"],
    }


def _aggregate_split_evaluations(
    splits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    current_metrics = [_mapping(split.get("current_row_linear_ranker")) for split in splits]
    baseline_names = ("logged_action", "stay", "immediate_homeostatic_delta", "weak_best_action_frequency")
    baseline_aggregates = {
        name: _aggregate_metrics(
            [_mapping(_mapping(split.get("baselines")).get(name)) for split in splits]
        )
        for name in baseline_names
    }
    current = _aggregate_metrics(current_metrics)
    logged = baseline_aggregates["logged_action"]
    stay = baseline_aggregates["stay"]
    return {
        "current_row_linear_ranker": current,
        "baselines": baseline_aggregates,
        "beats_logged_baseline": _beats_baseline(current, logged),
        "beats_stay_baseline": _beats_baseline(current, stay),
        "split_count": len(splits),
    }


def _aggregate_metrics(metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total_groups = sum(_int(metric.get("group_count")) for metric in metrics)
    evaluable = sum(_int(metric.get("evaluable_group_count")) for metric in metrics)
    pairwise_total = sum(_int(metric.get("pairwise_comparison_count")) for metric in metrics)
    pairwise_correct = sum(
        _number(metric.get("pairwise_accuracy")) * _int(metric.get("pairwise_comparison_count"))
        for metric in metrics
    )
    material_total = sum(_int(metric.get("material_gain_group_count")) for metric in metrics)
    material_selected = sum(_int(metric.get("selected_material_gain_group_count")) for metric in metrics)
    unsupported = sum(_int(metric.get("unsupported_action_selection_count")) for metric in metrics)
    counts = Counter()
    for metric in metrics:
        counts.update(
            {str(key): _int(value) for key, value in _mapping(metric.get("selected_action_counts")).items()}
        )
    dominant = _dominant_count_share(counts)

    def weighted_mean(field: str) -> float:
        return _round(
            _weighted_mean(
                (
                    (
                        _number(metric.get(field)),
                        _int(metric.get("evaluable_group_count")),
                    )
                    for metric in metrics
                )
            )
        )

    return {
        "group_count": total_groups,
        "evaluable_group_count": evaluable,
        "pairwise_accuracy": _share_float(pairwise_correct, pairwise_total),
        "pairwise_comparison_count": pairwise_total,
        "top1_oracle_match_rate": weighted_mean("top1_oracle_match_rate"),
        "top2_oracle_inclusion_rate": weighted_mean("top2_oracle_inclusion_rate"),
        "mrr": weighted_mean("mrr"),
        "ndcg": weighted_mean("ndcg"),
        "mean_selected_oracle_rank": weighted_mean("mean_selected_oracle_rank"),
        "mean_rank_regret_vs_oracle": weighted_mean("mean_rank_regret_vs_oracle"),
        "mean_rank_gain_vs_logged_action": weighted_mean("mean_rank_gain_vs_logged_action"),
        "mean_rank_gain_vs_stay_baseline": weighted_mean("mean_rank_gain_vs_stay_baseline"),
        "material_gain_group_count": material_total,
        "selected_material_gain_group_count": material_selected,
        "material_gain_recall": _share(material_selected, material_total),
        "selected_action_counts": _counter_to_ordered_dict(counts),
        "dominant_selected_action": dominant["key"],
        "dominant_selected_action_count": dominant["count"],
        "dominant_selected_action_share": dominant["share"],
        "unsupported_action_selection_count": unsupported,
        "unsupported_action_rate": _share(unsupported, evaluable),
    }


def _current_row_ranker_section(
    *,
    all_data: Mapping[str, object],
    leave_one_seed: Mapping[str, object],
    leave_one_source: Mapping[str, object],
    fixture_open: Mapping[str, object],
    seed29_evaluation: Mapping[str, object],
) -> dict[str, object]:
    aggregate = _mapping(_mapping(leave_one_seed.get("aggregate")).get("current_row_linear_ranker"))
    answer = _signal_answer(_mapping(leave_one_seed.get("aggregate")))
    model = _mapping(all_data.get("model"))
    return {
        "answer": answer,
        "feature_policy": CURRENT_ROW_FEATURE_POLICY,
        "training_policy": "deterministic_full_batch_ranknet_style_linear_logistic_v1",
        "hyperparameters": {
            "epochs": 400,
            "learning_rate": 0.08,
            "l2": 0.001,
            "randomness": "none",
            "feature_scaling": "fit_on_training_rows_only_per_split",
        },
        "all_data_descriptive": all_data,
        "heldout_summary": {
            "leave_one_seed_answer": leave_one_seed.get("answer"),
            "leave_one_source_answer": leave_one_source.get("answer"),
            "fixture_open_answer": fixture_open.get("answer"),
            "seed29_answer": seed29_evaluation.get("answer"),
            "leave_one_seed_metrics": aggregate,
        },
        "coefficients_report_only_non_promoted": model.get("coefficients", []),
        "runtime_loadable_artifact_emitted": False,
    }


def _optional_observation_input_ranker(
    *,
    groups: Sequence[BranchTargetGroup],
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
    enabled: bool,
    current_reference: Mapping[str, object],
) -> dict[str, object]:
    if not enabled:
        return _optional_ranker_not_run(
            enabled=False,
            answer="shadow_ranker_observation_input_no_improvement",
            reason="disabled",
        )
    join = _joined_observation_features(groups, trajectory_paths, trajectory_glob_patterns)
    if join["missing_evidence"]:
        return {
            "enabled": True,
            "answer": "missing_evidence_inconclusive",
            "feature_policy": OBSERVATION_INPUT_FEATURE_POLICY,
            "join_evidence": join["evidence"],
            "missing_evidence": join["missing_evidence"],
        }
    feature_names = current_row_feature_names() + tuple(join["feature_names"])
    feature_map = join["feature_map"]

    def feature(row: Mapping[str, object]) -> tuple[float, ...]:
        extra = tuple(feature_map.get(str(row.get("archive_row_id")), ()))
        return current_row_feature_vector(row) + extra

    evaluation = _leave_one_seed_evaluation(
        groups,
        feature_names=feature_names,
        feature_func=feature,
        feature_policy=OBSERVATION_INPUT_FEATURE_POLICY,
    )
    answer = _optional_improvement_answer(
        current_reference=current_reference,
        candidate=evaluation,
        improves_label="shadow_ranker_observation_input_improves",
        no_improvement_label="shadow_ranker_observation_input_no_improvement",
    )
    return {
        "enabled": True,
        "answer": answer,
        "feature_policy": OBSERVATION_INPUT_FEATURE_POLICY,
        "join_evidence": join["evidence"],
        "evaluation": evaluation,
    }


def _optional_public_history_ranker(
    *,
    groups: Sequence[BranchTargetGroup],
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
    enabled: bool,
    history_window: int,
    current_reference: Mapping[str, object],
) -> dict[str, object]:
    if not enabled:
        return _optional_ranker_not_run(
            enabled=False,
            answer="shadow_ranker_history_no_improvement",
            reason="disabled",
        )
    history = _joined_history_features(
        groups,
        trajectory_paths,
        trajectory_glob_patterns,
        history_window=history_window,
    )
    if history["missing_evidence"]:
        return {
            "enabled": True,
            "answer": "missing_evidence_inconclusive",
            "feature_policy": PUBLIC_HISTORY_FEATURE_POLICY,
            "history_window": history_window,
            "join_evidence": history["evidence"],
            "missing_evidence": history["missing_evidence"],
        }
    feature_names = current_row_feature_names() + tuple(history["feature_names"])
    feature_map = history["feature_map"]

    def feature(row: Mapping[str, object]) -> tuple[float, ...]:
        extra = tuple(feature_map.get(str(row.get("archive_row_id")), ()))
        return current_row_feature_vector(row) + extra

    evaluation = _leave_one_seed_evaluation(
        groups,
        feature_names=feature_names,
        feature_func=feature,
        feature_policy=PUBLIC_HISTORY_FEATURE_POLICY,
    )
    current_metrics = _mapping(_mapping(current_reference.get("aggregate")).get("current_row_linear_ranker"))
    history_metrics = _mapping(_mapping(evaluation.get("aggregate")).get("current_row_linear_ranker"))
    answer = history_usefulness_answer(
        current_metrics=current_metrics,
        history_metrics=history_metrics,
    )
    return {
        "enabled": True,
        "answer": answer,
        "feature_policy": PUBLIC_HISTORY_FEATURE_POLICY,
        "history_window": history_window,
        "join_evidence": history["evidence"],
        "evaluation": evaluation,
    }


def _joined_observation_features(
    groups: Sequence[BranchTargetGroup],
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> dict[str, object]:
    records = _load_join_records(trajectory_paths, trajectory_glob_patterns)
    if records["missing_evidence"]:
        return {**records, "feature_names": (), "feature_map": {}}
    record_by_key = records["record_by_key"]
    feature_map: dict[str, tuple[float, ...]] = {}
    missing = 0
    malformed = 0
    feature_count: int | None = None
    for row in _rows_from_groups(groups):
        record = _matching_record(row, record_by_key)
        if record is None:
            missing += 1
            continue
        try:
            values = ecological_policy_input_values(dict(_mapping(record.get("observation_input"))))
        except (PolicyInputError, ValueError, TypeError):
            malformed += 1
            continue
        feature_count = len(values)
        feature_map[str(row.get("archive_row_id"))] = tuple(values)
    evidence = {
        **_mapping(records["evidence"]),
        "matched_archive_row_count": len(feature_map),
        "missing_match_count": missing,
        "malformed_observation_input_count": malformed,
        "ecological_policy_input_contract": ecological_policy_input_contract(),
    }
    missing_evidence: list[str] = []
    if missing or malformed or not feature_map:
        missing_evidence.append("observation_input_join")
    names = tuple(
        f"observation_input.ecological_value_{index}"
        for index in range(int(feature_count or 0))
    )
    return {
        "evidence": evidence,
        "missing_evidence": missing_evidence,
        "feature_names": names,
        "feature_map": feature_map,
    }


def _joined_history_features(
    groups: Sequence[BranchTargetGroup],
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
    *,
    history_window: int,
) -> dict[str, object]:
    records = _load_join_records(trajectory_paths, trajectory_glob_patterns)
    if records["missing_evidence"]:
        return {**records, "feature_names": (), "feature_map": {}}
    by_path = records["records_by_path"]
    feature_names = (
        "public_history.record_count",
        "public_history.mean_energy_ratio_before",
        "public_history.mean_hydration_ratio_before",
        "public_history.mean_health_ratio_before",
        "public_history.mean_resource_gain",
        "public_history.failed_movement_rate",
    )
    feature_map: dict[str, tuple[float, ...]] = {}
    missing = 0
    for row in _rows_from_groups(groups):
        provenance = _mapping(row.get("provenance"))
        source_path = _optional_string(provenance.get("source_path"))
        record_index = _int_or_none(provenance.get("record_index"))
        agent_id = _int_or_none(provenance.get("agent_id"))
        if source_path is None or record_index is None or agent_id is None:
            missing += 1
            continue
        candidates = [
            record
            for index, record in by_path.get(source_path, ())
            if index < record_index and _int_or_none(record.get("agent_id")) == agent_id
        ][-history_window:]
        feature_map[str(row.get("archive_row_id"))] = _history_values(candidates)
    evidence = {
        **_mapping(records["evidence"]),
        "matched_archive_row_count": len(feature_map),
        "missing_match_count": missing,
        "history_window": int(history_window),
    }
    missing_evidence: list[str] = []
    if missing or not feature_map:
        missing_evidence.append("public_history_join")
    return {
        "evidence": evidence,
        "missing_evidence": missing_evidence,
        "feature_names": feature_names,
        "feature_map": feature_map,
    }


def _load_join_records(
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
) -> dict[str, object]:
    paths = tuple(sorted({Path(path) for path in trajectory_paths}, key=lambda item: str(item)))
    if not paths and trajectory_glob_patterns:
        import glob

        paths = tuple(
            sorted(
                {Path(path) for pattern in trajectory_glob_patterns for path in glob.glob(pattern)},
                key=lambda item: str(item),
            )
        )
    evidence = {
        "trajectory_paths": [str(path) for path in paths],
        "trajectory_globs": list(trajectory_glob_patterns),
        "loaded_path_count": 0,
        "load_failure_count": 0,
        "record_count": 0,
        "load_failures": [],
    }
    record_by_key: dict[tuple[str, int], dict[str, object]] = {}
    records_by_path: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for path in paths:
        try:
            dataset = load_trajectory_jsonl(path)
        except (OSError, TrajectoryDatasetError, ValueError) as exc:
            evidence["load_failure_count"] = _int(evidence.get("load_failure_count")) + 1
            _list_mut(evidence, "load_failures").append(
                {"path": str(path), "reason": type(exc).__name__, "message": str(exc)}
            )
            continue
        evidence["loaded_path_count"] = _int(evidence.get("loaded_path_count")) + 1
        evidence["record_count"] = _int(evidence.get("record_count")) + len(dataset.records)
        path_keys = {str(path), str(dataset.path)}
        try:
            path_keys.add(str(Path(path).resolve()))
            path_keys.add(str(Path(dataset.path).resolve()))
        except OSError:
            pass
        for index, record in enumerate(dataset.records):
            row = dict(record)
            for key in path_keys:
                record_by_key[(key, index)] = row
                records_by_path[key].append((index, row))
    missing = []
    if not paths:
        missing.append("trajectory_paths")
    if _int(evidence.get("loaded_path_count")) == 0:
        missing.append("trajectory_loads")
    if _int(evidence.get("load_failure_count")) > 0:
        missing.append("trajectory_load_failures")
    return {
        "evidence": evidence,
        "missing_evidence": missing,
        "record_by_key": record_by_key,
        "records_by_path": records_by_path,
    }


def _matching_record(
    row: Mapping[str, object],
    record_by_key: Mapping[tuple[str, int], Mapping[str, object]],
) -> Mapping[str, object] | None:
    provenance = _mapping(row.get("provenance"))
    source_path = _optional_string(provenance.get("source_path"))
    record_index = _int_or_none(provenance.get("record_index"))
    if source_path is None or record_index is None:
        return None
    candidate = record_by_key.get((source_path, record_index))
    if candidate is None:
        try:
            candidate = record_by_key.get((str(Path(source_path).resolve()), record_index))
        except OSError:
            candidate = None
    if candidate is None:
        return None
    if _int_or_none(candidate.get("agent_id")) != _int_or_none(provenance.get("agent_id")):
        return None
    if _int_or_none(candidate.get("tick")) != _int_or_none(row.get("tick")):
        return None
    expected_digest = _optional_string(row.get("observation_digest"))
    if expected_digest and candidate.get("observation_digest") != expected_digest:
        return None
    return candidate


def _history_values(records: Sequence[Mapping[str, object]]) -> tuple[float, ...]:
    count = len(records)
    before_values = [_mapping(record.get("before")) for record in records]
    outcomes = [_mapping(record.get("outcome")) for record in records]
    failed_moves = sum(
        1
        for record in records
        if str(record.get("requested_action", "")).startswith("move_")
        and record.get("moved") is not True
    )
    return (
        _round(float(count)),
        _round(_mean(_number_or_none(before.get("energy_ratio")) for before in before_values)),
        _round(_mean(_number_or_none(before.get("hydration_ratio")) for before in before_values)),
        _round(_mean(_number_or_none(before.get("health_ratio")) for before in before_values)),
        _round(_mean(_number_or_none(outcome.get("resource_gain")) for outcome in outcomes)),
        _share(failed_moves, count),
    )


def _optional_improvement_answer(
    *,
    current_reference: Mapping[str, object],
    candidate: Mapping[str, object],
    improves_label: str,
    no_improvement_label: str,
) -> str:
    current = _mapping(_mapping(current_reference.get("aggregate")).get("current_row_linear_ranker"))
    candidate_metrics = _mapping(_mapping(candidate.get("aggregate")).get("current_row_linear_ranker"))
    if (
        _number(candidate_metrics.get("mrr")) > _number(current.get("mrr"))
        and _number(candidate_metrics.get("top1_oracle_match_rate"))
        >= _number(current.get("top1_oracle_match_rate"))
    ):
        return improves_label
    return no_improvement_label


def _evaluate_baseline(
    groups: Sequence[BranchTargetGroup],
    selector: Callable[[BranchTargetGroup], Mapping[str, object] | None],
) -> dict[str, object]:
    rankings: dict[str, list[dict[str, object]]] = {}
    for group in groups:
        selected = selector(group)
        legal_rows = list(_legal_candidate_rows(group.rows))
        if selected is None:
            rankings[group.group_key] = []
            continue
        selected_id = str(selected.get("archive_row_id"))
        ordered = [selected] + [
            row for row in sorted(legal_rows, key=_row_sort_key)
            if str(row.get("archive_row_id")) != selected_id
        ]
        rankings[group.group_key] = [
            {
                "row": row,
                "score": 1.0 if index == 0 else 0.0,
                "candidate_action": row.get("candidate_action"),
                "oracle_rank": _oracle_rank(row),
            }
            for index, row in enumerate(ordered)
        ]
    return _metrics_from_rankings(groups, rankings)


def _select_logged_baseline(group: BranchTargetGroup) -> Mapping[str, object] | None:
    return _logged_action_row(group)


def _select_stay_baseline(group: BranchTargetGroup) -> Mapping[str, object] | None:
    return _row_for_action(group.rows, "stay")


def _select_homeostatic_baseline(group: BranchTargetGroup) -> Mapping[str, object] | None:
    rows = tuple(_legal_candidate_rows(group.rows))
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            -_homeostatic_delta(row),
            _action_index(str(row.get("candidate_action"))),
            str(row.get("archive_row_id")),
        ),
    )[0]


def _select_frequency_baseline(
    group: BranchTargetGroup,
    counts: Mapping[str, int],
) -> Mapping[str, object] | None:
    rows = tuple(_legal_candidate_rows(group.rows))
    if not rows:
        return None
    actions = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), _action_index(item[0]), item[0]),
    )
    for action, _count in actions:
        row = _row_for_action(rows, action)
        if row is not None:
            return row
    return sorted(rows, key=_row_sort_key)[0]


def _oracle_best_action_counts(groups: Sequence[BranchTargetGroup]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for group in groups:
        best = _oracle_best_rows(group.rows)
        if best:
            counts[str(best[0].get("candidate_action"))] += 1
    return _counter_to_ordered_dict(counts)


def _classification(
    *,
    missing_evidence: Sequence[str],
    sections: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    scores = {label: 0 for label in ALLOWED_CLASSIFICATION_LABELS}
    labels: list[str] = []
    all_missing = sorted(
        {
            str(item)
            for item in (
                list(missing_evidence)
                + [
                    item
                    for section in sections.values()
                    for item in _section_missing_evidence(section)
                ]
            )
        }
    )
    if all_missing:
        labels.append("missing_evidence_inconclusive")
    for section in sections.values():
        for label in _section_labels(section):
            if label in CLASSIFICATION_LABEL_SET and label not in labels:
                labels.append(label)
    if not all_missing:
        ready_label = (
            "shadow_ranker_ready_for_runtime_planning"
            if _runtime_planning_ready(labels)
            else "shadow_ranker_not_ready_for_runtime_planning"
        )
        if ready_label not in labels:
            labels.append(ready_label)
    if not labels:
        labels.append("missing_evidence_inconclusive")
    for label in labels:
        scores[label] += 1
    return {
        "primary": _primary_label(labels),
        "labels": labels,
        "category_scores": scores,
        "diagnostic_answers": _diagnostic_answers(sections),
        "missing_evidence": all_missing,
    }


def _runtime_planning_ready(labels: Sequence[str]) -> bool:
    required = {
        "shadow_ranker_current_row_learns_signal",
        "shadow_ranker_fixture_open_generalizes",
        "shadow_ranker_seed29_passes",
        "shadow_ranker_action_distribution_clean",
        "shadow_ranker_unsupported_action_free",
        "shadow_ranker_leakage_free",
    }
    return required.issubset(set(labels))


def _primary_label(labels: Sequence[str]) -> str:
    priority = (
        "shadow_ranker_leakage_detected",
        "shadow_ranker_unsupported_action_selected",
        "missing_evidence_inconclusive",
        "shadow_ranker_action_distribution_collapsed",
        "shadow_ranker_seed29_fails",
        "shadow_ranker_fixture_open_fails",
        "shadow_ranker_current_row_no_signal",
        "shadow_ranker_ready_for_runtime_planning",
        "shadow_ranker_not_ready_for_runtime_planning",
    )
    for label in priority:
        if label in labels:
            return label
    return sorted(labels)[0]


def _research_recommendation(
    *,
    classification: Mapping[str, object],
    current_row_ranker: Mapping[str, object],
    seed29_evaluation: Mapping[str, object],
    fixture_open: Mapping[str, object],
    action_distribution: Mapping[str, object],
    unsupported_action_audit: Mapping[str, object],
    leakage_audit: Mapping[str, object],
    material_gain: Mapping[str, object],
) -> dict[str, object]:
    ready = (
        classification.get("primary") != "missing_evidence_inconclusive"
        and current_row_ranker.get("answer") == "shadow_ranker_current_row_learns_signal"
        and seed29_evaluation.get("answer") == "shadow_ranker_seed29_passes"
        and fixture_open.get("answer") == "shadow_ranker_fixture_open_generalizes"
        and action_distribution.get("answer") == "shadow_ranker_action_distribution_clean"
        and unsupported_action_audit.get("answer") == "shadow_ranker_unsupported_action_free"
        and leakage_audit.get("answer") == "shadow_ranker_leakage_free"
        and _number(material_gain.get("heldout_material_gain_recall"))
        >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
    )
    return {
        "classification_primary": classification.get("primary"),
        "answer": (
            "shadow_ranker_ready_for_runtime_planning"
            if ready
            else "shadow_ranker_not_ready_for_runtime_planning"
        ),
        "recommendation": (
            "planner_may_start_runtime_planning_from_v110_shadow_ranker"
            if ready
            else "do_not_start_runtime_policy_planning_from_v110"
        ),
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_relaxation_recommended": False,
        "next_research_direction": (
            "If held-out signal is weak, audit missing public recovery signals before adding fields."
        ),
        "go_criteria": {
            "heldout_signal": current_row_ranker.get("answer")
            == "shadow_ranker_current_row_learns_signal",
            "seed29_passes": seed29_evaluation.get("answer")
            == "shadow_ranker_seed29_passes",
            "fixture_open_generalizes": fixture_open.get("answer")
            == "shadow_ranker_fixture_open_generalizes",
            "dominant_selected_action_share_ok": action_distribution.get("answer")
            == "shadow_ranker_action_distribution_clean",
            "unsupported_rate_zero": unsupported_action_audit.get("answer")
            == "shadow_ranker_unsupported_action_free",
            "leakage_count_zero": leakage_audit.get("answer")
            == "shadow_ranker_leakage_free",
            "material_gain_recall_meaningful": _number(
                material_gain.get("heldout_material_gain_recall")
            )
            >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
        },
    }


def _contract(
    *,
    enable_observation_input: bool,
    enable_public_history: bool,
    history_window: int,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
        "scorer_policy": MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_POLICY,
        "shadow_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "private_world_state_input": False,
        "fixture_identity_input": False,
        "seed_identity_input": False,
        "source_identity_input": False,
        "branch_identity_input": False,
        "logged_action_fallback_input": False,
        "unsupported_action_selection_allowed": False,
        "runtime_loadable_artifact_emitted": False,
        "coefficients_policy": "report_only_non_promoted",
        "feature_sets": {
            "current_row": CURRENT_ROW_FEATURE_POLICY,
            "observation_input_enabled": bool(enable_observation_input),
            "public_history_enabled": bool(enable_public_history),
            "history_window": int(history_window),
        },
    }


def _provenance(
    *,
    contract: Mapping[str, object],
    archive_report_path: str | Path | None,
    archive_rows_path: str | Path | None,
    trajectory_paths: Sequence[str | Path],
    trajectory_glob_patterns: Sequence[str],
    report_evidence: Mapping[str, object],
    rows_evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "contract_digest": stable_payload_digest(contract),
        "archive_report_path": str(archive_report_path) if archive_report_path else None,
        "archive_rows_path": str(archive_rows_path) if archive_rows_path else None,
        "archive_report_sha256": report_evidence.get("file_sha256"),
        "archive_rows_sha256": rows_evidence.get("file_sha256"),
        "trajectory_paths": [str(path) for path in trajectory_paths],
        "trajectory_globs": list(trajectory_glob_patterns),
    }


def _resolve_archive_report(
    payload: Mapping[str, object] | None,
    path: str | Path | None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    if payload is not None:
        report = dict(payload)
        return report, {
            "path": str(path) if path is not None else None,
            "loaded": True,
            "schema_version": report.get("schema_version"),
            "schema_matches": report.get("schema_version")
            == MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
            "in_memory": True,
        }
    if path is None:
        return None, {
            "path": None,
            "loaded": False,
            "schema_matches": False,
            "error": "missing_archive_report_path",
        }
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, {
            "path": str(path),
            "loaded": False,
            "schema_matches": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    if not isinstance(report, dict):
        return None, {
            "path": str(path),
            "loaded": False,
            "schema_matches": False,
            "error": "archive_report_not_object",
        }
    return report, {
        "path": str(path),
        "loaded": True,
        "schema_version": report.get("schema_version"),
        "schema_matches": report.get("schema_version")
        == MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
        "file_sha256": _file_sha256(resolved),
    }


def _resolve_archive_rows(
    rows: Sequence[Mapping[str, object]] | None,
    path: str | Path | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if rows is not None:
        parsed = tuple(dict(row) for row in rows)
        return parsed, {
            "path": str(path) if path is not None else None,
            "loaded": True,
            "row_count": len(parsed),
            "malformed_row_count": 0,
            "in_memory": True,
            "rows_sha256": _rows_sha256(parsed),
        }
    if path is None:
        return (), {
            "path": None,
            "loaded": False,
            "row_count": 0,
            "malformed_row_count": 0,
            "load_failure_count": 1,
            "error": "missing_archive_rows_path",
        }
    return _load_archive_rows_path(Path(path))


def _load_archive_rows_path(path: Path) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    rows: list[dict[str, object]] = []
    malformed = 0
    examples: list[dict[str, object]] = []
    try:
        with _open_input(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "line_number": line_number,
                                "reason": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                    continue
                if not isinstance(payload, dict):
                    malformed += 1
                    if len(examples) < 8:
                        examples.append(
                            {
                                "line_number": line_number,
                                "reason": "row_not_object",
                            }
                        )
                    continue
                rows.append(payload)
    except OSError as exc:
        return (), {
            "path": str(path),
            "loaded": False,
            "row_count": 0,
            "malformed_row_count": 0,
            "load_failure_count": 1,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return tuple(rows), {
        "path": str(path),
        "loaded": True,
        "row_count": len(rows),
        "malformed_row_count": malformed,
        "malformed_examples": examples,
        "load_failure_count": 0,
        "file_sha256": _file_sha256(path),
        "rows_sha256": _rows_sha256(rows),
    }


def _archive_summary(
    *,
    archive_report: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
    groups: Sequence[BranchTargetGroup],
    rows_evidence: Mapping[str, object],
) -> dict[str, object]:
    summary = _mapping(_mapping(archive_report or {}).get("branch_archive_summary"))
    legality = _mapping(_mapping(archive_report or {}).get("legality_summary"))
    readiness = _mapping(_mapping(archive_report or {}).get("learnability_readiness"))
    expected = _int_or_none(summary.get("archive_row_count"))
    return {
        "source_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "source_schema_matches": _mapping(archive_report or {}).get("schema_version")
        == MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
        "expected_archive_row_count": expected,
        "loaded_archive_row_count": len(rows),
        "row_count_matches_archive_report": expected == len(rows)
        if expected is not None
        else False,
        "branch_target_group_count": len(groups),
        "source_reconstructed_first_recovery_row_count": _mapping(
            _mapping(archive_report or {}).get("row_reconstruction")
        ).get("reconstructed_first_recovery_row_count"),
        "source_selected_target_count": summary.get("selected_row_count"),
        "source_replay_verified": summary.get("replay_verified"),
        "source_heuristic_action_source_count": summary.get("heuristic_action_source_count"),
        "source_zero_heuristic_runtime_actions_except_diagnostic_force": summary.get(
            "zero_heuristic_runtime_actions_except_diagnostic_force"
        ),
        "source_legality_answer": legality.get("answer"),
        "source_resolution_answer": legality.get("resolution_answer"),
        "source_trainable_leakage_count": _mapping(
            readiness.get("trainable_public_input_leakage")
        ).get("leak_count"),
        "archive_rows_loaded": rows_evidence.get("loaded") is True,
        "archive_rows_malformed_count": rows_evidence.get("malformed_row_count", 0),
    }


def _feature_sets(
    *,
    rows: Sequence[Mapping[str, object]],
    leakage_audit: Mapping[str, object],
    enable_observation_input: bool,
    enable_public_history: bool,
    history_window: int,
) -> dict[str, object]:
    return {
        "current_row": {
            "enabled": True,
            "feature_policy": CURRENT_ROW_FEATURE_POLICY,
            "feature_count": len(current_row_feature_names()),
            "feature_names": list(current_row_feature_names()),
            "source": "trainable_public_input_only",
            "row_count": len(rows),
        },
        "observation_input": {
            "enabled": bool(enable_observation_input),
            "feature_policy": OBSERVATION_INPUT_FEATURE_POLICY,
            "uses_provenance_for_join_only": True,
        },
        "public_history": {
            "enabled": bool(enable_public_history),
            "feature_policy": PUBLIC_HISTORY_FEATURE_POLICY,
            "history_window": int(history_window),
            "uses_only_records_strictly_before_branch_target": True,
        },
        "leakage_audit": dict(leakage_audit),
    }


def _split_metadata(groups: Sequence[BranchTargetGroup]) -> dict[str, object]:
    seeds = sorted(seed for seed in {group.seed for group in groups} if seed is not None)
    sources = sorted(
        source for source in {group.source_kind for group in groups} if source is not None
    )
    by_seed = Counter(str(group.seed) for group in groups)
    by_source = Counter(str(group.source_kind) for group in groups)
    return {
        "split_policy": "grouped_branch_target_leave_one_seed_and_source_v1",
        "group_count": len(groups),
        "seed_count": len(seeds),
        "source_count": len(sources),
        "seeds": seeds,
        "sources": sources,
        "groups_by_seed": _counter_to_ordered_dict(by_seed),
        "groups_by_source": _counter_to_ordered_dict(by_source),
        "leave_one_seed_viable": len(seeds) >= 2,
        "leave_one_source_viable": len(sources) >= 2,
    }


def _leakage_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    trainable = trainable_public_input_leakage(rows)
    feature_name_leaks = [
        name
        for name in current_row_feature_names()
        if any(
            token in FORBIDDEN_FEATURE_KEY_FRAGMENTS
            for token in name.replace("=", ".").split(".")
        )
    ]
    leak_count = _int(trainable.get("leak_count")) + len(feature_name_leaks)
    answer = (
        "shadow_ranker_leakage_detected"
        if leak_count > 0
        else "shadow_ranker_leakage_free"
    )
    return {
        "answer": answer,
        "leakage_detected": leak_count > 0,
        "leak_count": leak_count,
        "trainable_public_input_leakage": trainable,
        "feature_name_leaks": feature_name_leaks,
        "forbidden_feature_key_fragments": sorted(FORBIDDEN_FEATURE_KEY_FRAGMENTS),
        "branch_id_used_for_grouping_only": True,
        "provenance_used_for_training_features": False,
    }


def _unsupported_action_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    unsupported_rows = [
        {
            "archive_row_id": row.get("archive_row_id"),
            "candidate_action": row.get("candidate_action"),
            "observation_legal": row.get("observation_legal"),
        }
        for row in rows
        if not _row_observation_supported(row)
    ]
    return {
        "answer": "shadow_ranker_unsupported_action_free",
        "archive_row_count": len(rows),
        "unsupported_archive_candidate_count": len(unsupported_rows),
        "selected_unsupported_action_count": 0,
        "unsupported_action_rate": 0.0,
        "examples": unsupported_rows[:16],
        "prediction_policy": "select_only_observation_legal_archive_candidates",
    }


def _missing_evidence(
    *,
    evidence: Mapping[str, object],
    archive_summary: Mapping[str, object],
    groups: Sequence[BranchTargetGroup],
    leakage_audit: Mapping[str, object],
    unsupported_action_audit: Mapping[str, object],
) -> list[str]:
    missing: list[str] = []
    report = _mapping(evidence.get("archive_report"))
    rows = _mapping(evidence.get("archive_rows"))
    if report.get("loaded") is not True:
        missing.append("archive_report")
    elif report.get("schema_matches") is not True:
        missing.append("archive_report_schema_mismatch")
    if rows.get("loaded") is not True:
        missing.append("archive_rows")
    if _int(rows.get("malformed_row_count")) > 0:
        missing.append("archive_rows_malformed")
    if archive_summary.get("row_count_matches_archive_report") is not True:
        missing.append("archive_row_count_mismatch")
    if archive_summary.get("source_replay_verified") is not True:
        missing.append("archive_replay_not_verified")
    if _int(archive_summary.get("source_heuristic_action_source_count")) != 0:
        missing.append("archive_heuristic_action_source_count_nonzero")
    if leakage_audit.get("answer") == "shadow_ranker_leakage_detected":
        missing.append("trainable_feature_leakage")
    if not groups:
        missing.append("branch_target_groups")
    elif not build_pairwise_examples(groups):
        missing.append("pairwise_examples")
    if unsupported_action_audit.get("unsupported_archive_candidate_count") is None:
        missing.append("unsupported_action_audit")
    return sorted(set(missing))


def _selected_action_distribution(metrics: Mapping[str, object]) -> dict[str, object]:
    payload = selected_action_distribution_answer(
        _mapping(metrics.get("selected_action_counts"))
    )
    payload.update(
        {
            "source_metric": "leave_one_seed_current_row_linear_ranker",
            "evaluable_group_count": _int(metrics.get("evaluable_group_count")),
        }
    )
    return payload


def _material_gain_recall_section(metrics: Mapping[str, object]) -> dict[str, object]:
    recall = _number(metrics.get("material_gain_recall"))
    return {
        "heldout_material_gain_recall": recall,
        "material_gain_group_count": _int(metrics.get("material_gain_group_count")),
        "selected_material_gain_group_count": _int(
            metrics.get("selected_material_gain_group_count")
        ),
        "meaningful_recall_threshold": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
        "meaningful_material_gain_recall": recall >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
    }


def _signal_answer(aggregate: Mapping[str, object]) -> str:
    current = _mapping(aggregate.get("current_row_linear_ranker"))
    baselines = _mapping(aggregate.get("baselines"))
    logged = _mapping(baselines.get("logged_action"))
    stay = _mapping(baselines.get("stay"))
    learns = (
        _int(current.get("evaluable_group_count")) > 0
        and _number(current.get("unsupported_action_rate")) == 0.0
        and _beats_baseline(current, logged)
        and _beats_baseline(current, stay)
    )
    return (
        "shadow_ranker_current_row_learns_signal"
        if learns
        else "shadow_ranker_current_row_no_signal"
    )


def _source_generalization_answer(aggregate: Mapping[str, object]) -> str:
    current = _mapping(aggregate.get("current_row_linear_ranker"))
    generalizes = (
        _int(current.get("evaluable_group_count")) > 0
        and aggregate.get("beats_logged_baseline") is True
        and aggregate.get("beats_stay_baseline") is True
        and _number(current.get("unsupported_action_rate")) == 0.0
    )
    return (
        "shadow_ranker_fixture_open_generalizes"
        if generalizes
        else "shadow_ranker_fixture_open_fails"
    )


def _beats_baseline(
    current: Mapping[str, object],
    baseline: Mapping[str, object],
) -> bool:
    if _int(current.get("evaluable_group_count")) <= 0:
        return False
    return (
        _number(current.get("mrr")) > _number(baseline.get("mrr"))
        or _number(current.get("top1_oracle_match_rate"))
        > _number(baseline.get("top1_oracle_match_rate"))
        or _number(current.get("mean_selected_oracle_rank"))
        < _number(baseline.get("mean_selected_oracle_rank"), default=999.0)
    )


def _model_summary(model: LinearRankerModel) -> dict[str, object]:
    coefficients = [
        {
            "feature": name,
            "weight": weight,
            "report_only_non_promoted": True,
        }
        for name, weight in sorted(
            zip(model.feature_names, model.weights, strict=True),
            key=lambda item: (-abs(item[1]), item[0]),
        )
    ]
    return {
        "feature_policy": model.feature_policy,
        "feature_count": len(model.feature_names),
        "pairwise_example_count": model.pairwise_example_count,
        "train_group_count": model.train_group_count,
        "train_row_count": model.train_row_count,
        "coefficients": coefficients,
        "runtime_loadable_artifact": False,
    }


def _empty_evaluation(answer: str) -> dict[str, object]:
    return {
        "answer": answer,
        "group_count": 0,
        "evaluable_group_count": 0,
        "selected_action_counts": {},
        "unsupported_action_rate": 0.0,
    }


def _missing_current_ranker(missing_evidence: Sequence[str]) -> dict[str, object]:
    return {
        "answer": "missing_evidence_inconclusive",
        "feature_policy": CURRENT_ROW_FEATURE_POLICY,
        "missing_evidence": list(missing_evidence),
        "runtime_loadable_artifact_emitted": False,
    }


def _empty_split_evaluation(missing_evidence: Sequence[str]) -> dict[str, object]:
    return {
        "answer": "missing_evidence_inconclusive",
        "split_count": 0,
        "splits": [],
        "aggregate": {"current_row_linear_ranker": _empty_evaluation("missing_evidence_inconclusive")},
        "missing_evidence": list(missing_evidence),
    }


def _empty_fixture_open_evaluation(missing_evidence: Sequence[str]) -> dict[str, object]:
    payload = _empty_split_evaluation(missing_evidence)
    payload["answer"] = "missing_evidence_inconclusive"
    payload["split_policy"] = "fixture_open_bidirectional_group_holdout_v1"
    return payload


def _empty_seed29_evaluation(missing_evidence: Sequence[str]) -> dict[str, object]:
    return {
        "answer": "missing_evidence_inconclusive",
        "holdout_seed": 29,
        "evaluation": {},
        "missing_evidence": list(missing_evidence),
    }


def _optional_ranker_not_run(*, enabled: bool, answer: str, reason: str) -> dict[str, object]:
    return {
        "enabled": bool(enabled),
        "answer": answer,
        "reason": reason,
        "runtime_loadable_artifact_emitted": False,
    }


def _section_labels(section: Mapping[str, object]) -> list[str]:
    labels: list[str] = []
    answer = section.get("answer")
    if isinstance(answer, str):
        labels.append(answer)
    for value in _list(section.get("labels")):
        if isinstance(value, str):
            labels.append(value)
    nested_keys = (
        "classification",
        "aggregate",
        "current_row_linear_ranker",
        "observation_input_ranker",
        "public_history_ranker",
    )
    for key in nested_keys:
        nested = section.get(key)
        if isinstance(nested, Mapping):
            labels.extend(_section_labels(nested))
    return labels


def _section_missing_evidence(section: Mapping[str, object]) -> list[str]:
    missing = [
        str(value)
        for value in _list(section.get("missing_evidence"))
        if isinstance(value, str)
    ]
    for key in (
        "aggregate",
        "current_row_linear_ranker",
        "observation_input_ranker",
        "public_history_ranker",
    ):
        nested = section.get(key)
        if isinstance(nested, Mapping):
            missing.extend(_section_missing_evidence(nested))
    return missing


def _diagnostic_answers(sections: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    answers: dict[str, object] = {}
    for name, section in sorted(sections.items()):
        answer = section.get("answer")
        if isinstance(answer, str):
            answers[f"{name}.answer"] = answer
    return answers


def _legal_candidate_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(row)
        for row in sorted(rows, key=_row_sort_key)
        if _row_observation_supported(row)
    )


def _row_observation_supported(row: Mapping[str, object]) -> bool:
    action = _optional_string(row.get("candidate_action"))
    if action is None:
        return False
    mask = _mapping(row.get("action_mask"))
    trainable_mask = _mapping(_mapping(row.get("trainable_public_input")).get("action_mask"))
    return (
        row.get("observation_legal") is True
        and (mask.get(action) is True or trainable_mask.get(action) is True)
    )


def _logged_action_row(group: BranchTargetGroup) -> Mapping[str, object] | None:
    provenance = _mapping(group.rows[0].get("provenance")) if group.rows else {}
    logged_action = _optional_string(provenance.get("logged_action"))
    if logged_action is None:
        return None
    return _row_for_action(group.rows, logged_action)


def _row_for_action(
    rows: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    for row in sorted(rows, key=_row_sort_key):
        if row.get("candidate_action") == action and _row_observation_supported(row):
            return row
    return None


def _oracle_best_rows(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    legal = tuple(_legal_candidate_rows(rows))
    if not legal:
        return ()
    best_rank = min(_oracle_rank(row) for row in legal)
    return tuple(row for row in legal if _oracle_rank(row) == best_rank)


def _oracle_rank(row: Mapping[str, object]) -> int:
    rank = _int(row.get("oracle_rank"), default=999)
    return rank if rank > 0 else 999


def _homeostatic_delta(row: Mapping[str, object]) -> float:
    vitals = _mapping(row.get("recovery_vitals_deltas"))
    return _round(
        sum(
            _number(vitals.get(field))
            for field in (
                "energy_ratio_delta",
                "hydration_ratio_delta",
                "health_ratio_delta",
                "target_recovery_score_delta",
            )
        )
    )


def _ndcg(rows: Sequence[Mapping[str, object]]) -> float:
    if not rows:
        return 0.0
    relevances = [_rank_relevance(row) for row in rows]
    ideal = sorted(relevances, reverse=True)
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevances))
    idcg = sum(value / math.log2(index + 2) for index, value in enumerate(ideal))
    return _round(dcg / idcg if idcg > 0 else 0.0)


def _rank_relevance(row: Mapping[str, object]) -> float:
    rank = _oracle_rank(row)
    return 0.0 if rank <= 0 or rank >= 999 else 1.0 / float(rank)


def _fit_scaler(
    rows: Sequence[Mapping[str, object]],
    feature_func: FeatureFunction,
    feature_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not rows:
        return tuple(0.0 for _ in range(feature_count)), tuple(1.0 for _ in range(feature_count))
    vectors = [feature_func(row) for row in rows]
    means = [
        _mean(vector[index] for vector in vectors)
        for index in range(feature_count)
    ]
    scales: list[float] = []
    for index, mean in enumerate(means):
        variance = _mean((vector[index] - mean) ** 2 for vector in vectors)
        scale = math.sqrt(max(variance, 0.0))
        scales.append(1.0 if scale < 1e-9 else scale)
    return tuple(means), tuple(scales)


def _scale_vector(
    values: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        _round((float(value) - float(mean)) / float(scale or 1.0))
        for value, mean, scale in zip(values, means, scales, strict=True)
    )


def _rows_from_groups(groups: Sequence[BranchTargetGroup]) -> tuple[dict[str, object], ...]:
    return tuple(row for group in groups for row in group.rows)


def _group_sort_key(group: BranchTargetGroup) -> tuple[str, int, str]:
    first = group.rows[0] if group.rows else {}
    return (
        str(_mapping(first.get("provenance")).get("source_path", "")),
        _int(first.get("record_index"), default=-1),
        group.group_key,
    )


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, int, str, str]:
    return (
        _oracle_rank(row),
        _action_index(str(row.get("candidate_action", ""))),
        str(row.get("candidate_action", "")),
        str(row.get("archive_row_id", "")),
    )


def _same_archive_row(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return str(left.get("archive_row_id")) == str(right.get("archive_row_id"))


def _action_index(action: str) -> int:
    try:
        return ACTION_NAMES.index(action)
    except ValueError:
        return len(ACTION_NAMES) + 1


def _nested_number(payload: Mapping[str, object], path: Sequence[str]) -> float:
    current: object = payload
    for key in path:
        current = _mapping(current).get(key)
    return _number(current)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _file_sha256(path: str | Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _counter_to_ordered_dict(counter: Mapping[str, int] | Counter[str]) -> dict[str, int]:
    return {
        str(key): int(counter[key])
        for key in sorted(counter, key=lambda item: (str(item)))
    }


def _dominant_count_share(counts: Mapping[str, int] | Counter[str]) -> dict[str, object]:
    if not counts:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0]
    return {
        "key": key,
        "count": int(count),
        "share": _share(int(count), sum(int(value) for value in counts.values())),
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _list_mut(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    payload[key] = []
    return payload[key]  # type: ignore[return-value]


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _int_or_none(value: object) -> int | None:
    parsed = _int(value, default=-10**12)
    return None if parsed == -10**12 else parsed


def _positive_int(value: object, *, field: str) -> int:
    parsed = _int(value, default=-1)
    if parsed <= 0:
        raise FirstRecoveryShadowRankerError(f"{field} must be a positive integer")
    return parsed


def _number(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _round(value: float) -> float:
    return round(float(value), 6)


def _mean(values: Iterable[float | None]) -> float:
    parsed = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not parsed:
        return 0.0
    return sum(parsed) / float(len(parsed))


def _weighted_mean(values: Iterable[tuple[float, int]]) -> float:
    total_weight = 0
    total = 0.0
    for value, weight in values:
        if weight <= 0:
            continue
        total += float(value) * int(weight)
        total_weight += int(weight)
    return total / float(total_weight) if total_weight else 0.0


def _share(numerator: int, denominator: int) -> float:
    return _round(float(numerator) / float(denominator)) if denominator else 0.0


def _share_float(numerator: float, denominator: int) -> float:
    return _round(float(numerator) / float(denominator)) if denominator else 0.0


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))
