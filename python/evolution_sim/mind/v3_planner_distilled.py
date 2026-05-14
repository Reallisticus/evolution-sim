from __future__ import annotations

import gzip
import json
import math
import zlib
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    LOCAL_PATCH_RADIUS,
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)

MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION = (
    "mind_v3_planner_distilled_action_scorer_artifact_v1"
)
MIND_V3_PLANNER_DISTILLED_RUNTIME_POLICY = (
    "mind_v3_planner_distilled_runtime_scorer_v1"
)
V96_DEFAULT_SEQUENCE_NEIGHBOR_COUNT = 5
V96_DEFAULT_TEACHER_IMITATION_NEIGHBOR_COUNT = 5
MIND_V3_PLANNER_DISTILLED_HISTORY_STEPS = 8
_SELF_FIELD_INDEX = {field: index for index, field in enumerate(SELF_INPUT_FIELDS)}
_PATCH_FIELD_INDEX = {field: index for index, field in enumerate(PATCH_INPUT_FIELDS)}
_NAVIGATION_FIELD_INDEX = {
    field: index for index, field in enumerate(NAVIGATION_INPUT_FIELDS)
}
_PATCH_INPUT_START = len(SELF_INPUT_FIELDS)
_PATCH_STRIDE = len(PATCH_INPUT_FIELDS)
_NAVIGATION_INPUT_START = _PATCH_INPUT_START + PATCH_CELL_COUNT * _PATCH_STRIDE
_NAVIGATION_STRIDE = len(NAVIGATION_INPUT_FIELDS)
_CENTER_PATCH_INDEX = PATCH_CELL_COUNT // 2
_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}


class MindV3PlannerDistilledArtifactError(ValueError):
    pass


def load_mind_v3_planner_distilled_artifact(
    source: str | Path | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            with _open_input(path) as handle:
                loaded = json.load(handle)
        except OSError as exc:
            raise MindV3PlannerDistilledArtifactError(
                f"failed to read planner distilled artifact: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise MindV3PlannerDistilledArtifactError(
                f"planner distilled artifact is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(loaded, dict):
            raise MindV3PlannerDistilledArtifactError(
                "planner distilled artifact payload must be a JSON object"
            )
        payload = loaded
    artifact = payload.get("distilled_artifact")
    if isinstance(artifact, Mapping):
        payload = dict(artifact)
    validate_mind_v3_planner_distilled_artifact(payload)
    return dict(payload)


def validate_mind_v3_planner_distilled_artifact(
    artifact: Mapping[str, object],
) -> None:
    if (
        artifact.get("schema_version")
        != MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION
    ):
        raise MindV3PlannerDistilledArtifactError(
            "planner distilled artifact has unsupported schema_version"
        )
    inference = _mapping(artifact.get("inference_contract"))
    required_contract = {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_planner_outcome_tables": False,
        "requires_global_batch_assignment": False,
        "uses_heuristic_fallback": False,
    }
    for key, expected in required_contract.items():
        if inference.get(key) is not expected:
            raise MindV3PlannerDistilledArtifactError(
                f"planner distilled artifact violates inference_contract.{key}"
            )
    if artifact_has_forbidden_example_keys(artifact):
        raise MindV3PlannerDistilledArtifactError(
            "planner distilled artifact examples contain forbidden runtime keys"
        )
    _validate_example_section(artifact, "sequence_support_examples")
    _validate_example_section(artifact, "teacher_imitation_examples")


def artifact_has_forbidden_example_keys(artifact: Mapping[str, object]) -> bool:
    forbidden = {
        "seed",
        "seed_id",
        "branch_id",
        "fixture",
        "fixture_identity",
        "source",
        "logged_action",
        "strict_eval_label_identity",
        "planner_candidate_outcome_table",
        "global_batch_action_quota",
        "private_simulation_world_state",
    }
    for section in ("sequence_support_examples", "teacher_imitation_examples"):
        for item in _list_of_mappings(artifact.get(section)):
            if set(item) & forbidden:
                return True
    return False


def planner_distilled_runtime_row(
    *,
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    public_history_trace: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    values = _decode_policy_observation_values(observation_input)
    compact_state: dict[str, object] = {}
    if values:
        compact_state = _compact_state_from_values(values)
    complete_mask = _complete_action_mask(action_mask)
    return {
        "action_mask": complete_mask,
        "compact_state": compact_state,
        "observation_values": tuple(_round(value) for value in values)
        if compact_state
        else (),
        "policy_observation_values": tuple(_round(value) for value in values),
        "public_history_trace": [
            _public_history_item(item) for item in public_history_trace
        ],
    }


def score_mind_v3_planner_distilled_runtime(
    *,
    artifact: Mapping[str, object],
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    public_history_trace: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    row = planner_distilled_runtime_row(
        observation_input=observation_input,
        action_mask=action_mask,
        public_history_trace=public_history_trace,
    )
    scored = score_distilled_planner_artifact(row=row, artifact=artifact)
    if scored.get("selected_action") is None:
        raise MindV3PlannerDistilledArtifactError(
            "planner distilled scorer produced no legal action"
        )
    selected = str(scored["selected_action"])
    if not bool(_mapping(action_mask).get(selected, False)):
        raise MindV3PlannerDistilledArtifactError(
            f"planner distilled scorer selected unsupported action: {selected}"
        )
    candidate_scores = _list_of_mappings(scored.get("candidate_scores"))
    top_scores = sorted(
        candidate_scores,
        key=lambda item: (
            _float(item.get("final_score")),
            _float(item.get("sequence_cvar_score")),
            str(item.get("action", "")),
        ),
        reverse=True,
    )
    margin = 0.0
    if len(top_scores) >= 2:
        margin = _round(
            _float(top_scores[0].get("final_score"))
            - _float(top_scores[1].get("final_score"))
        )
    return {
        "selected_action": selected,
        "selected_mode": scored.get("selected_mode"),
        "selected_score": _float(top_scores[0].get("final_score"))
        if top_scores
        else 0.0,
        "score_margin": margin,
        "scores": {
            str(item.get("action", "")): _float(item.get("final_score"))
            for item in candidate_scores
            if str(item.get("action", "")) in ACTION_NAMES
        },
        "candidate_scores": candidate_scores,
        "diagnostics": planner_distilled_runtime_diagnostics(
            artifact=artifact,
            scored=scored,
            public_history_trace=public_history_trace,
            score_margin=margin,
        ),
    }


def planner_distilled_runtime_diagnostics(
    *,
    artifact: Mapping[str, object],
    scored: Mapping[str, object],
    public_history_trace: Sequence[Mapping[str, object]],
    score_margin: float,
) -> dict[str, object]:
    candidate_scores = _list_of_mappings(scored.get("candidate_scores"))
    sorted_scores = sorted(
        candidate_scores,
        key=lambda item: (
            _float(item.get("final_score")),
            _float(item.get("sequence_cvar_score")),
            str(item.get("action", "")),
        ),
        reverse=True,
    )
    return {
        "planner_distilled_runtime_policy": MIND_V3_PLANNER_DISTILLED_RUNTIME_POLICY,
        "planner_distilled_artifact_schema_version": artifact.get(
            "schema_version"
        ),
        "planner_distilled_model_type": artifact.get("model_type"),
        "planner_distilled_selected_action": scored.get("selected_action"),
        "planner_distilled_selected_mode": scored.get("selected_mode"),
        "planner_distilled_score_margin": _round(score_margin),
        "planner_distilled_candidate_score_count": len(candidate_scores),
        "planner_distilled_candidate_scores_top": [
            _candidate_score_summary(item) for item in sorted_scores[:5]
        ],
        "planner_distilled_public_history_steps": len(public_history_trace),
        "planner_distilled_forbidden_runtime_inputs_used": False,
        "planner_distilled_requires_planner_outcome_tables": False,
        "planner_distilled_requires_global_batch_assignment": False,
        "planner_distilled_uses_heuristic_fallback": False,
    }


def score_distilled_planner_artifact(
    *,
    row: Mapping[str, object],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    validate_mind_v3_planner_distilled_artifact(artifact)
    candidate_scores = _artifact_candidate_scores(row=row, artifact=artifact)
    if not candidate_scores:
        return {
            "selected_action": None,
            "candidate_scores": [],
            "unsupported_predicted_action": True,
        }
    selected = max(
        candidate_scores,
        key=lambda item: (
            _float(item.get("final_score")),
            _float(item.get("sequence_cvar_score")),
            str(item.get("action", "")),
        ),
    )
    return {
        "selected_action": selected.get("action"),
        "selected_mode": selected.get("mode"),
        "candidate_scores": candidate_scores,
        "unsupported_predicted_action": str(selected.get("action", ""))
        not in ACTION_NAMES,
    }


def candidate_feature_vector(
    row: Mapping[str, object],
    action: str,
) -> tuple[float, ...]:
    archive_features = _branch_continuation_archive_feature_vector(
        row=row,
        action=action,
    )
    if archive_features:
        return archive_features
    values = row.get("policy_observation_values") or row.get("observation_values")
    if not isinstance(values, (tuple, list)):
        return ()
    action_mask = _mapping(row.get("action_mask"))
    supported_count = sum(1 for value in action_mask.values() if bool(value))
    dx, dy = _MOVE_DELTAS.get(action, (0, 0))
    history = _list_of_mappings(row.get("public_history_trace"))
    features = [
        *[float(value) for value in values],
        *[1.0 if action == name else 0.0 for name in ACTION_NAMES],
        float(dx),
        float(dy),
        1.0 if bool(action_mask.get(action, False)) else 0.0,
        min(float(supported_count) / max(float(len(ACTION_NAMES)), 1.0), 1.0),
        *_public_history_feature_vector(history),
    ]
    return tuple(_round(value) for value in features)


def _branch_continuation_archive_feature_vector(
    *,
    row: Mapping[str, object],
    action: str,
) -> tuple[float, ...]:
    state = _mapping(row.get("compact_state"))
    if not state:
        return ()
    self_state = _mapping(state.get("self"))
    center = _mapping(state.get("center"))
    local = _mapping(state.get("local"))
    navigation = _mapping(state.get("navigation"))
    adjacent = _mapping(state.get("adjacent"))
    if not self_state or not center:
        return ()
    action_mask = _mapping(row.get("action_mask"))
    action_dx, action_dy = _MOVE_DELTAS.get(action, (0, 0))
    target_cell = _target_cell_for_action(action, adjacent)
    current_carrion = _cell_carrion(center)
    target_carrion = _cell_carrion(target_cell)
    history = _list_of_mappings(row.get("public_history_trace"))
    supported_count = sum(1 for value in action_mask.values() if bool(value))
    features = tuple(
        _round(value)
        for value in (
            _feature_float(self_state.get("energy_ratio")),
            _feature_float(self_state.get("hydration_ratio")),
            _feature_float(self_state.get("health_ratio")),
            1.0 - _feature_float(self_state.get("energy_ratio")),
            1.0 - _feature_float(self_state.get("hydration_ratio")),
            1.0 - _feature_float(self_state.get("health_ratio")),
            _feature_float(self_state.get("injury_load")),
            _feature_float(self_state.get("trophic_role_code")),
            _feature_float(self_state.get("meat_mode_code")),
            _feature_float(center.get("water")),
            _feature_float(center.get("food")),
            current_carrion,
            _cell_risk(center),
            _feature_float(local.get("radius1_water")),
            _feature_float(local.get("radius1_food")),
            _feature_float(local.get("radius1_carrion")),
            _feature_float(local.get("radius1_risk")),
            _feature_float(local.get("radius2_water")),
            _feature_float(local.get("radius2_food")),
            _feature_float(local.get("radius2_carrion")),
            _feature_float(local.get("radius2_risk")),
            *_navigation_features(navigation),
            *_action_one_hot(action),
            _round(action_dx),
            _round(action_dy),
            1.0 if bool(action_mask.get(action, False)) else 0.0,
            min(float(supported_count) / max(float(len(ACTION_NAMES)), 1.0), 1.0),
            1.0 if action in _MOVE_DELTAS else 0.0,
            1.0 if action == "drink" else 0.0,
            1.0 if action == "eat" else 0.0,
            1.0 if action == "stay" else 0.0,
            action_dx * _feature_float(target_cell.get("water")),
            action_dy * _feature_float(target_cell.get("water")),
            _feature_float(target_cell.get("food")),
            target_carrion,
            _cell_risk(target_cell),
            *_movement_navigation_alignment(action_dx, action_dy, navigation),
            *_public_history_summary_feature_vector(history),
        )
    )
    return tuple(_archive_quantize_feature(value) for value in features)


def _compact_state_from_values(values: Sequence[float]) -> dict[str, object]:
    expected_size = _NAVIGATION_INPUT_START + len(NAVIGATION_TARGETS) * (
        _NAVIGATION_STRIDE
    )
    if len(values) != expected_size:
        return {}
    cells = [_patch_cell_from_values(values, index) for index in range(PATCH_CELL_COUNT)]
    center = next(
        (
            cell
            for cell in cells
            if _int(cell.get("dx")) == 0 and _int(cell.get("dy")) == 0
        ),
        cells[_CENTER_PATCH_INDEX],
    )
    adjacent = {
        _cell_direction(cell): cell
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) == 1
    }
    local = _local_summaries(cells)
    navigation = {
        target: _navigation_state(values, target_index)
        for target_index, target in enumerate(NAVIGATION_TARGETS)
    }
    self_state = {
        field: _round(float(values[index]))
        for field, index in _SELF_FIELD_INDEX.items()
    }
    return {
        "self": self_state,
        "center": center,
        "adjacent": adjacent,
        "local": local,
        "navigation": navigation,
    }


def _patch_cell_from_values(
    values: Sequence[float],
    cell_index: int,
) -> dict[str, object]:
    base = _PATCH_INPUT_START + cell_index * _PATCH_STRIDE
    return {
        "dx": _round_int(
            float(values[base + _PATCH_FIELD_INDEX["dx"]]) * LOCAL_PATCH_RADIUS
        ),
        "dy": _round_int(
            float(values[base + _PATCH_FIELD_INDEX["dy"]]) * LOCAL_PATCH_RADIUS
        ),
        "in_bounds": float(values[base + _PATCH_FIELD_INDEX["in_bounds"]]),
        "terrain_code": float(values[base + _PATCH_FIELD_INDEX["terrain_code"]]),
        "occupant_code": float(values[base + _PATCH_FIELD_INDEX["occupant_code"]]),
        "same_lineage": float(values[base + _PATCH_FIELD_INDEX["same_lineage"]]),
        "water": max(
            float(values[base + _PATCH_FIELD_INDEX["water_access_reason_code"]]),
            1.0
            if float(values[base + _PATCH_FIELD_INDEX["terrain_code"]]) >= 0.99
            else 0.0,
        ),
        "food": float(values[base + _PATCH_FIELD_INDEX["food"]]),
        "vegetation": float(values[base + _PATCH_FIELD_INDEX["vegetation"]]),
        "recovery_debt": float(values[base + _PATCH_FIELD_INDEX["recovery_debt"]]),
        "fresh_kill": float(values[base + _PATCH_FIELD_INDEX["fresh_kill_energy"]]),
        "carcass": float(values[base + _PATCH_FIELD_INDEX["carcass_energy"]]),
        "hazard_type_code": float(
            values[base + _PATCH_FIELD_INDEX["hazard_type_code"]]
        ),
        "hazard_level": float(values[base + _PATCH_FIELD_INDEX["hazard_level"]]),
        "prey_biomass": float(values[base + _PATCH_FIELD_INDEX["prey_biomass"]]),
        "carrion_signal": float(values[base + _PATCH_FIELD_INDEX["carrion_signal"]]),
        "predator_risk": float(values[base + _PATCH_FIELD_INDEX["predator_risk"]]),
    }


def _navigation_state(
    values: Sequence[float],
    target_index: int,
) -> dict[str, object]:
    base = _NAVIGATION_INPUT_START + target_index * _NAVIGATION_STRIDE
    return {
        "dx": float(values[base + _NAVIGATION_FIELD_INDEX["dx"]]),
        "dy": float(values[base + _NAVIGATION_FIELD_INDEX["dy"]]),
        "distance": float(values[base + _NAVIGATION_FIELD_INDEX["distance"]]),
        "strength": float(values[base + _NAVIGATION_FIELD_INDEX["strength"]]),
    }


def _local_summaries(
    cells: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    return {
        "radius1_food": _max_cell_signal(cells, "food", radius=1),
        "radius1_water": _max_cell_signal(cells, "water", radius=1),
        "radius1_carrion": _max_carrion_signal(cells, radius=1),
        "radius1_prey": _max_cell_signal(cells, "prey_biomass", radius=1),
        "radius1_risk": _max_risk_signal(cells, radius=1),
        "radius2_food": _max_cell_signal(cells, "food", radius=2),
        "radius2_water": _max_cell_signal(cells, "water", radius=2),
        "radius2_carrion": _max_carrion_signal(cells, radius=2),
        "radius2_prey": _max_cell_signal(cells, "prey_biomass", radius=2),
        "radius2_risk": _max_risk_signal(cells, radius=2),
    }


def _navigation_features(navigation: Mapping[str, object]) -> tuple[float, ...]:
    features: list[float] = []
    for target in NAVIGATION_TARGETS:
        item = _mapping(navigation.get(target))
        features.extend(
            [
                _feature_float(item.get("dx")),
                _feature_float(item.get("dy")),
                _feature_float(item.get("distance")),
                _feature_float(item.get("strength")),
            ]
        )
    return tuple(features)


def _movement_navigation_alignment(
    action_dx: int,
    action_dy: int,
    navigation: Mapping[str, object],
) -> tuple[float, ...]:
    if action_dx == 0 and action_dy == 0:
        return tuple(0.0 for _ in NAVIGATION_TARGETS)
    values = []
    for target in NAVIGATION_TARGETS:
        item = _mapping(navigation.get(target))
        dx = _feature_float(item.get("dx"))
        dy = _feature_float(item.get("dy"))
        strength = _feature_float(item.get("strength"))
        aligned = max(0.0, action_dx * dx + action_dy * dy)
        values.append(_round(aligned * strength))
    return tuple(values)


def _target_cell_for_action(
    action: str,
    adjacent: Mapping[str, object],
) -> Mapping[str, object]:
    direction = {
        "move_north": "north",
        "move_south": "south",
        "move_east": "east",
        "move_west": "west",
    }.get(action)
    if direction is None:
        return {}
    return _mapping(adjacent.get(direction))


def _cell_direction(cell: Mapping[str, object]) -> str:
    dx = _int(cell.get("dx"))
    dy = _int(cell.get("dy"))
    if dx == 0 and dy < 0:
        return "north"
    if dx == 0 and dy > 0:
        return "south"
    if dx > 0 and dy == 0:
        return "east"
    if dx < 0 and dy == 0:
        return "west"
    return "center"


def _max_cell_signal(
    cells: Sequence[Mapping[str, object]],
    field: str,
    *,
    radius: int,
) -> float:
    values = [
        _feature_float(cell.get(field))
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) <= radius
    ]
    return max(values) if values else 0.0


def _max_carrion_signal(
    cells: Sequence[Mapping[str, object]],
    *,
    radius: int,
) -> float:
    values = [
        _cell_carrion(cell)
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) <= radius
    ]
    return max(values) if values else 0.0


def _max_risk_signal(
    cells: Sequence[Mapping[str, object]],
    *,
    radius: int,
) -> float:
    values = [
        _cell_risk(cell)
        for cell in cells
        if abs(_int(cell.get("dx"))) + abs(_int(cell.get("dy"))) <= radius
    ]
    return max(values) if values else 0.0


def _cell_carrion(cell: Mapping[str, object]) -> float:
    return max(
        _feature_float(cell.get("fresh_kill")),
        _feature_float(cell.get("carcass")),
        _feature_float(cell.get("carrion_signal")),
    )


def _cell_risk(cell: Mapping[str, object]) -> float:
    return max(
        _feature_float(cell.get("hazard_level")),
        _feature_float(cell.get("predator_risk")),
    )


def _public_history_summary_feature_vector(
    history: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    if not history:
        return (0.0,) * 16
    count = float(len(history))
    latest = _mapping(history[-1])
    return tuple(
        _round(value)
        for value in (
            min(count / float(MIND_V3_PLANNER_DISTILLED_HISTORY_STEPS), 1.0),
            sum(1.0 for item in history if bool(item.get("moved", False))) / count,
            sum(1.0 for item in history if bool(item.get("drank", False))) / count,
            sum(1.0 for item in history if bool(item.get("ate", False))) / count,
            sum(
                1.0
                for item in history
                if bool(item.get("post_carrion_contact", False))
            )
            / count,
            sum(_feature_float(item.get("resource_gain")) for item in history) / count,
            _feature_float(latest.get("energy_ratio_after")),
            _clamped_delta(latest.get("energy_ratio_delta")),
            _feature_float(latest.get("hydration_ratio_after")),
            _clamped_delta(latest.get("hydration_ratio_delta")),
            _feature_float(latest.get("health_ratio_after")),
            _clamped_delta(latest.get("health_ratio_delta")),
            min(_feature_float(latest.get("tick_delta")) / 120.0, 1.0),
            min(
                _feature_float(latest.get("ticks_since_animal_resource_gain")) / 32.0,
                1.0,
            ),
            min(_feature_float(latest.get("ticks_since_drink")) / 32.0, 1.0),
            _clamped_delta(latest.get("x_delta"))
            + _clamped_delta(latest.get("y_delta")),
        )
    )


def _public_history_feature_vector(
    history: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    selected = list(history)[-MIND_V3_PLANNER_DISTILLED_HISTORY_STEPS:]
    padded: list[Mapping[str, object] | None] = [None] * (
        MIND_V3_PLANNER_DISTILLED_HISTORY_STEPS - len(selected)
    )
    padded.extend(selected)
    values: list[float] = []
    empty_slot = (0.0,) * len(_public_history_item_features({}))
    for item in padded:
        if item is None:
            values.extend(empty_slot)
            continue
        values.extend(_public_history_item_features(item))
    return tuple(_round(value) for value in values)


def _public_history_item_features(item: Mapping[str, object]) -> tuple[float, ...]:
    return tuple(
        _round(value)
        for value in (
            1.0,
            min(_feature_float(item.get("tick_delta")) / 120.0, 1.0),
            min(_feature_float(item.get("record_index_delta")) / 256.0, 1.0),
            *_action_one_hot(_optional_string(item.get("requested_action"))),
            *_action_one_hot(_optional_string(item.get("resolved_action"))),
            1.0 if bool(item.get("action_valid", False)) else 0.0,
            1.0 if bool(item.get("resolution_action_valid", False)) else 0.0,
            1.0 if bool(item.get("moved", False)) else 0.0,
            1.0 if bool(item.get("drank", False)) else 0.0,
            1.0 if bool(item.get("ate", False)) else 0.0,
            1.0 if bool(item.get("died", False)) else 0.0,
            1.0 if bool(item.get("died_after_action", False)) else 0.0,
            1.0 if bool(item.get("post_carrion_contact", False)) else 0.0,
            _clamped_delta(item.get("x_delta")),
            _clamped_delta(item.get("y_delta")),
            _feature_float(item.get("resource_gain")),
            _feature_float(item.get("energy_ratio_after")),
            _clamped_delta(item.get("energy_ratio_delta")),
            _feature_float(item.get("hydration_ratio_after")),
            _clamped_delta(item.get("hydration_ratio_delta")),
            _feature_float(item.get("health_ratio_after")),
            _clamped_delta(item.get("health_ratio_delta")),
            min(
                _feature_float(item.get("ticks_since_animal_resource_gain")) / 32.0,
                1.0,
            ),
            min(_feature_float(item.get("ticks_since_drink")) / 32.0, 1.0),
        )
    )


def _action_one_hot(action: str | None) -> tuple[float, ...]:
    return tuple(1.0 if action == name else 0.0 for name in ACTION_NAMES)


def _action_option_mode(action: str) -> str:
    if action == "drink":
        return "recover_hydration"
    if action == "eat":
        return "exploit_resource"
    if action == "stay":
        return "conserve"
    if action.startswith("move_"):
        return "reposition"
    return "other"


def _artifact_candidate_scores(
    *,
    row: Mapping[str, object],
    artifact: Mapping[str, object],
) -> list[dict[str, object]]:
    sequence_examples = _list_of_mappings(artifact.get("sequence_support_examples"))
    teacher_examples = _list_of_mappings(artifact.get("teacher_imitation_examples"))
    action_penalties = _mapping(artifact.get("learned_action_penalties"))
    scoring = _mapping(artifact.get("scoring_policy"))
    teacher_weight = _float(scoring.get("teacher_imitation_weight"))
    scores = []
    for action in _candidate_actions(row):
        features = candidate_feature_vector(row, action)
        if not features:
            continue
        sequence_neighbors = _nearest_artifact_examples(
            features,
            sequence_examples,
            k=_int(scoring.get("sequence_neighbor_count"))
            or V96_DEFAULT_SEQUENCE_NEIGHBOR_COUNT,
        )
        if not sequence_neighbors:
            continue
        sequence_predicted = _predicted_sequence_stats(sequence_neighbors)
        sequence_cvar = _float(
            sequence_predicted.get("target_local_sequence_score_cvar_25")
        )
        imitation_margin = _teacher_imitation_margin(
            features,
            teacher_examples,
            k=_int(scoring.get("teacher_imitation_neighbor_count"))
            or V96_DEFAULT_TEACHER_IMITATION_NEIGHBOR_COUNT,
        )
        action_penalty = _float(action_penalties.get(action))
        final_score = sequence_cvar + teacher_weight * imitation_margin - action_penalty
        scores.append(
            {
                "action": action,
                "mode": _action_option_mode(action),
                "sequence_cvar_score": _round(sequence_cvar),
                "utility_weighted_teacher_margin": _round(imitation_margin),
                "learned_action_penalty": _round(action_penalty),
                "final_score": _round(final_score),
                "sequence_neighbor_count": len(sequence_neighbors),
            }
        )
    return sorted(scores, key=lambda item: str(item.get("action", "")))


def _candidate_actions(row: Mapping[str, object]) -> list[str]:
    action_mask = _mapping(row.get("action_mask"))
    actions_from_values = []
    raw_values = row.get("action_values")
    if isinstance(raw_values, list):
        actions_from_values = [
            str(item.get("action", ""))
            for item in raw_values
            if isinstance(item, Mapping)
        ]
    if not actions_from_values:
        actions_from_values = list(ACTION_NAMES)
    return sorted(
        action
        for action in actions_from_values
        if action in ACTION_NAMES and bool(action_mask.get(action, False))
    )


def _nearest_artifact_examples(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> list[tuple[float, Mapping[str, object]]]:
    neighbors = [
        (_squared_distance(features, _tuple(item.get("features"))), item)
        for item in examples
        if item.get("features")
    ]
    neighbors.sort(
        key=lambda item: (
            item[0],
            _int(item[1].get("example_index")),
            str(item[1].get("action", "")),
        )
    )
    return neighbors[: max(1, int(k))]


def _teacher_imitation_margin(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> float:
    if not examples:
        return 0.0
    neighbors = _nearest_artifact_examples(features, examples, k=k)
    weighted_total = 0.0
    weight_sum = 0.0
    for distance, item in neighbors:
        distance_weight = 1.0 / (1.0 + float(distance))
        utility_weight = max(0.0, _float(item.get("utility_weight")))
        signed = 1.0 if item.get("teacher_selected") is True else -1.0
        weight = distance_weight * utility_weight
        weighted_total += weight * signed
        weight_sum += weight
    return _round(weighted_total / weight_sum) if weight_sum else 0.0


def _predicted_sequence_stats(
    selected: Sequence[tuple[float, Mapping[str, object]]],
) -> dict[str, object]:
    metrics = [_mapping(item[1].get("metrics")) for item in selected]

    def mean(field: str) -> float:
        return _mean([_float(item.get(field)) for item in metrics])

    def cvar(field: str) -> float:
        return _cvar([_float(item.get(field)) for item in metrics], fraction=0.25)

    target_alive = mean("terminal_target_alive")
    first_survival = mean("first_action_survived")
    after_first_survival = mean("target_survival_area_after_first")
    short_survival = mean("target_short_horizon_survival_area")
    min_vital = mean("target_min_vital_after_first")
    vital_envelope = mean("vital_envelope_score")
    target_score = mean("target_local_score")
    target_score_cvar = cvar("target_local_score")
    population_alive = mean("terminal_alive_agents")
    births = mean("births")
    population_score = population_alive * 10.0 + births
    combined = (
        target_alive * 1000.0
        + first_survival * 250.0
        + after_first_survival * 200.0
        + short_survival * 150.0
        + vital_envelope * 100.0
        + target_score_cvar * 0.1
        + population_score
    )
    return {
        "terminal_target_alive_probability": _round(target_alive),
        "first_action_survival_probability": _round(first_survival),
        "target_survival_area_after_first_mean": _round(after_first_survival),
        "target_short_horizon_survival_area_mean": _round(short_survival),
        "target_min_vital_after_first_mean": _round(min_vital),
        "vital_envelope_score_mean": _round(vital_envelope),
        "target_resource_gain_after_first_mean": _round(
            mean("target_resource_gain_after_first")
        ),
        "first_action_resource_gain_mean": _round(
            mean("first_action_resource_gain")
        ),
        "target_local_sequence_score_mean": _round(target_score),
        "target_local_sequence_score_cvar_25": _round(target_score_cvar),
        "terminal_alive_agents_mean": _round(population_alive),
        "births_mean": _round(births),
        "population_birth_continuation_score": _round(population_score),
        "combined_target_survival_first_score": _round(combined),
        "first_action_death_risk": _round(1.0 - first_survival),
    }


def _validate_example_section(
    artifact: Mapping[str, object],
    section: str,
) -> None:
    examples = _list_of_mappings(artifact.get(section))
    if not examples:
        raise MindV3PlannerDistilledArtifactError(
            f"planner distilled artifact has no {section}"
        )
    feature_lengths = {
        len(_tuple(item.get("features"))) for item in examples if item.get("features")
    }
    if not feature_lengths:
        raise MindV3PlannerDistilledArtifactError(
            f"planner distilled artifact {section} has no feature vectors"
        )
    if len(feature_lengths) != 1:
        raise MindV3PlannerDistilledArtifactError(
            f"planner distilled artifact {section} feature lengths are inconsistent"
        )


def _decode_policy_observation_values(
    observation_input: Mapping[str, object],
) -> tuple[float, ...]:
    try:
        return tuple(
            _round(float(value))
            for value in decode_observation_input(dict(observation_input))
        )
    except (ValueError, TypeError, zlib.error):
        return ()


def _complete_action_mask(raw: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(raw.get(action, False)) for action in ACTION_NAMES}


def _public_history_item(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "tick": _int(item.get("tick")),
        "tick_delta": _int(item.get("tick_delta")),
        "record_index": _int(item.get("record_index")),
        "record_index_delta": _int(item.get("record_index_delta")),
        "requested_action": _optional_string(item.get("requested_action")),
        "resolved_action": _optional_string(item.get("resolved_action")),
        "action_valid": bool(item.get("action_valid", False)),
        "resolution_action_valid": bool(
            item.get("resolution_action_valid", False)
        ),
        "moved": bool(item.get("moved", False)),
        "x_delta": _int(item.get("x_delta")),
        "y_delta": _int(item.get("y_delta")),
        "energy_ratio_before": _optional_float(item.get("energy_ratio_before")),
        "energy_ratio_after": _optional_float(item.get("energy_ratio_after")),
        "energy_ratio_delta": _optional_float(item.get("energy_ratio_delta")),
        "hydration_ratio_before": _optional_float(
            item.get("hydration_ratio_before")
        ),
        "hydration_ratio_after": _optional_float(item.get("hydration_ratio_after")),
        "hydration_ratio_delta": _optional_float(item.get("hydration_ratio_delta")),
        "health_ratio_before": _optional_float(item.get("health_ratio_before")),
        "health_ratio_after": _optional_float(item.get("health_ratio_after")),
        "health_ratio_delta": _optional_float(item.get("health_ratio_delta")),
        "resource_gain": _optional_float(item.get("resource_gain")),
        "drank": bool(item.get("drank", False)),
        "ate": bool(item.get("ate", False)),
        "died": bool(item.get("died", False)),
        "death_cause": _optional_string(item.get("death_cause")),
        "died_after_action": bool(item.get("died_after_action", False)),
        "post_carrion_contact": bool(item.get("post_carrion_contact", False)),
        "ticks_since_animal_resource_gain": _optional_float(
            item.get("ticks_since_animal_resource_gain")
        ),
        "ticks_since_drink": _optional_float(item.get("ticks_since_drink")),
    }


def _candidate_score_summary(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "action": str(item.get("action", "")),
        "mode": str(item.get("mode", "")),
        "final_score": _float(item.get("final_score")),
        "sequence_cvar_score": _float(item.get("sequence_cvar_score")),
        "utility_weighted_teacher_margin": _float(
            item.get("utility_weighted_teacher_margin")
        ),
        "learned_action_penalty": _float(item.get("learned_action_penalty")),
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(float(item) for item in value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _cvar(values: Sequence[float], *, fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    count = max(1, int(math.ceil(float(len(values)) * fraction)))
    return _mean(sorted_values[:count])


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _feature_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _clamped_delta(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_string(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _round(value: float) -> float:
    return round(float(value), 6)


def _round_int(value: float) -> int:
    return int(round(float(value)))


def _archive_quantize_feature(value: float) -> float:
    return _round(round(float(value) * 8.0) / 8.0)


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
