from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.cli.mind_v3_evaluate import (
    _aggregate_runs,
    _fixture_world,
    _run_world,
)
from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.policy import (
    ActionDecision,
    _cell_at,
    _float,
    _navigation,
    _navigation_action,
    _patch_cells,
    _self_state,
    _target_distance,
    _valid,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION = (
    "mind_v3_carrion_counterfactual_rollout_v1"
)
MIND_V3_CARRION_COUNTERFACTUAL_POLICY_VERSION = (
    "scripted_policy_visible_carrion_water_recovery_v1"
)
DEFAULT_COUNTERFACTUAL_SCRIPTS: tuple[str, ...] = (
    "carrion_then_water",
    "water_first_recovery",
    "conserve_after_carrion",
    "hydration_safe_carrion_cycle",
)
DEFAULT_CARRION_COUNTERFACTUAL_SEEDS: tuple[int, ...] = (29, 37)
DEFAULT_CARRION_COUNTERFACTUAL_TICKS = 120


class CarrionCounterfactualError(ValueError):
    pass


@dataclass(slots=True)
class CarrionCounterfactualPolicy:
    script_name: str
    policy_id: str = field(init=False)
    policy_version: str = MIND_V3_CARRION_COUNTERFACTUAL_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.script_name not in DEFAULT_COUNTERFACTUAL_SCRIPTS:
            raise CarrionCounterfactualError(
                f"unsupported counterfactual script: {self.script_name}"
            )
        self.policy_id = f"mind_v3_counterfactual_{self.script_name}"

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        facts = _PolicyVisibleFacts.from_observation(observation)
        action, reason = self._script_action(facts, action_mask)
        if action not in ACTION_NAMES or not _valid(action_mask, action):
            action, reason = _fallback_action(action_mask), f"{reason}:fallback"
        return ActionDecision(
            requested_action=action,
            source=f"counterfactual_script:{self.script_name}",
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics={
                "script": self.script_name,
                "reason": reason,
                "energy_ratio": facts.energy,
                "hydration_ratio": facts.hydration,
                "health_ratio": facts.health,
                "center_animal_resource": facts.center_animal_resource,
                "center_food": facts.center_food,
                "carrion_distance": facts.carrion_distance,
                "water_distance": facts.water_distance,
            },
        )

    def _script_action(
        self,
        facts: "_PolicyVisibleFacts",
        action_mask: dict[str, bool],
    ) -> tuple[str, str]:
        if self.script_name == "carrion_then_water":
            return _carrion_then_water(facts, action_mask)
        if self.script_name == "water_first_recovery":
            return _water_first_recovery(facts, action_mask)
        if self.script_name == "conserve_after_carrion":
            return _conserve_after_carrion(facts, action_mask)
        if self.script_name == "hydration_safe_carrion_cycle":
            return _hydration_safe_carrion_cycle(facts, action_mask)
        raise CarrionCounterfactualError(
            f"unsupported counterfactual script: {self.script_name}"
        )


@dataclass(frozen=True, slots=True)
class _PolicyVisibleFacts:
    energy: float
    hydration: float
    health: float
    matched_diet: float
    center_animal_resource: float
    center_food: float
    carrion_distance: int
    carrion_strength: float
    water_distance: int
    water_strength: float
    carrion_target: dict[str, object]
    water_target: dict[str, object]

    @classmethod
    def from_observation(cls, observation: dict[str, object]) -> "_PolicyVisibleFacts":
        self_state = _self_state(observation)
        patch_cells = _patch_cells(observation)
        navigation = _navigation(observation)
        center = _cell_at(patch_cells, 0, 0) or {}
        carrion_target = navigation["carrion"]
        water_target = navigation["water"]
        return cls(
            energy=_float(self_state.get("energy_ratio", 0.0)),
            hydration=_float(self_state.get("hydration_ratio", 0.0)),
            health=_float(self_state.get("health_ratio", 0.0)),
            matched_diet=_float(self_state.get("matched_diet_ratio", 0.0)),
            center_animal_resource=_animal_resource(center),
            center_food=_float(center.get("food", 0.0)),
            carrion_distance=_target_distance(carrion_target),
            carrion_strength=_float(carrion_target.get("strength", 0.0)),
            water_distance=_target_distance(water_target),
            water_strength=_float(water_target.get("strength", 0.0)),
            carrion_target=carrion_target,
            water_target=water_target,
        )


def build_carrion_counterfactual_report(
    *,
    seeds: Sequence[int] = DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    ticks: int = DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    scripts: Sequence[str] = DEFAULT_COUNTERFACTUAL_SCRIPTS,
    fixture_name: str = "carrion_only",
    trajectory_output_dir: str | Path | None = None,
    source_autopsy_path: str | Path | None = None,
) -> dict[str, object]:
    if fixture_name != "carrion_only":
        raise CarrionCounterfactualError(
            "carrion counterfactual rollouts currently support carrion_only only"
        )
    seed_values = _validated_seeds(seeds)
    tick_count = _positive_int(ticks, field="ticks")
    script_values = _validated_scripts(scripts)
    output_dir = Path(trajectory_output_dir) if trajectory_output_dir else None
    contract = _counterfactual_contract(
        seeds=seed_values,
        ticks=tick_count,
        scripts=script_values,
        fixture_name=fixture_name,
        source_autopsy_path=source_autopsy_path,
    )
    script_reports = [
        _run_counterfactual_script(
            script_name=script_name,
            seeds=seed_values,
            ticks=tick_count,
            fixture_name=fixture_name,
            trajectory_output_dir=output_dir,
        )
        for script_name in script_values
    ]
    aggregate = _aggregate_script_reports(script_reports)
    return {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
        "counterfactual_policy": MIND_V3_CARRION_COUNTERFACTUAL_POLICY_VERSION,
        "counterfactual_contract": contract,
        "provenance": {
            "counterfactual_contract_digest": stable_payload_digest(contract),
        },
        "scope": {
            "fixture_name": fixture_name,
            "state_restore_available": False,
            "branch_policy": (
                "same_fixture_seed_scripted_rollout_v1; trajectory JSONL does "
                "not contain full world snapshots for exact mid-run branching"
            ),
            "policy_input_policy": (
                "scripts use observation self, local_patch, navigation, and "
                "action_mask only"
            ),
        },
        "aggregate": aggregate,
        "scripts": script_reports,
    }


def write_carrion_counterfactual_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _run_counterfactual_script(
    *,
    script_name: str,
    seeds: Sequence[int],
    ticks: int,
    fixture_name: str,
    trajectory_output_dir: Path | None,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for seed in seeds:
        trajectory_path = None
        if trajectory_output_dir is not None:
            trajectory_path = (
                trajectory_output_dir
                / f"counterfactual-{script_name}-seed-{int(seed)}.jsonl.gz"
            )
        world = _fixture_world(
            fixture_name=fixture_name,
            seed=int(seed),
            ticks=ticks,
            policy=CarrionCounterfactualPolicy(script_name),
        )
        run = _run_world(
            world=world,
            seed=int(seed),
            ticks=ticks,
            trajectory_output_path=trajectory_path,
            trajectory_split_id=f"mind_v3_carrion_counterfactual_{script_name}",
        )
        run["fixture"] = fixture_name
        run["counterfactual_script"] = script_name
        run["evaluation_context"] = "carrion_counterfactual_scripted_rollout"
        if trajectory_path is not None:
            run["trajectory_path"] = str(trajectory_path)
        runs.append(run)
    aggregate = _aggregate_runs(runs)
    return {
        "script_name": script_name,
        "policy_id": f"mind_v3_counterfactual_{script_name}",
        "policy_version": MIND_V3_CARRION_COUNTERFACTUAL_POLICY_VERSION,
        "aggregate": aggregate,
        "acceptance": _script_acceptance(runs, aggregate),
        "runs": runs,
    }


def _script_acceptance(
    runs: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
) -> dict[str, object]:
    alive_values = [int(run["alive_agents"]) for run in runs]
    successful_run_count = sum(1 for alive in alive_values if alive > 0)
    heuristic_action_source_count = int(
        aggregate.get("heuristic_action_source_count", 0)
    )
    zero_heuristic_runtime_actions = heuristic_action_source_count == 0
    return {
        "terminal_alive_nonzero": successful_run_count > 0,
        "successful_run_count": successful_run_count,
        "run_count": len(runs),
        "max_alive_agents": max(alive_values) if alive_values else 0,
        "heuristic_action_source_count": heuristic_action_source_count,
        "zero_heuristic_runtime_actions": zero_heuristic_runtime_actions,
        "dominant_action_share_le_0_50": float(
            aggregate.get("dominant_requested_action_share", 1.0)
        )
        <= 0.5,
        "diagnostic_acceptance_passed": successful_run_count > 0
        and zero_heuristic_runtime_actions,
    }


def _aggregate_script_reports(
    script_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    successful_scripts = [
        str(report["script_name"])
        for report in script_reports
        if bool(dict(report["acceptance"])["diagnostic_acceptance_passed"])
    ]
    all_runs = [
        run
        for report in script_reports
        for run in list(report.get("runs", []))
        if isinstance(run, dict)
    ]
    aggregate = _aggregate_runs(all_runs) if all_runs else {}
    best = _best_script(script_reports)
    return {
        "script_count": len(script_reports),
        "run_count": len(all_runs),
        "survivable_sequence_found": bool(successful_scripts),
        "successful_scripts": successful_scripts,
        "best_script_by_terminal_alive_then_births": best,
        "combined": aggregate,
    }


def _best_script(
    script_reports: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not script_reports:
        return None

    def key(report: Mapping[str, object]) -> tuple[float, float, float, str]:
        aggregate = dict(report["aggregate"])
        return (
            float(aggregate.get("alive_agents_mean", 0.0)),
            float(aggregate.get("births_mean", 0.0)),
            -float(aggregate.get("dominant_requested_action_share", 1.0)),
            str(report["script_name"]),
        )

    selected = max(script_reports, key=key)
    aggregate = dict(selected["aggregate"])
    return {
        "script_name": str(selected["script_name"]),
        "alive_agents_mean": aggregate.get("alive_agents_mean", 0.0),
        "births_mean": aggregate.get("births_mean", 0.0),
        "dominant_requested_action": aggregate.get("dominant_requested_action"),
        "dominant_requested_action_share": aggregate.get(
            "dominant_requested_action_share",
            0.0,
        ),
    }


def _counterfactual_contract(
    *,
    seeds: Sequence[int],
    ticks: int,
    scripts: Sequence[str],
    fixture_name: str,
    source_autopsy_path: str | Path | None,
) -> dict[str, object]:
    return {
        "schema_version": MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
        "policy": MIND_V3_CARRION_COUNTERFACTUAL_POLICY_VERSION,
        "fixture_name": fixture_name,
        "seeds": [int(seed) for seed in seeds],
        "ticks": int(ticks),
        "scripts": list(scripts),
        "source_autopsy_path": (
            str(source_autopsy_path) if source_autopsy_path is not None else None
        ),
        "acceptance": (
            "diagnostic acceptance only: at least one scripted legal rollout "
            "has terminal alive_agents > 0 with zero heuristic action sources"
        ),
    }


def _carrion_then_water(
    facts: _PolicyVisibleFacts,
    action_mask: dict[str, bool],
) -> tuple[str, str]:
    if _valid(action_mask, "drink") and facts.hydration < 0.96:
        return "drink", "drink_when_water_accessible"
    if facts.center_animal_resource > 0.02 and facts.energy < 0.98:
        return "eat", "eat_local_animal_resource"
    if facts.hydration < 0.68:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "move_to_water_after_carrion_or_low_hydration"
    if facts.energy < 0.86 and facts.hydration > 0.42:
        carrion_move = _move_to_carrion(facts, action_mask)
        if carrion_move is not None:
            return carrion_move, "move_to_visible_carrion"
    if facts.hydration < 0.9:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "top_up_hydration"
    if _valid(action_mask, "eat") and facts.energy < 0.24:
        return "eat", "critical_fallback_eat"
    return "stay", "conserve"


def _water_first_recovery(
    facts: _PolicyVisibleFacts,
    action_mask: dict[str, bool],
) -> tuple[str, str]:
    if _valid(action_mask, "drink") and facts.hydration < 0.98:
        return "drink", "water_first_drink"
    if facts.hydration < 0.88 and facts.energy > 0.22:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "water_first_move_to_water"
    if facts.center_animal_resource > 0.02 and facts.energy < 0.96:
        return "eat", "water_first_eat_local_animal_resource"
    if facts.energy < 0.78 and facts.hydration > 0.6:
        carrion_move = _move_to_carrion(facts, action_mask)
        if carrion_move is not None:
            return carrion_move, "water_first_move_to_carrion"
    if facts.hydration < 0.96:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "water_first_hydration_top_up"
    return "stay", "water_first_conserve"


def _conserve_after_carrion(
    facts: _PolicyVisibleFacts,
    action_mask: dict[str, bool],
) -> tuple[str, str]:
    if facts.center_animal_resource > 0.02 and facts.energy < 0.99:
        return "eat", "conserve_eat_local_animal_resource"
    if _valid(action_mask, "drink") and facts.hydration < 0.96:
        return "drink", "conserve_drink_when_accessible"
    if facts.hydration < 0.72 and facts.energy > 0.32:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "conserve_move_to_water_when_needed"
    if facts.energy < 0.34 and facts.hydration > 0.48:
        carrion_move = _move_to_carrion(facts, action_mask)
        if carrion_move is not None:
            return carrion_move, "conserve_move_to_carrion_when_critical"
    if _valid(action_mask, "eat") and facts.energy < 0.18:
        return "eat", "conserve_critical_fallback_eat"
    return "stay", "conserve_stay"


def _hydration_safe_carrion_cycle(
    facts: _PolicyVisibleFacts,
    action_mask: dict[str, bool],
) -> tuple[str, str]:
    if _valid(action_mask, "drink") and facts.hydration < 0.94:
        return "drink", "cycle_drink_before_foraging"
    if facts.hydration < 0.74 and facts.energy > 0.28:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "cycle_move_to_water"
    if facts.center_animal_resource > 0.02 and facts.energy < 0.94:
        return "eat", "cycle_eat_local_animal_resource"
    if facts.energy < 0.72 and facts.hydration > 0.76:
        carrion_move = _move_to_carrion(facts, action_mask)
        if carrion_move is not None:
            return carrion_move, "cycle_move_to_carrion_with_hydration_margin"
    if facts.hydration < 0.9:
        water_move = _move_to_water(facts, action_mask)
        if water_move is not None:
            return water_move, "cycle_hydration_margin_top_up"
    if _valid(action_mask, "eat") and facts.energy < 0.22:
        return "eat", "cycle_critical_fallback_eat"
    return "stay", "cycle_conserve"


def _move_to_water(
    facts: _PolicyVisibleFacts,
    action_mask: dict[str, bool],
) -> str | None:
    if facts.water_strength <= 0.0 or facts.water_distance <= 0:
        return None
    return _navigation_action(
        facts.water_target,
        action_mask,
        min_strength=0.0,
        allow_detour=True,
    )


def _move_to_carrion(
    facts: _PolicyVisibleFacts,
    action_mask: dict[str, bool],
) -> str | None:
    if facts.carrion_strength <= 0.0 or facts.carrion_distance <= 0:
        return None
    return _navigation_action(
        facts.carrion_target,
        action_mask,
        min_strength=0.0,
        allow_detour=True,
    )


def _animal_resource(cell: Mapping[str, object]) -> float:
    return _float(cell.get("fresh_kill_energy", 0.0)) + _float(
        cell.get("carcass_energy", 0.0)
    )


def _fallback_action(action_mask: dict[str, bool]) -> str:
    if _valid(action_mask, "stay"):
        return "stay"
    for action in ACTION_NAMES:
        if _valid(action_mask, action):
            return action
    return "stay"


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values:
        raise CarrionCounterfactualError("at least one seed is required")
    return values


def _validated_scripts(scripts: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(str(script) for script in scripts if str(script)))
    if not values:
        raise CarrionCounterfactualError("at least one script is required")
    unsupported = sorted(
        script for script in values if script not in DEFAULT_COUNTERFACTUAL_SCRIPTS
    )
    if unsupported:
        raise CarrionCounterfactualError(
            "unsupported counterfactual script(s): " + ", ".join(unsupported)
        )
    return values


def _positive_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise CarrionCounterfactualError(f"{field} must be positive")
    return parsed


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
