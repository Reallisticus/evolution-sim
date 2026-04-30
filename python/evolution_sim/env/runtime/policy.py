from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evolution_sim.env.runtime.action_space import ACTION_NAMES
from evolution_sim.env.runtime.observations import LOCAL_PATCH_RADIUS

POLICY_INTERFACE_VERSION = "mind_policy_interface_v1"
OBSERVATION_HEURISTIC_POLICY_ID = "observation_heuristic"
OBSERVATION_HEURISTIC_POLICY_VERSION = "observation_heuristic_v10"

MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_north": (0, -1),
    "move_south": (0, 1),
    "move_east": (1, 0),
    "move_west": (-1, 0),
}
SCAVENGER_ADJACENT_CARCASS_SURVIVAL_ENERGY_THRESHOLD = 0.36
SCAVENGER_ADJACENT_CARCASS_SURVIVAL_HYDRATION_THRESHOLD = 0.64
SCAVENGER_MATCHED_DIET_CARRION_THRESHOLD = 0.2
SCAVENGER_MATCHED_DIET_CARRION_MIN_ENERGY = 0.18
SCAVENGER_MATCHED_DIET_CARRION_MAX_ENERGY = 0.28
SCAVENGER_MATCHED_DIET_CARRION_MAX_HEALTH = 0.3
HUNTER_MATCHED_DIET_PURSUIT_THRESHOLD = 0.62
HUNTER_MATCHED_DIET_PURSUIT_MIN_ENERGY = 0.72
HUNTER_MATCHED_DIET_PURSUIT_MIN_HYDRATION = 0.68
HUNTER_MATCHED_DIET_PURSUIT_MIN_HEALTH = 0.72
HUNTER_MATCHED_DIET_PURSUIT_MAX_PREY_DISTANCE = 3
ANIMAL_CRITICAL_LOCAL_FOOD_BEFORE_WATER_ENERGY = 0.36
ANIMAL_BLOCKED_WATER_LOCAL_FOOD_ENERGY = 0.16
ANIMAL_BLOCKED_WATER_SURVIVAL_HYDRATION = 0.06
ATTACK_DELTAS: dict[str, tuple[int, int]] = {
    action.replace("move_", "attack_"): delta for action, delta in MOVE_DELTAS.items()
}


@dataclass(frozen=True, slots=True)
class ActionDecision:
    requested_action: str
    source: str
    policy_id: str
    policy_version: str


class Policy(Protocol):
    policy_id: str
    policy_version: str

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        ...


@dataclass(frozen=True, slots=True)
class ObservationHeuristicPolicyConfig:
    policy_id: str = OBSERVATION_HEURISTIC_POLICY_ID
    policy_version: str = OBSERVATION_HEURISTIC_POLICY_VERSION
    drink_threshold: float = 0.82
    eat_threshold: float = 0.9
    desperate_energy_threshold: float = 0.58
    urgent_hydration_threshold: float = 0.56
    critical_hydration_threshold: float = 0.28
    critical_local_food_energy_threshold: float = 0.52
    critical_local_food_min: float = 0.12
    animal_pursuit_min_hydration: float = 0.64
    desperate_prey_max_distance: int = 2
    carrion_navigation_min_strength: float = 0.025
    scavenger_carrion_seek_energy_threshold: float = 0.94
    scavenger_carrion_seek_max_distance: int = 8
    hunter_carrion_seek_energy_threshold: float = 0.92
    hunter_carrion_seek_max_distance: int = 8
    mixed_carrion_seek_energy_threshold: float = 0.94
    mixed_carrion_seek_max_distance: int = 10
    animal_survival_forage_max_distance: int = 2
    local_patch_radius: int = LOCAL_PATCH_RADIUS

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "drink_threshold": self.drink_threshold,
            "eat_threshold": self.eat_threshold,
            "desperate_energy_threshold": self.desperate_energy_threshold,
            "urgent_hydration_threshold": self.urgent_hydration_threshold,
            "critical_hydration_threshold": self.critical_hydration_threshold,
            "critical_local_food_energy_threshold": (
                self.critical_local_food_energy_threshold
            ),
            "critical_local_food_min": self.critical_local_food_min,
            "animal_pursuit_min_hydration": self.animal_pursuit_min_hydration,
            "desperate_prey_max_distance": self.desperate_prey_max_distance,
            "carrion_navigation_min_strength": self.carrion_navigation_min_strength,
            "scavenger_carrion_seek_energy_threshold": (
                self.scavenger_carrion_seek_energy_threshold
            ),
            "scavenger_carrion_seek_max_distance": (
                self.scavenger_carrion_seek_max_distance
            ),
            "hunter_carrion_seek_energy_threshold": (
                self.hunter_carrion_seek_energy_threshold
            ),
            "hunter_carrion_seek_max_distance": self.hunter_carrion_seek_max_distance,
            "mixed_carrion_seek_energy_threshold": (
                self.mixed_carrion_seek_energy_threshold
            ),
            "mixed_carrion_seek_max_distance": self.mixed_carrion_seek_max_distance,
            "animal_survival_forage_max_distance": (
                self.animal_survival_forage_max_distance
            ),
            "local_patch_radius": self.local_patch_radius,
        }


@dataclass(slots=True)
class ObservationHeuristicPolicy:
    config: ObservationHeuristicPolicyConfig = ObservationHeuristicPolicyConfig()

    @property
    def policy_id(self) -> str:
        return self.config.policy_id

    @property
    def policy_version(self) -> str:
        return self.config.policy_version

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        self_state = _self_state(observation)
        patch_cells = _patch_cells(observation)
        navigation = _navigation(observation)
        center = _cell_at(patch_cells, 0, 0)
        energy_ratio = _float(self_state["energy_ratio"])
        hydration_ratio = _float(self_state["hydration_ratio"])
        trophic_role = str(self_state["trophic_role"])
        meat_mode = str(self_state["meat_mode"])
        is_omnivore = trophic_role == "omnivore"
        center_food = _float(center["food"]) if center is not None else 0.0
        center_hazard_level = _float(center["hazard_level"]) if center is not None else 0.0
        health_ratio = _float(self_state.get("health_ratio", 1.0))
        matched_diet_ratio = _float(self_state.get("matched_diet_ratio", 1.0))
        omnivore_needs_animal_channel = (
            energy_ratio < self.config.desperate_energy_threshold
            or center_food < 0.08
            or (
                meat_mode == "mixed"
                and center_food < 0.32
                and energy_ratio < self.config.eat_threshold
            )
        )
        prefer_meat = trophic_role == "carnivore" or (
            meat_mode in {"hunter", "scavenger", "mixed"} and omnivore_needs_animal_channel
        )
        prefer_hunting = (
            trophic_role == "carnivore"
            and meat_mode != "scavenger"
        ) or (
            meat_mode == "hunter"
            and omnivore_needs_animal_channel
            and not (is_omnivore and center_food > 0.1)
        )
        prefer_scavenging = trophic_role == "carnivore" or (
            meat_mode in {"scavenger", "mixed"} and omnivore_needs_animal_channel
        )

        if _valid(action_mask, "eat") and center is not None:
            if self._should_eat_current(
                center,
                energy_ratio=energy_ratio,
                hydration_ratio=hydration_ratio,
                meat_mode=meat_mode,
                prefer_meat=prefer_meat,
                prefer_scavenging=prefer_scavenging,
                animal_capable=meat_mode in {"hunter", "scavenger", "mixed"},
            ):
                return self._decision("eat", "heuristic_observation")

        if self._should_pursue_hunter_matched_diet(
            meat_mode=meat_mode,
            energy_ratio=energy_ratio,
            hydration_ratio=hydration_ratio,
            health_ratio=health_ratio,
            matched_diet_ratio=matched_diet_ratio,
        ):
            attack = self._best_attack_action(
                patch_cells,
                action_mask,
                energy_ratio=energy_ratio,
            )
            if attack is not None:
                return self._decision(attack, "heuristic_observation")
            prey_distance = _target_distance(navigation["prey"])
            if (
                0
                < prey_distance
                <= HUNTER_MATCHED_DIET_PURSUIT_MAX_PREY_DISTANCE
            ):
                prey_move = _navigation_action(navigation["prey"], action_mask)
                if prey_move is not None:
                    return self._decision(prey_move, "heuristic_observation")

        if (
            meat_mode == "scavenger"
            and _valid(action_mask, "eat")
            and self._should_eat_adjacent_carcass(
                patch_cells,
                action_mask,
                energy_ratio=energy_ratio,
                hydration_ratio=hydration_ratio,
            )
        ):
            return self._decision("eat", "heuristic_observation")

        if (
            _valid(action_mask, "drink")
            and hydration_ratio < self.config.drink_threshold
            and (
                hydration_ratio < 0.66
                or hydration_ratio <= energy_ratio + 0.08
            )
        ):
            return self._decision("drink", "heuristic_observation")

        water_distance = _target_distance(navigation["water"])
        water_step = _navigation_action(navigation["water"], action_mask)
        if (
            meat_mode in {"hunter", "mixed"}
            and _valid(action_mask, "eat")
            and center is not None
            and center_food >= self.config.critical_local_food_min
            and energy_ratio < ANIMAL_CRITICAL_LOCAL_FOOD_BEFORE_WATER_ENERGY
            and hydration_ratio < self.config.critical_hydration_threshold
            and (
                water_distance > 1
                or (
                    water_distance > 0
                    and water_step is None
                    and energy_ratio < ANIMAL_BLOCKED_WATER_LOCAL_FOOD_ENERGY
                )
            )
        ):
            return self._decision("eat", "heuristic_observation")

        if (
            meat_mode in {"hunter", "scavenger", "mixed"}
            and hydration_ratio < self.config.critical_hydration_threshold
        ):
            water_move = water_step
            if water_move is None and meat_mode in {"hunter", "mixed"}:
                water_move = self._best_water_access_move(
                    patch_cells,
                    action_mask,
                    max_distance=self.config.local_patch_radius * 2,
                )
            if (
                water_move is None
                and meat_mode in {"hunter", "mixed"}
                and water_distance > 1
            ):
                water_move = _navigation_action(
                    navigation["water"],
                    action_mask,
                    allow_detour=True,
                )
            if water_move is not None:
                return self._decision(water_move, "heuristic_observation")
            if (
                meat_mode in {"hunter", "mixed"}
                and _target_distance(navigation["water"]) > 0
                and _valid(action_mask, "stay")
            ):
                return self._decision("stay", "heuristic_observation_conserve")

        if (
            meat_mode == "scavenger"
            and energy_ratio < self.config.scavenger_carrion_seek_energy_threshold
        ):
            carrion_move = self._best_carrion_resource_move(
                patch_cells,
                action_mask,
                max_distance=self.config.animal_survival_forage_max_distance,
            )
            if carrion_move is not None:
                return self._decision(carrion_move, "heuristic_observation")
            if (
                matched_diet_ratio < SCAVENGER_MATCHED_DIET_CARRION_THRESHOLD
                and energy_ratio >= SCAVENGER_MATCHED_DIET_CARRION_MIN_ENERGY
                and energy_ratio <= SCAVENGER_MATCHED_DIET_CARRION_MAX_ENERGY
                and hydration_ratio >= self.config.animal_pursuit_min_hydration
                and health_ratio <= SCAVENGER_MATCHED_DIET_CARRION_MAX_HEALTH
                and self._should_follow_carrion_navigation(
                    navigation["carrion"],
                    meat_mode=meat_mode,
                    energy_ratio=energy_ratio,
                    hydration_ratio=hydration_ratio,
                )
            ):
                carrion_move = _navigation_action(
                    navigation["carrion"],
                    action_mask,
                    min_strength=self.config.carrion_navigation_min_strength,
                    allow_detour=True,
                )
                if carrion_move is not None:
                    return self._decision(carrion_move, "heuristic_observation")
            if (
                _valid(action_mask, "eat")
                and center is not None
                and center_food >= self.config.critical_local_food_min
                and energy_ratio < self.config.critical_local_food_energy_threshold
                and hydration_ratio >= 0.42
            ):
                return self._decision("eat", "heuristic_observation")
            if hydration_ratio < self.config.animal_pursuit_min_hydration:
                water_move = _navigation_action(navigation["water"], action_mask)
                if water_move is not None:
                    return self._decision(water_move, "heuristic_observation")
            if self._should_follow_carrion_navigation(
                navigation["carrion"],
                meat_mode=meat_mode,
                energy_ratio=energy_ratio,
                hydration_ratio=hydration_ratio,
            ):
                carrion_move = _navigation_action(
                    navigation["carrion"],
                    action_mask,
                    min_strength=0.0,
                    allow_detour=True,
                )
                if carrion_move is not None:
                    return self._decision(carrion_move, "heuristic_observation")

        if (
            meat_mode in {"hunter", "mixed"}
            and hydration_ratio < self.config.animal_pursuit_min_hydration
        ):
            water_move = _navigation_action(navigation["water"], action_mask)
            if water_move is None:
                water_move = self._best_water_access_move(
                    patch_cells,
                    action_mask,
                    max_distance=self.config.local_patch_radius * 2,
                )
            if water_move is not None:
                return self._decision(water_move, "heuristic_observation")

        if (
            meat_mode in {"hunter", "mixed"}
            and _valid(action_mask, "eat")
            and center is not None
            and center_food >= self.config.critical_local_food_min
            and energy_ratio < self.config.critical_local_food_energy_threshold
            and hydration_ratio >= 0.42
        ):
            return self._decision("eat", "heuristic_observation")

        if (
            meat_mode in {"hunter", "scavenger", "mixed"}
            and hydration_ratio < self.config.animal_pursuit_min_hydration
        ):
            water_move = _navigation_action(navigation["water"], action_mask)
            if water_move is not None:
                return self._decision(water_move, "heuristic_observation")

        if (
            meat_mode == "hunter"
            and hydration_ratio >= self.config.animal_pursuit_min_hydration
            and energy_ratio < 0.92
        ):
            attack = self._best_attack_action(
                patch_cells,
                action_mask,
                energy_ratio=energy_ratio,
            )
            if attack is not None:
                return self._decision(attack, "heuristic_observation")

        if (
            meat_mode in {"hunter", "mixed"}
            and energy_ratio
            < (
                self.config.hunter_carrion_seek_energy_threshold
                if meat_mode == "hunter"
                else self.config.mixed_carrion_seek_energy_threshold
            )
            and self._should_follow_carrion_navigation(
                navigation["carrion"],
                meat_mode=meat_mode,
                energy_ratio=energy_ratio,
                hydration_ratio=hydration_ratio,
            )
        ):
            carrion_move = _navigation_action(
                navigation["carrion"],
                action_mask,
                min_strength=self.config.carrion_navigation_min_strength,
                allow_detour=True,
            )
            if carrion_move is not None:
                return self._decision(carrion_move, "heuristic_observation")

        if (
            meat_mode == "mixed"
            and energy_ratio < self.config.mixed_carrion_seek_energy_threshold
            and hydration_ratio >= self.config.animal_pursuit_min_hydration
        ):
            carrion_move = self._best_carrion_resource_move(
                patch_cells,
                action_mask,
                max_distance=self.config.animal_survival_forage_max_distance,
            )
            if carrion_move is not None:
                return self._decision(carrion_move, "heuristic_observation")

        if (
            is_omnivore
            and _valid(action_mask, "eat")
            and center is not None
            and center_food > (0.04 if energy_ratio < 0.5 else 0.08)
            and energy_ratio < self.config.eat_threshold
        ):
            return self._decision("eat", "heuristic_observation")

        if prefer_hunting:
            attack = self._best_attack_action(
                patch_cells,
                action_mask,
                energy_ratio=energy_ratio,
            )
            if attack is not None:
                return self._decision(attack, "heuristic_observation")

        if prefer_meat and hydration_ratio < self.config.urgent_hydration_threshold:
            water_move = _navigation_action(navigation["water"], action_mask)
            if water_move is not None:
                return self._decision(water_move, "heuristic_observation")

        if is_omnivore and energy_ratio < 0.74:
            plant_move = _navigation_action(navigation["plant"], action_mask)
            if plant_move is not None:
                return self._decision(plant_move, "heuristic_observation")

        desperate_meat_search = (
            prefer_meat and energy_ratio < self.config.desperate_energy_threshold
        )
        if desperate_meat_search:
            if prefer_scavenging:
                if self._should_follow_carrion_navigation(
                    navigation["carrion"],
                    meat_mode=meat_mode,
                    energy_ratio=energy_ratio,
                    hydration_ratio=hydration_ratio,
                ):
                    carrion_move = _navigation_action(
                        navigation["carrion"],
                        action_mask,
                        min_strength=self.config.carrion_navigation_min_strength,
                        allow_detour=True,
                    )
                    if carrion_move is not None:
                        return self._decision(carrion_move, "heuristic_observation")
            if (
                _valid(action_mask, "eat")
                and center is not None
                and center_food > 0.08
                and hydration_ratio >= 0.5
            ):
                return self._decision("eat", "heuristic_observation")

        if prefer_meat:
            prey_navigation_distance = _target_distance(navigation["prey"])
            prey_close_enough = (
                prey_navigation_distance > 0
                and prey_navigation_distance <= self.config.desperate_prey_max_distance
            )
            can_spend_energy_chasing = (
                (not desperate_meat_search or prey_close_enough)
                and hydration_ratio >= self.config.animal_pursuit_min_hydration
            )
            if can_spend_energy_chasing:
                biotic_move = self._best_move_toward(
                    patch_cells,
                    action_mask,
                    scoring="biotic",
                    energy_ratio=energy_ratio,
                    hydration_ratio=hydration_ratio,
                    prefer_hunting=prefer_hunting,
                    prefer_scavenging=prefer_scavenging,
                )
                if biotic_move is not None:
                    return self._decision(biotic_move, "heuristic_observation")
            if prefer_scavenging:
                if self._should_follow_carrion_navigation(
                    navigation["carrion"],
                    meat_mode=meat_mode,
                    energy_ratio=energy_ratio,
                    hydration_ratio=hydration_ratio,
                ):
                    carrion_move = _navigation_action(
                        navigation["carrion"],
                        action_mask,
                        min_strength=(
                            self.config.carrion_navigation_min_strength
                            if desperate_meat_search
                            else 0.0
                        ),
                        allow_detour=True,
                    )
                    if carrion_move is not None:
                        return self._decision(carrion_move, "heuristic_observation")
            if prefer_hunting and can_spend_energy_chasing:
                prey_move = _navigation_action(navigation["prey"], action_mask)
                if prey_move is not None:
                    return self._decision(prey_move, "heuristic_observation")

        if (
            meat_mode in {"hunter", "scavenger", "mixed"}
            and energy_ratio < self.config.desperate_energy_threshold
            and 0
            < _target_distance(navigation["plant"])
            <= self.config.animal_survival_forage_max_distance
        ):
            plant_move = _navigation_action(navigation["plant"], action_mask)
            if plant_move is not None:
                return self._decision(plant_move, "heuristic_observation")

        if (
            prefer_meat
            and _valid(action_mask, "stay")
            and (energy_ratio < self.config.desperate_energy_threshold or energy_ratio > 0.62)
            and hydration_ratio > 0.62
            and center_hazard_level <= 0.0
        ):
            return self._decision("stay", "heuristic_observation_conserve")

        if (
            _valid(action_mask, "eat")
            and center is not None
            and center_food > (0.04 if energy_ratio < 0.5 else 0.08)
            and (hydration_ratio >= 0.58 or center_food > 0.18)
            and energy_ratio < self.config.eat_threshold
        ):
            return self._decision("eat", "heuristic_observation")

        if hydration_ratio < self.config.drink_threshold:
            water_move = _navigation_action(navigation["water"], action_mask)
            if water_move is not None:
                return self._decision(water_move, "heuristic_observation")

        if energy_ratio < self.config.eat_threshold:
            plant_move = _navigation_action(navigation["plant"], action_mask)
            if plant_move is not None:
                return self._decision(plant_move, "heuristic_observation")

        general_move = self._best_move_toward(
            patch_cells,
            action_mask,
            scoring="general",
            energy_ratio=energy_ratio,
            hydration_ratio=hydration_ratio,
            prefer_hunting=prefer_hunting,
            prefer_scavenging=prefer_scavenging,
        )
        if general_move is not None:
            return self._decision(general_move, "heuristic_observation")

        if not prefer_hunting:
            attack = self._best_attack_action(
                patch_cells,
                action_mask,
                energy_ratio=energy_ratio,
            )
            if attack is not None:
                return self._decision(attack, "heuristic_observation")

        return self._fallback(action_mask)

    def _should_eat_current(
        self,
        center: dict[str, object],
        *,
        energy_ratio: float,
        hydration_ratio: float,
        meat_mode: str,
        prefer_meat: bool,
        prefer_scavenging: bool,
        animal_capable: bool,
    ) -> bool:
        food = _float(center["food"])
        fresh_kill = _float(center["fresh_kill_energy"])
        carcass = _float(center["carcass_energy"])
        if fresh_kill > 0.0 and (prefer_meat or animal_capable or energy_ratio < 0.72):
            return energy_ratio < 0.96
        if carcass > 0.0 and (
            prefer_scavenging
            or prefer_meat
            or animal_capable
            or energy_ratio < 0.74
        ):
            if (
                meat_mode == "scavenger"
                and hydration_ratio < self.config.drink_threshold
            ):
                return True
            return energy_ratio < 0.94
        return False

    def _should_eat_adjacent_carcass(
        self,
        patch_cells: list[dict[str, object]],
        action_mask: dict[str, bool],
        *,
        energy_ratio: float,
        hydration_ratio: float,
    ) -> bool:
        if (
            energy_ratio >= SCAVENGER_ADJACENT_CARCASS_SURVIVAL_ENERGY_THRESHOLD
            and hydration_ratio >= SCAVENGER_ADJACENT_CARCASS_SURVIVAL_HYDRATION_THRESHOLD
        ):
            return False
        for cell in patch_cells:
            distance = abs(int(cell["dx"])) + abs(int(cell["dy"]))
            if distance != 1:
                continue
            if not bool(cell["in_bounds"]) or cell["terrain"] == "water":
                continue
            if _float(cell["carcass_energy"]) <= 0.0:
                continue
            blocked_by_occupant = str(cell["occupant"]) not in {"none", "self"}
            movement_action = _move_action_for_delta(int(cell["dx"]), int(cell["dy"]))
            blocked_by_mask = movement_action is not None and not _valid(
                action_mask,
                movement_action,
            )
            if blocked_by_occupant or blocked_by_mask:
                return True
        return False

    @staticmethod
    def _should_pursue_hunter_matched_diet(
        *,
        meat_mode: str,
        energy_ratio: float,
        hydration_ratio: float,
        health_ratio: float,
        matched_diet_ratio: float,
    ) -> bool:
        return (
            meat_mode == "hunter"
            and matched_diet_ratio < HUNTER_MATCHED_DIET_PURSUIT_THRESHOLD
            and energy_ratio >= HUNTER_MATCHED_DIET_PURSUIT_MIN_ENERGY
            and hydration_ratio >= HUNTER_MATCHED_DIET_PURSUIT_MIN_HYDRATION
            and health_ratio >= HUNTER_MATCHED_DIET_PURSUIT_MIN_HEALTH
        )

    def _best_attack_action(
        self,
        patch_cells: list[dict[str, object]],
        action_mask: dict[str, bool],
        *,
        energy_ratio: float,
    ) -> str | None:
        best_action: str | None = None
        best_score = float("-inf")
        for action, (dx, dy) in ATTACK_DELTAS.items():
            if not _valid(action_mask, action):
                continue
            cell = _cell_at(patch_cells, dx, dy)
            if cell is None or cell.get("occupant") != "agent":
                continue
            score = 0.2
            score += _float(cell["prey_biomass"]) * 1.1
            score -= _float(cell["predator_risk"]) * 0.25
            score += max(0.0, 0.72 - energy_ratio) * 0.3
            score -= _float(cell["hazard_level"]) * 0.08
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _best_move_toward(
        self,
        patch_cells: list[dict[str, object]],
        action_mask: dict[str, bool],
        *,
        scoring: str,
        energy_ratio: float,
        hydration_ratio: float,
        prefer_hunting: bool,
        prefer_scavenging: bool,
    ) -> str | None:
        best_action: str | None = None
        best_score = float("-inf")
        for cell in patch_cells:
            dx = int(cell["dx"])
            dy = int(cell["dy"])
            if dx == 0 and dy == 0:
                continue
            if not bool(cell["in_bounds"]) or cell["terrain"] == "water":
                continue
            if cell["occupant"] == "agent" and abs(dx) + abs(dy) <= 1:
                continue
            action = _step_action_for_delta(dx, dy, action_mask)
            if action is None:
                continue
            score = self._cell_score(
                cell,
                scoring=scoring,
                energy_ratio=energy_ratio,
                hydration_ratio=hydration_ratio,
                prefer_hunting=prefer_hunting,
                prefer_scavenging=prefer_scavenging,
            )
            distance = abs(dx) + abs(dy)
            score -= distance * 0.045
            if score > best_score:
                best_score = score
                best_action = action
        if best_score <= 0.02:
            return None
        return best_action

    def _cell_score(
        self,
        cell: dict[str, object],
        *,
        scoring: str,
        energy_ratio: float,
        hydration_ratio: float,
        prefer_hunting: bool,
        prefer_scavenging: bool,
    ) -> float:
        water_urgency = max(0.0, 1.0 - hydration_ratio)
        food_urgency = max(0.0, 1.0 - energy_ratio)
        score = 0.0
        if cell["water_access_reason"] != "none":
            score += 1.65 * water_urgency
        score += _float(cell["vegetation"]) * 0.08
        score -= _float(cell["recovery_debt"]) * 0.08
        score -= _float(cell["hazard_level"]) * (0.1 + food_urgency * 0.16)

        if scoring == "biotic":
            if prefer_hunting:
                score += _float(cell["prey_biomass"]) * (0.45 + food_urgency)
            if prefer_scavenging:
                score += _float(cell["carrion_signal"]) * (1.0 + food_urgency * 1.3)
                score += _float(cell["carcass_energy"]) * (0.85 + food_urgency * 1.2)
            score += _float(cell["fresh_kill_energy"]) * (0.65 + food_urgency * 1.1)
            score += _float(cell["carcass_energy"]) * (0.5 + food_urgency)
            score -= _float(cell["predator_risk"]) * 0.12
            return score

        score += _float(cell["food"]) * (0.25 + food_urgency)
        score += _float(cell["fresh_kill_energy"]) * (0.12 + food_urgency * 0.3)
        score += _float(cell["carcass_energy"]) * (0.1 + food_urgency * 0.28)
        score += _float(cell["prey_biomass"]) * 0.12
        score += _float(cell["carrion_signal"]) * 0.1
        score -= _float(cell["predator_risk"]) * 0.08
        return score

    def _fallback(self, action_mask: dict[str, bool]) -> ActionDecision:
        for action in ("stay", "eat", "drink", *MOVE_DELTAS):
            if _valid(action_mask, action):
                return self._decision(action, "heuristic_observation_fallback")
        return self._decision("stay", "heuristic_observation_fallback")

    def _carrion_seek_max_distance(self, meat_mode: str) -> int:
        if meat_mode == "hunter":
            return self.config.hunter_carrion_seek_max_distance
        if meat_mode == "mixed":
            return self.config.mixed_carrion_seek_max_distance
        return self.config.scavenger_carrion_seek_max_distance

    def _should_follow_carrion_navigation(
        self,
        target: dict[str, object],
        *,
        meat_mode: str,
        energy_ratio: float,
        hydration_ratio: float,
    ) -> bool:
        if hydration_ratio < self.config.animal_pursuit_min_hydration:
            return False
        distance = _target_distance(target)
        if distance <= 0 or distance > self._carrion_seek_max_distance(meat_mode):
            return False
        strength = _float(target.get("strength", 0.0))
        if energy_ratio < self.config.desperate_energy_threshold and strength < 0.12:
            return False
        return True

    def _best_carrion_resource_move(
        self,
        patch_cells: list[dict[str, object]],
        action_mask: dict[str, bool],
        *,
        max_distance: int,
    ) -> str | None:
        best_action: str | None = None
        best_score = float("-inf")
        for cell in patch_cells:
            dx = int(cell["dx"])
            dy = int(cell["dy"])
            distance = abs(dx) + abs(dy)
            if distance <= 0 or distance > max_distance:
                continue
            if not bool(cell["in_bounds"]) or cell["terrain"] == "water":
                continue
            if cell["occupant"] == "agent":
                continue
            action = _step_action_for_delta(dx, dy, action_mask)
            if action is None:
                continue
            carcass = _float(cell["carcass_energy"])
            fresh_kill = _float(cell["fresh_kill_energy"])
            resource = carcass + fresh_kill
            if resource <= 0.0:
                continue
            score = resource
            score += _float(cell["carrion_signal"]) * 0.08
            score -= distance * 0.04
            score -= _float(cell["hazard_level"]) * 0.18
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _best_water_access_move(
        self,
        patch_cells: list[dict[str, object]],
        action_mask: dict[str, bool],
        *,
        max_distance: int,
    ) -> str | None:
        best_action: str | None = None
        best_score = float("-inf")
        for cell in patch_cells:
            dx = int(cell["dx"])
            dy = int(cell["dy"])
            distance = abs(dx) + abs(dy)
            if distance <= 0 or distance > max_distance:
                continue
            if not bool(cell["in_bounds"]) or cell["terrain"] == "water":
                continue
            if cell["occupant"] == "agent":
                continue
            if cell["water_access_reason"] == "none":
                continue
            action = _step_action_for_delta(dx, dy, action_mask)
            if action is None:
                continue
            score = 1.0
            score -= distance * 0.12
            score -= _float(cell["hazard_level"]) * 0.25
            score -= _float(cell["predator_risk"]) * 0.08
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _decision(self, action: str, source: str) -> ActionDecision:
        if action not in ACTION_NAMES:
            action = "stay"
        return ActionDecision(
            requested_action=action,
            source=source,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )


def _self_state(observation: dict[str, object]) -> dict[str, object]:
    self_state = observation.get("self")
    if not isinstance(self_state, dict):
        raise ValueError("observation self section must be a mapping")
    return self_state


def _patch_cells(observation: dict[str, object]) -> list[dict[str, object]]:
    patch_cells = observation.get("local_patch")
    if not isinstance(patch_cells, list):
        raise ValueError("observation local_patch section must be a list")
    if not all(isinstance(cell, dict) for cell in patch_cells):
        raise ValueError("observation local_patch cells must be mappings")
    return patch_cells


def _navigation(observation: dict[str, object]) -> dict[str, dict[str, object]]:
    navigation = observation.get("navigation")
    if not isinstance(navigation, dict):
        raise ValueError("observation navigation section must be a mapping")
    required = ("water", "plant", "carrion", "prey")
    for target in required:
        if not isinstance(navigation.get(target), dict):
            raise ValueError(f"observation navigation target {target} must be a mapping")
    return {target: navigation[target] for target in required}


def _cell_at(
    patch_cells: list[dict[str, object]],
    dx: int,
    dy: int,
) -> dict[str, object] | None:
    return next(
        (
            cell
            for cell in patch_cells
            if int(cell["dx"]) == dx and int(cell["dy"]) == dy
        ),
        None,
    )


def _step_action_for_delta(
    dx: int,
    dy: int,
    action_mask: dict[str, bool],
) -> str | None:
    candidates: list[str] = []
    if abs(dx) >= abs(dy):
        candidates.append("move_east" if dx > 0 else "move_west")
        if dy != 0:
            candidates.append("move_south" if dy > 0 else "move_north")
    else:
        candidates.append("move_south" if dy > 0 else "move_north")
        if dx != 0:
            candidates.append("move_east" if dx > 0 else "move_west")
    for action in candidates:
        if _valid(action_mask, action):
            return action
    return None


def _navigation_action(
    target: dict[str, object],
    action_mask: dict[str, bool],
    *,
    min_strength: float = 0.0,
    allow_detour: bool = False,
) -> str | None:
    if _float(target.get("strength", 0.0)) < min_strength:
        return None
    dx = int(target.get("dx", 0))
    dy = int(target.get("dy", 0))
    if dx == 0 and dy == 0:
        return None
    direct_step = _step_action_for_delta(dx, dy, action_mask)
    if direct_step is not None or not allow_detour:
        return direct_step
    if dx != 0 and dy == 0:
        for action in ("move_north", "move_south"):
            if _valid(action_mask, action):
                return action
    if dy != 0 and dx == 0:
        for action in ("move_east", "move_west"):
            if _valid(action_mask, action):
                return action
    return None


def _move_action_for_delta(dx: int, dy: int) -> str | None:
    for action, delta in MOVE_DELTAS.items():
        if delta == (dx, dy):
            return action
    return None


def _target_distance(target: dict[str, object]) -> int:
    distance = target.get("distance", 0)
    if isinstance(distance, bool) or not isinstance(distance, int):
        return 0
    return max(0, distance)


def _valid(action_mask: dict[str, bool], action: str) -> bool:
    return bool(action_mask.get(action, False))


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
