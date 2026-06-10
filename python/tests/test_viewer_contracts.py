from __future__ import annotations

import unittest
from pathlib import Path

from evolution_sim.env.contracts import VIEWER_AGENT_ENCODING
from evolution_sim.env.runtime.action_contract import ACTION_CONTRACT_VERSION
from evolution_sim.env.runtime.feeding import FOOD_SOURCES
from evolution_sim.env.runtime.mating import (
    ASEXUAL_REPRODUCTION_MODE,
    SEXUAL_REPRODUCTION_MODE,
)
from evolution_sim.env.runtime.observations import (
    OBSERVATION_INPUT_VECTOR_SIZE,
    OBSERVATION_SCHEMA_VERSION,
)
from evolution_sim.env.runtime.policy import POLICY_INTERFACE_VERSION
from evolution_sim.env.runtime.reporting_contracts import (
    ECOLOGY_STATES,
    HABITAT_STATES,
    HAZARD_TYPES,
    HYDROLOGY_REASONS,
    MEAT_MODE_SERIES,
    REFUGE_REASONS,
    TROPHIC_ROLES,
)
from evolution_sim.env.runtime.trajectory import TRAJECTORY_SCHEMA_VERSION
from evolution_sim.env.viewer_contracts import (
    render_viewer_contract_module,
    viewer_contract_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class ViewerContractsTests(unittest.TestCase):
    def test_viewer_contract_payload_uses_python_contract_versions(self) -> None:
        payload = viewer_contract_payload()

        self.assertEqual(
            payload["REQUIRED_AGENT_FIELDS"],
            list(VIEWER_AGENT_ENCODING),
        )
        self.assertEqual(
            payload["TRAJECTORY_SCHEMA_VERSION"],
            TRAJECTORY_SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["OBSERVATION_SCHEMA_VERSION"],
            OBSERVATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["OBSERVATION_INPUT_VECTOR_SIZE"],
            OBSERVATION_INPUT_VECTOR_SIZE,
        )
        self.assertEqual(
            payload["POLICY_INTERFACE_VERSION"],
            POLICY_INTERFACE_VERSION,
        )
        self.assertEqual(
            payload["ACTION_CONTRACT_VERSION"],
            ACTION_CONTRACT_VERSION,
        )

    def test_viewer_display_labels_cover_replay_and_event_vocabulary(self) -> None:
        payload = viewer_contract_payload()
        labels = payload["VIEWER_DISPLAY_LABELS"]

        expected_domains = {
            "trophic_role": ("none", *TROPHIC_ROLES),
            "meat_mode": MEAT_MODE_SERIES,
            "water_access_reason": HYDROLOGY_REASONS,
            "soft_refuge_reason": REFUGE_REASONS,
            "hazard_type": HAZARD_TYPES,
            "habitat_state": ("non_land", *HABITAT_STATES),
            "ecology_state": ECOLOGY_STATES,
            "food_source": tuple(sorted(FOOD_SOURCES)),
            "reproduction_mode": (
                ASEXUAL_REPRODUCTION_MODE,
                SEXUAL_REPRODUCTION_MODE,
            ),
            "damage_source": (
                "none",
                "attack",
                "hazard_exposure",
                "hazard_instability",
                "health_depletion",
            ),
            "death_cause": (
                "unknown",
                "attack",
                "hazard_exposure",
                "hazard_instability",
                "health_depletion",
                "energy_depletion",
                "hydration_depletion",
                "old_age",
            ),
        }
        for domain, values in expected_domains.items():
            with self.subTest(domain=domain):
                self.assertEqual(set(labels[domain]), set(values))
                self.assertTrue(all(labels[domain][value] for value in values))

        self.assertEqual(labels["meat_mode"]["none"], "Plant-focused")
        self.assertEqual(labels["water_access_reason"]["none"], "No water access")
        self.assertEqual(labels["hazard_type"]["none"], "No active hazard")
        self.assertEqual(labels["habitat_state"]["non_land"], "Non-land")

    def test_viewer_empty_labels_are_generated_with_contracts(self) -> None:
        payload = viewer_contract_payload()

        self.assertEqual(
            payload["VIEWER_EMPTY_LABELS"],
            {
                "missing": "-",
                "unknown": "Unknown",
                "not_applicable": "Not applicable",
                "still_alive": "Still alive",
                "no_recorded_death": "No recorded death",
            },
        )

    def test_tracked_viewer_contract_module_is_current(self) -> None:
        generated_path = REPO_ROOT / "viewer/contracts.generated.mjs"

        self.assertEqual(
            generated_path.read_text(),
            render_viewer_contract_module(),
        )


if __name__ == "__main__":
    unittest.main()
