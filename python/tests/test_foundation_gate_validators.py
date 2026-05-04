from __future__ import annotations

import unittest
from types import SimpleNamespace

from evolution_sim.cli.foundation_gate_validators.summary import (
    _resource_pressure_budget_flags,
)
from evolution_sim.cli.foundation_gate_validators.trajectory import (
    _signal_action_capacity_flags,
)


def _action_contract(*, token_count: int = 2) -> dict[str, object]:
    action_keys = [
        f"signal_{token_index}_profile_{profile_index}"
        for token_index in range(token_count)
        for profile_index in range(2)
    ]
    return {
        "communication": {
            "token_count": token_count,
            "profiles_per_token": 2,
            "action_keys": action_keys,
            "emission_enabled": False,
        },
        "reserved_action_keys": ["mate", *action_keys],
        "active_action_keys": ["stay"],
    }


class FoundationGateValidatorModuleTests(unittest.TestCase):
    def test_signal_capacity_validator_reports_action_contract_drift(self) -> None:
        good_action_contract = _action_contract(token_count=2)
        signal_contract = {
            "communication_token_count": 2,
            "communication_profiles_per_token": 2,
            "communication_signal_emission_enabled": False,
            "reserved_profiles": [{}, {}, {}, {}],
        }
        observation_contract = {
            "action_contract": good_action_contract,
            "action_names": ["stay", "mate", *good_action_contract["reserved_action_keys"][1:]],
        }

        self.assertEqual(
            _signal_action_capacity_flags(
                scope="unit",
                trajectory={"action_contract": good_action_contract},
                observation_contract=observation_contract,
                signal_contract=signal_contract,
            ),
            [],
        )

        stale_action_contract = _action_contract(token_count=1)
        flags = _signal_action_capacity_flags(
            scope="unit",
            trajectory={"action_contract": stale_action_contract},
            observation_contract=observation_contract,
            signal_contract=signal_contract,
        )

        self.assertEqual(flags[0]["field"], "viewer.trajectory.action_contract")

    def test_resource_pressure_validator_is_independent_from_gate_cli(self) -> None:
        profile = SimpleNamespace(
            min_plant_energy_available_per_land_tile_warning=0.5,
        )
        flags = _resource_pressure_budget_flags(
            scope="unit",
            run={
                "land_tile_count": 10,
                "resource_pressure": {
                    "plant_budget": {"energy_available_at_end": 1.0},
                },
            },
            profile=profile,
        )

        self.assertEqual(
            flags[0]["field"],
            "resource_pressure.plant_budget.energy_available_per_land_tile",
        )


if __name__ == "__main__":
    unittest.main()
