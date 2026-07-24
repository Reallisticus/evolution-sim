from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from evolution_sim.cli.foundation_gate_validators.summary import (
    _resource_pressure_budget_flags,
)
from evolution_sim.cli.foundation_gate_validators.trajectory import (
    _mind_contract_flags,
    _signal_action_capacity_flags,
)
from evolution_sim.config import SignalConfig, WorldConfig
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import (
    action_contract,
    action_names,
)


def _action_contract(*, token_count: int = 2) -> dict[str, object]:
    return action_contract(
        SignalConfig(
            communication_token_count=token_count,
            communication_profiles_per_token=2,
        )
    )


class FoundationGateValidatorModuleTests(unittest.TestCase):
    def test_mind_contract_validator_accepts_consistent_tokenized_replay(self) -> None:
        result = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=2,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=2,
                    communication_profiles_per_token=1,
                ),
            )
        ).run()

        self.assertEqual(
            _mind_contract_flags(
                scope="unit",
                summary=result.summary,
                viewer=result.viewer,
                events=result.events,
            ),
            [],
        )
        mixed_viewer = copy.deepcopy(result.viewer)
        mixed_viewer["trajectory"]["observation_contract"]["policy_input"][
            "encoder_version"
        ] = "mind_observation_encoder_v2"
        flags = _mind_contract_flags(
            scope="unit",
            summary=result.summary,
            viewer=mixed_viewer,
            events=result.events,
        )

        self.assertIn(
            "viewer.trajectory.observation_contract.policy_input",
            {flag["field"] for flag in flags},
        )

    def test_mind_contract_validator_rejects_mixed_token_contracts(self) -> None:
        token_result = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=2,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=2,
                    communication_profiles_per_token=1,
                ),
            )
        ).run()
        legacy_result = SimulationWorld(WorldConfig(seed=7, max_ticks=2)).run()
        three_token_result = SimulationWorld(
            WorldConfig(
                seed=7,
                max_ticks=2,
                signals=SignalConfig(
                    communication_signal_emission_enabled=True,
                    communication_token_count=3,
                    communication_profiles_per_token=1,
                ),
            )
        ).run()

        def replace_with_legacy_input(
            summary: dict[str, object],
            viewer: dict[str, object],
        ) -> None:
            del summary
            viewer["trajectory"]["records"][0]["observation_input"] = copy.deepcopy(
                legacy_result.viewer["trajectory"]["records"][0][
                    "observation_input"
                ]
            )

        def replace_with_three_token_input(
            summary: dict[str, object],
            viewer: dict[str, object],
        ) -> None:
            del summary
            viewer["trajectory"]["records"][0]["observation_input"] = copy.deepcopy(
                three_token_result.viewer["trajectory"]["records"][0][
                    "observation_input"
                ]
            )

        def assign_token_meaning(
            summary: dict[str, object],
            viewer: dict[str, object],
        ) -> None:
            del summary
            trajectory = viewer["trajectory"]
            trajectory["action_contract"]["communication"][
                "meaning"
            ] = "token_0_means_food"
            trajectory["observation_contract"]["action_contract"]["communication"][
                "meaning"
            ] = "token_0_means_food"

        mutations = (
            (
                "legacy summary schema",
                lambda summary, viewer: summary.__setitem__(
                    "summary_schema_version",
                    "foundation_summary_v1",
                ),
                "summary.summary_schema_version",
            ),
            (
                "legacy observation input",
                replace_with_legacy_input,
                "viewer.trajectory.records.observation_input",
            ),
            (
                "wrong token-count observation input",
                replace_with_three_token_input,
                "viewer.trajectory.records.observation_input",
            ),
            (
                "reversed observation token order",
                lambda summary, viewer: viewer["trajectory"][
                    "observation_contract"
                ]["communication_token_channels"].__setitem__(
                    "token_order",
                    [1, 0],
                ),
                "viewer.trajectory.observation_contract.communication_token_channels",
            ),
            (
                "stale nested signal schema",
                lambda summary, viewer: viewer["trajectory"][
                    "observation_contract"
                ]["signal_contract"]["communication_token_observation"].__setitem__(
                    "schema_version",
                    "stale",
                ),
                "viewer.trajectory.observation_contract.communication_token_channels",
            ),
            (
                "policy-hidden token fields",
                lambda summary, viewer: viewer["trajectory"][
                    "observation_contract"
                ]["communication_token_channels"].__setitem__(
                    "policy_visible",
                    False,
                ),
                "viewer.trajectory.observation_contract.communication_token_channels",
            ),
            (
                "reversed policy token fields",
                lambda summary, viewer: viewer["trajectory"][
                    "observation_contract"
                ]["policy_input"].__setitem__(
                    "patch_input_fields",
                    [
                        *viewer["trajectory"]["observation_contract"][
                            "policy_input"
                        ]["patch_input_fields"][:-2],
                        "communication_signal_token_1",
                        "communication_signal_token_0",
                    ],
                ),
                "viewer.trajectory.observation_contract.communication_token_channels",
            ),
            (
                "missing frame token matrix",
                lambda summary, viewer: viewer["frames"][0][
                    "signal_fields"
                ].pop("communication_signal_token_0"),
                "viewer.frames.signal_emissions",
            ),
            (
                "extra frame token matrix",
                lambda summary, viewer: viewer["frames"][0][
                    "signal_fields"
                ].__setitem__("communication_signal_token_2", [[0.0]]),
                "viewer.frames.signal_emissions",
            ),
            (
                "out-of-range reserved profile token",
                lambda summary, viewer: viewer["trajectory"][
                    "observation_contract"
                ]["signal_contract"].__setitem__(
                    "reserved_profiles",
                    [
                        *viewer["trajectory"]["observation_contract"][
                            "signal_contract"
                        ]["reserved_profiles"],
                        *(
                            {
                                "token_id": token_id,
                                "profile_index": 0,
                                "field_name": (
                                    f"communication_signal_token_{token_id}"
                                ),
                                "policy_visible": False,
                            }
                            for token_id in range(2, 100)
                        ),
                    ],
                ),
                (
                    "viewer.trajectory.observation_contract.signal_contract."
                    "reserved_profiles"
                ),
            ),
            (
                "assigned token meaning",
                assign_token_meaning,
                "viewer.trajectory.action_contract",
            ),
            (
                "token reward contract",
                lambda summary, viewer: viewer["trajectory"]["reward_contract"][
                    "component_bounds"
                ].__setitem__("communication_token_0_bonus", [0.0, 1.0]),
                "viewer.trajectory.reward_contract",
            ),
            (
                "token reward record",
                lambda summary, viewer: viewer["trajectory"]["records"][0][
                    "reward"
                ]["components"].__setitem__("communication_token_0_bonus", 1.0),
                "viewer.trajectory.records.reward",
            ),
        )

        for name, mutate, expected_field in mutations:
            with self.subTest(name=name):
                summary = copy.deepcopy(token_result.summary)
                viewer = copy.deepcopy(token_result.viewer)
                mutate(summary, viewer)
                flags = _mind_contract_flags(
                    scope="unit",
                    summary=summary,
                    viewer=viewer,
                    events=token_result.events,
                )
                self.assertIn(
                    expected_field,
                    {flag["field"] for flag in flags},
                )

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
            "action_names": list(
                action_names(
                    SignalConfig(
                        communication_token_count=2,
                        communication_profiles_per_token=2,
                    )
                )
            ),
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
