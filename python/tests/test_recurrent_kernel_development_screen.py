from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from torch.nn import functional as F

    from evolution_sim.cli import recurrent_kernel_development_screen as cli
    from evolution_sim.mind import recurrent_kernel_development_screen as screen
    from evolution_sim.mind import recurrent_actor_critic
    from evolution_sim.mind.recurrent_kernel_development_screen import (
        RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES,
        _candidate_integrity_gate,
        _fixed_row_tile_4_linear,
        recurrent_kernel_candidate,
        select_development_candidate,
        write_development_screen_report,
    )


def _completed_result(
    candidate: str,
    *,
    scalar_ns: int,
    batched_ns: int,
    rss: int = 100,
    max_ratio: float = 0.25,
    wins: int = 5,
) -> dict[str, object]:
    semantic = "a" * 64
    return {
        "candidate": candidate,
        "status": "completed",
        "peak_rss_bytes": rss,
        "equivalence": {
            "action_mismatch_count": 0,
            "max_hidden_tolerance_ratio": max_ratio,
            "scalar_semantic_sha256": semantic,
            "batched_semantic_sha256": semantic,
            "reference_semantic_sha256": semantic,
            "bucket_matrix": {
                "action_mismatch_count": 0,
                "max_hidden_tolerance_ratio": max_ratio,
                "scalar_semantic_sha256": semantic,
                "batched_semantic_sha256": semantic,
                "reference_semantic_sha256": semantic,
            },
        },
        "collector_equivalence": {
            "semantic_mismatch_count": 0,
            "identity_mismatch_count": 0,
            "hidden_shape_mismatch_count": 0,
            "bootstrap_none_mismatch_count": 0,
            "max_input_hidden_tolerance_ratio": max_ratio,
        },
        "timing": {
            "paired_batched_wins": wins,
            "scalar": {
                "median_elapsed_ns": scalar_ns,
                "semantic_sha256": [semantic] * 5,
            },
            "batched": {
                "median_elapsed_ns": batched_ns,
                "semantic_sha256": [semantic] * 5,
            },
        },
    }


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentKernelDevelopmentScreenTests(unittest.TestCase):
    def test_candidate_context_restores_production_functions(self) -> None:
        original_linear = recurrent_actor_critic._backend_stable_linear
        original_gru = recurrent_actor_critic.BackendStableGRU.forward
        with recurrent_kernel_candidate("native_linear_manual_gru_v1"):
            self.assertIsNot(
                recurrent_actor_critic._backend_stable_linear,
                original_linear,
            )
            self.assertIs(
                recurrent_actor_critic.BackendStableGRU.forward,
                original_gru,
            )
        self.assertIs(
            recurrent_actor_critic._backend_stable_linear,
            original_linear,
        )
        self.assertIs(
            recurrent_actor_critic.BackendStableGRU.forward,
            original_gru,
        )

        with recurrent_kernel_candidate("native_linear_fused_gru_speed_ceiling_v1"):
            self.assertIs(
                recurrent_actor_critic.BackendStableGRU.forward,
                torch.nn.GRU.forward,
            )
        self.assertIs(
            recurrent_actor_critic.BackendStableGRU.forward,
            original_gru,
        )

    def test_fixed_tile_matches_one_invariant_dense_tile(self) -> None:
        inputs = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7.0
        bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
        observed = _fixed_row_tile_4_linear(inputs, weight, bias)
        expected = torch.cat(
            (
                F.linear(inputs[:4], weight, bias),
                F.linear(
                    torch.cat((inputs[4:], torch.zeros(3, 3)), dim=0),
                    weight,
                    bias,
                )[:1],
            )
        )
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    def test_selection_requires_integrity_and_absolute_speed(self) -> None:
        baseline = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
            scalar_ns=1000,
            batched_ns=1100,
        )
        native = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[1],
            scalar_ns=900,
            batched_ns=850,
        )
        fused = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[2],
            scalar_ns=800,
            batched_ns=700,
            max_ratio=0.8,
        )
        tiled = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[3],
            scalar_ns=1300,
            batched_ns=1200,
        )

        def integrity(
            result: dict[str, object],
            **_kwargs: object,
        ) -> dict[str, object]:
            if result["candidate"] == "native_linear_fused_gru_speed_ceiling_v1":
                return {
                    "passed": False,
                    "reasons": ["numeric headroom exceeded"],
                    "max_numeric_tolerance_ratio": 0.8,
                }
            return {
                "passed": True,
                "reasons": [],
                "max_numeric_tolerance_ratio": 0.25,
            }

        confirmation = copy.deepcopy(baseline)
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            selection = select_development_candidate(
                [baseline, native, fused, tiled],
                baseline_confirmation=confirmation,
            )
        self.assertEqual(
            selection["selected_candidate"],
            "native_linear_manual_gru_v1",
        )
        self.assertTrue(selection["selection_authorized"])
        gates = selection["candidate_gates"]
        self.assertIn(
            "numeric headroom exceeded",
            " ".join(gates["native_linear_fused_gru_speed_ceiling_v1"]["reasons"]),
        )
        self.assertIn(
            "baseline scalar",
            " ".join(gates["fixed_row_tile_4_linear_manual_gru_v1"]["reasons"]),
        )

        native["equivalence"]["scalar_semantic_sha256"] = "b" * 64
        with patch.object(
            screen,
            "_candidate_integrity_gate",
            side_effect=integrity,
        ):
            drifted = select_development_candidate(
                [baseline, native, fused, tiled],
                baseline_confirmation=confirmation,
            )
        self.assertIsNone(drifted["selected_candidate"])
        self.assertIn(
            "behavior semantics differ",
            " ".join(
                drifted["candidate_gates"]["native_linear_manual_gru_v1"]["reasons"]
            ),
        )

    def test_missing_evidence_and_duplicate_candidates_fail_closed(self) -> None:
        incomplete = _candidate_integrity_gate(
            {
                "candidate": "native_linear_manual_gru_v1",
                "status": "completed",
            }
        )
        self.assertFalse(incomplete["passed"])
        self.assertTrue(incomplete["reasons"])

        baseline = _completed_result(
            RECURRENT_KERNEL_DEVELOPMENT_SCREEN_CANDIDATES[0],
            scalar_ns=1000,
            batched_ns=1100,
        )
        duplicate = select_development_candidate(
            [baseline, baseline, baseline, baseline],
            baseline_confirmation=copy.deepcopy(baseline),
        )
        self.assertFalse(duplicate["selection_authorized"])
        self.assertIn("order", duplicate["reason"])

    def test_report_requires_development_namespace_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            development_root = root / "development-screens"
            with patch.dict(
                os.environ,
                {
                    "EVOLUTION_SIM_OPEN_ECOLOGY_DEVELOPMENT_SCREEN_ROOT": str(
                        development_root
                    )
                },
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "approved",
                ):
                    write_development_screen_report(
                        root / "authority" / "development-screens" / "report.json",
                        {"development_only": True},
                    )
                report = development_root / "report.json"
                write_development_screen_report(
                    report,
                    {"development_only": True},
                )
                self.assertTrue(report.is_file())
                with self.assertRaisesRegex(ValueError, "clobbering"):
                    write_development_screen_report(
                        report,
                        {"development_only": True},
                    )

    def test_cli_is_explicitly_development_only(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--development-run"):
            cli.main(["--expected-source-sha", "a" * 40])
        with self.assertRaisesRegex(SystemExit, "canonical .json"):
            cli.main(
                [
                    "--development-run",
                    "--expected-source-sha",
                    "a" * 40,
                    "--report",
                    "screen.pt",
                ]
            )
        with self.assertRaisesRegex(SystemExit, "--child-nonce"):
            cli.main(
                [
                    "--development-run",
                    "--expected-source-sha",
                    "a" * 40,
                    "--child-candidate",
                    "native_linear_manual_gru_v1",
                ]
            )
        closed_report = {
            "selection": {"selected_candidate": None},
            "development_only": True,
            "launch_authorized": False,
        }
        with (
            patch.object(cli, "validate_development_screen_report_path"),
            patch.object(cli, "run_development_screen", return_value=closed_report),
            patch.object(cli, "write_development_screen_report") as write_report,
        ):
            observed = cli.main(
                [
                    "--development-run",
                    "--expected-source-sha",
                    "a" * 40,
                ]
            )
        self.assertEqual(observed, 0)
        write_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
