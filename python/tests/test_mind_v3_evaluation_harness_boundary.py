from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
MIND_ROOT = ROOT / "python" / "evolution_sim" / "mind"


class MindV3EvaluationHarnessBoundaryTests(unittest.TestCase):
    def test_mind_modules_do_not_import_cli_as_library(self) -> None:
        offenders: list[str] = []
        for path in sorted(MIND_ROOT.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "evolution_sim.cli" in text:
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])

    def test_live_ab_clis_do_not_import_evaluate_cli_helpers(self) -> None:
        for relative_path in (
            "python/evolution_sim/cli/mind_v3_sequence_history_live_ab.py",
            "python/evolution_sim/cli/mind_v3_transition_value_live_ab.py",
        ):
            with self.subTest(path=relative_path):
                text = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("mind_v3_evaluate", text)
                self.assertIn("from evolution_sim.mind.", text)


if __name__ == "__main__":
    unittest.main()
