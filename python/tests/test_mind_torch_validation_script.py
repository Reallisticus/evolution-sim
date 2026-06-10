from __future__ import annotations

import unittest

from scripts import validate_mind_torch


class MindTorchValidationScriptTests(unittest.TestCase):
    def test_discovers_torch_gated_mind_tests(self) -> None:
        tests = validate_mind_torch._discover_torch_gated_tests()

        self.assertIn(
            "test_torch_actor_critic_trainer_writes_real_ml_artifact",
            tests,
        )
        self.assertGreater(len(tests), 1)

    def test_dependency_report_covers_optional_mind_ml_packages(self) -> None:
        versions = validate_mind_torch._dependency_versions()

        self.assertEqual(
            set(versions),
            set(validate_mind_torch.OPTIONAL_ML_PACKAGES),
        )

    def test_runtime_environment_records_reproducibility_context(self) -> None:
        environment = validate_mind_torch._runtime_environment()

        self.assertIn("python_version", environment)
        self.assertIn("git_branch", environment)
        self.assertIn("git_head", environment)
        self.assertIn("git_dirty", environment)


if __name__ == "__main__":
    unittest.main()
