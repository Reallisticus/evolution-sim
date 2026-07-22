from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LAUNCHER = _REPOSITORY_ROOT / "scripts" / "run_recurrent_scale_campaign_gpu.sh"


class RecurrentScaleGpuLauncherTests(unittest.TestCase):
    def test_launcher_is_executable_and_has_valid_bash_syntax(self) -> None:
        self.assertTrue(os.access(_LAUNCHER, os.X_OK))
        subprocess.run(["bash", "-n", str(_LAUNCHER)], check=True)
        help_result = subprocess.run(
            [str(_LAUNCHER), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--workspace-parent", help_result.stdout)
        self.assertIn("--output-parent", help_result.stdout)

    def test_launcher_contract_is_isolated_resumable_and_bounded(self) -> None:
        source = _LAUNCHER.read_text(encoding="utf-8")

        for required in (
            'git -C "$checkout" status --porcelain --untracked-files=all',
            'git -C "$scale_checkout" checkout --detach',
            "runtime-provenance.json",
            "mind_v3_public_recurrent_ippo_runtime_provenance",
            "mind_v3_public_recurrent_ippo_cuda_training_smoke",
            "preflight/cuda-training-smoke",
            "--resume",
            "arm_processes",
            'wait "$arm_process"',
            "scale_concurrent_worker_budget=$((3 * scale_max_workers))",
            "tmux new-session",
        ):
            self.assertIn(required, source)
        self.assertNotIn("gpu4070", source)
        self.assertNotIn("/home/", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertLess(
            source.index("run_cuda_training_resume_smoke"),
            source.index("tmux new-session"),
        )


if __name__ == "__main__":
    unittest.main()
