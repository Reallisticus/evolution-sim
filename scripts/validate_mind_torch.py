#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (PYTHON_ROOT, REPO_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.baseline import train_baseline_with_trainer
from evolution_sim.mind.dataset import dataset_provenance, load_trajectory_jsonl

TORCH_VALIDATION_SCHEMA_VERSION = "mind_torch_validation_v1"
TORCH_VALIDATION_POLICY = "explicit_optional_mind_ml_stack_validation_v1"
TORCH_TEST_MODULE = "tests.test_mind_v1"
TORCH_TEST_CLASS = "MindV1Tests"
TORCH_TEST_FILE = REPO_ROOT / "python/tests/test_mind_v1.py"
OPTIONAL_ML_PACKAGES = (
    "torch",
    "torchrl",
    "tensordict",
    "gymnasium",
    "pettingzoo",
    "minari",
    "d3rlpy",
)


def _discover_torch_gated_tests() -> tuple[str, ...]:
    source = TORCH_TEST_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TORCH_TEST_FILE))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != TORCH_TEST_CLASS:
            continue
        names: list[str] = []
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                continue
            if any(_is_torch_skip_decorator(source, dec) for dec in item.decorator_list):
                names.append(item.name)
        return tuple(names)
    raise SystemExit(f"{TORCH_TEST_CLASS} not found in {TORCH_TEST_FILE}")


def _is_torch_skip_decorator(source: str, node: ast.expr) -> bool:
    text = ast.get_source_segment(source, node) or ""
    return (
        "skipUnless" in text
        and "find_spec" in text
        and ("\"torch\"" in text or "'torch'" in text)
    )


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in OPTIONAL_ML_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _runtime_environment() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_branch": _git_output("branch", "--show-current"),
        "git_head": _git_output("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_output("status", "--short")),
    }


def _require_torch(*, require_cuda: bool) -> Any:
    if importlib.util.find_spec("torch") is None:
        raise SystemExit(
            "PyTorch is not installed. Install requirements-mind-ml.txt or run "
            "`npm run trainer -- run npm run sim:mind:torch:validate:cuda` "
            "on the configured GPU trainer."
        )
    torch = importlib.import_module("torch")
    if require_cuda and not bool(torch.cuda.is_available()):
        raise SystemExit(
            "CUDA validation requested, but torch.cuda.is_available() is false. "
            "Run this on the configured NVIDIA trainer or check its driver/CUDA stack."
        )
    return torch


def _torch_environment(torch: Any) -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": cuda_available,
        "cuda_version": (
            None if getattr(torch, "version", None) is None else torch.version.cuda
        ),
        "cuda_device_count": device_count,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
        ),
        "mps_available": bool(
            getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
        ),
    }


def _run_torch_unittests(test_names: tuple[str, ...], *, verbosity: int) -> dict[str, object]:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in test_names:
        suite.addTests(
            loader.loadTestsFromName(f"{TORCH_TEST_MODULE}.{TORCH_TEST_CLASS}.{name}")
        )
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=verbosity).run(suite)
    return {
        "test_module": TORCH_TEST_MODULE,
        "test_class": TORCH_TEST_CLASS,
        "test_count": int(result.testsRun),
        "selected_tests": list(test_names),
        "seconds": round(time.perf_counter() - started, 4),
        "passed": result.wasSuccessful() and not result.skipped,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skip_count": len(result.skipped),
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skips": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
        "runner_output": stream.getvalue(),
    }


def _write_tiny_trajectory(path: Path, *, seed: int = 7) -> None:
    writer = JsonlTrajectoryWriter(path, source_seeds=[seed], split_id="torch_cuda_smoke")
    SimulationWorld(WorldConfig(seed=seed, max_ticks=2)).run(
        mode=RunMode.SUMMARY_ONLY,
        trajectory_sink=writer,
    )


def _run_cuda_training_smoke() -> dict[str, object]:
    with TemporaryDirectory() as tmpdir:
        trajectory_path = Path(tmpdir) / "trajectory.jsonl.gz"
        _write_tiny_trajectory(trajectory_path)
        dataset = load_trajectory_jsonl(trajectory_path)
        artifact = train_baseline_with_trainer(
            dataset.records,
            provenance=dataset_provenance(dataset),
            trainer="torch-actor-critic-bc",
            torch_device="cuda",
        ).to_artifact()
    model = artifact["model"]
    device_metadata = dict(model["torch_device_metadata"])
    passed = device_metadata.get("resolved_device") == "cuda"
    return {
        "passed": passed,
        "trainer": model.get("trainer"),
        "model_type": artifact["manifest"].get("model_type"),
        "requested_device": device_metadata.get("requested_device"),
        "resolved_device": device_metadata.get("resolved_device"),
        "cuda_available": device_metadata.get("cuda_available"),
        "cuda_device_count": device_metadata.get("cuda_device_count"),
        "cuda_device_name": device_metadata.get("cuda_device_name"),
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the optional Mind PyTorch/ML stack. This is intended for "
            "the configured NVIDIA trainer or any local environment with the "
            "Mind ML dependencies installed."
        )
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless torch can see CUDA, then run a tiny CUDA training smoke.",
    )
    parser.add_argument(
        "--skip-cuda-smoke",
        action="store_true",
        help="With --require-cuda, only verify CUDA availability and torch-gated tests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path, usually under output/mind/ on the trainer.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered torch-gated unittest methods and exit.",
    )
    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        help="Run a specific MindV1Tests method. Repeat to run a subset.",
    )
    parser.add_argument("--verbosity", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    discovered_tests = _discover_torch_gated_tests()
    if args.list:
        for name in discovered_tests:
            print(name)
        print(f"torch_gated_test_count={len(discovered_tests)}")
        return 0
    test_names = tuple(args.tests) if args.tests else discovered_tests
    unknown = sorted(set(test_names) - set(discovered_tests))
    if unknown:
        raise SystemExit("unknown torch-gated test(s): " + ", ".join(unknown))
    torch = _require_torch(require_cuda=args.require_cuda)
    report: dict[str, object] = {
        "schema_version": TORCH_VALIDATION_SCHEMA_VERSION,
        "validation_policy": TORCH_VALIDATION_POLICY,
        "require_cuda": bool(args.require_cuda),
        "runtime_environment": _runtime_environment(),
        "dependency_versions": _dependency_versions(),
        "torch_environment": _torch_environment(torch),
    }
    tests = _run_torch_unittests(test_names, verbosity=args.verbosity)
    report["unittest"] = tests
    if args.require_cuda and not args.skip_cuda_smoke:
        report["cuda_training_smoke"] = _run_cuda_training_smoke()
    passed = bool(tests["passed"]) and bool(
        dict(report.get("cuda_training_smoke", {"passed": True})).get("passed", True)
    )
    report["passed"] = passed
    if args.output is not None:
        _write_report(args.output, report)
        print(f"mind_torch_validation_report={args.output}")
    print(f"mind_torch_validation_passed={passed}")
    print(f"torch_gated_test_count={tests['test_count']}")
    print(f"torch_gated_skip_count={tests['skip_count']}")
    environment = dict(report["torch_environment"])
    print(f"torch_version={environment.get('torch_version')}")
    print(f"cuda_available={environment.get('cuda_available')}")
    print(f"cuda_device_name={environment.get('cuda_device_name')}")
    if not passed:
        sys.stdout.write(str(tests["runner_output"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
