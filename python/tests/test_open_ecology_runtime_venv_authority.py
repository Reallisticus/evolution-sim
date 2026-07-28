from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from evolution_sim.io import open_ecology_runtime_venv_authority as runtime


class OpenEcologyRuntimeVenvAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / "source"
        (self.source / "python" / "evolution_sim").mkdir(parents=True)
        (self.source / "python" / "evolution_sim" / "__init__.py").write_text(
            "",
            encoding="utf-8",
        )
        (self.source / "python" / "evolution_sim" / "cli").mkdir()
        (
            self.source
            / "python"
            / "evolution_sim"
            / "cli"
            / "open_ecology_phase_a_guardian.py"
        ).write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.venv = self.base / "venv"
        (self.venv / "bin").mkdir(parents=True)
        self.version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        self.site = self.venv / "lib" / self.version / "site-packages"
        (self.site / "torch").mkdir(parents=True)
        (self.site / "torch" / "__init__.py").write_text(
            "__version__ = 'test'\n",
            encoding="utf-8",
        )
        (self.site / "runtime.bin").write_bytes(b"runtime")
        (self.site / "_virtualenv.pth").write_text(
            "import module_that_must_remain_inert\n",
            encoding="utf-8",
        )
        (self.venv / "pyvenv.cfg").write_text(
            "home = /pinned/base\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )
        self.interpreter = self.base / "python-base"
        shutil.copy2(sys.executable, self.interpreter)
        self.launcher = self.venv / "bin" / "python"
        self.launcher.symlink_to(self.interpreter)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _authority(self) -> dict[str, object]:
        return runtime.build_runtime_venv_authority(
            venv_python=self.launcher,
            source_root=self.source,
            source_git_sha="a" * 40,
            source_manifest_sha256="b" * 64,
            archive_authority_sha256="c" * 64,
            ssh_target="trainer-node",
            ssh_connection_sha256="d" * 64,
        )

    def test_seals_and_revalidates_launcher_cfg_and_complete_site_packages(
        self,
    ) -> None:
        authority = self._authority()
        runtime.validate_runtime_venv_authority(authority)
        launcher = authority["launcher"]
        self.assertEqual(launcher["invocation_path"], str(self.launcher))
        self.assertEqual(launcher["chain"][0]["type"], "symlink")
        self.assertGreater(
            authority["virtualenv"]["site_packages"]["entry_count"],
            0,
        )

        (self.site / "runtime.bin").write_bytes(b"replaced")
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "site-packages content manifest drifted",
        ):
            runtime.validate_runtime_venv_authority(authority)

    def test_launcher_pyvenv_and_symlink_replacements_fail_closed(self) -> None:
        authority = self._authority()
        self.launcher.unlink()
        replacement = self.base / "replacement-python"
        shutil.copy2(sys.executable, replacement)
        self.launcher.symlink_to(replacement)
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "launcher chain drifted",
        ):
            runtime.validate_runtime_venv_authority(authority)

        self.launcher.unlink()
        self.launcher.symlink_to(self.interpreter)
        authority = self._authority()
        (self.venv / "pyvenv.cfg").write_text("hostile = true\n", encoding="utf-8")
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "pyvenv.cfg bytes drifted",
        ):
            runtime.validate_runtime_venv_authority(authority)

        (self.venv / "pyvenv.cfg").write_text(
            "home = /pinned/base\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )
        authority = self._authority()
        (self.site / "torch-link").symlink_to("/tmp")
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "symbolic links are forbidden",
        ):
            runtime.validate_runtime_venv_authority(authority)

    def test_system_site_packages_true_is_rejected(self) -> None:
        (self.venv / "pyvenv.cfg").write_text(
            "home = /pinned/base\ninclude-system-site-packages = true\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "must disable system site-packages",
        ):
            self._authority()

    def test_authority_file_is_externally_hashed_and_published_no_replace(
        self,
    ) -> None:
        authority = self._authority()
        path = self.base / "runtime-authority.json"
        digest = runtime.write_new_runtime_venv_authority(path, authority)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        loaded = runtime.load_runtime_venv_authority(
            path,
            expected_sha256=digest,
        )
        self.assertEqual(loaded, authority)
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "output must be new",
        ):
            runtime.write_new_runtime_venv_authority(path, authority)

        racing = self.base / "racing-authority.json"
        real_link = os.link

        def competitor(source: object, destination: object, **kwargs: object) -> None:
            Path(destination).write_text("competitor", encoding="utf-8")
            real_link(source, destination, **kwargs)

        with (
            mock.patch.object(os, "link", side_effect=competitor),
            self.assertRaisesRegex(
                runtime.RuntimeVenvAuthorityError,
                "appeared during publication",
            ),
        ):
            runtime.write_new_runtime_venv_authority(racing, authority)
        self.assertEqual(racing.read_text(encoding="utf-8"), "competitor")

    def test_bootstrap_execs_only_guardian_with_nonmutating_environment(self) -> None:
        authority = self._authority()
        path = self.base / "runtime-authority.json"
        digest = runtime.write_new_runtime_venv_authority(path, authority)
        observed: dict[str, object] = {}

        def fake_execve(
            executable: str,
            argv: tuple[str, ...],
            environment: dict[str, str],
        ) -> None:
            observed.update(
                {
                    "executable": executable,
                    "argv": argv,
                    "environment": environment,
                }
            )
            raise RuntimeError("exec intercepted")

        with (
            mock.patch.object(os, "execve", side_effect=fake_execve),
            mock.patch.object(
                os,
                "chdir",
                side_effect=lambda path: observed.update({"cwd": path}),
            ),
            mock.patch.object(runtime, "_verify_exact_source"),
            self.assertRaisesRegex(RuntimeError, "exec intercepted"),
        ):
            runtime.exec_guardian_from_runtime_authority(
                authority_path=path,
                expected_authority_sha256=digest,
                expected_source_git_sha="a" * 40,
                expected_source_manifest_sha256="b" * 64,
                expected_archive_authority_sha256="c" * 64,
                expected_ssh_target="trainer-node",
                expected_ssh_connection_sha256="d" * 64,
                git_executable=Path("/usr/bin/git"),
                git_executable_sha256="e" * 64,
                guardian_arguments=("serve", "--device", "cuda"),
            )
        self.assertEqual(observed["executable"], str(self.launcher))
        self.assertEqual(
            observed["argv"][:5],
            (
                str(self.launcher),
                "-I",
                "-S",
                "-c",
                runtime._ISOLATED_RUNTIME_REEXEC_CODE,
            ),
        )
        self.assertEqual(observed["argv"][5], str(self.launcher))
        self.assertEqual(observed["argv"][6], runtime._RUNTIME_GUARDIAN_EXEC_CODE)
        self.assertEqual(observed["cwd"], self.source)
        environment = observed["environment"]
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["PYTHONSAFEPATH"], "1")
        self.assertNotIn("PYTHONPATH", environment)
        serialized = json.dumps(environment)
        self.assertNotIn("TOKEN", serialized.upper())

    def test_safe_no_site_runtime_imports_only_explicit_sealed_paths(self) -> None:
        entrypoint = (
            self.source
            / "python"
            / "evolution_sim"
            / "cli"
            / "open_ecology_phase_a_guardian.py"
        )
        entrypoint.write_text(
            "import hashlib\n"
            "import evolution_sim\n"
            "import sys\n"
            "import torch\n"
            "print(evolution_sim.__file__)\n"
            "print(torch.__file__)\n"
            "print(sys.prefix)\n"
            "print(sys.flags.isolated, sys.flags.no_site, sys.flags.safe_path)\n"
            "print(getattr(hashlib, 'HOSTILE_SOURCE_PYC', False))\n",
            encoding="utf-8",
        )
        hostile_hashlib_source = self.base / "hostile_hashlib.py"
        hostile_hashlib_source.write_text(
            "HOSTILE_SOURCE_PYC = True\n",
            encoding="utf-8",
        )
        py_compile.compile(
            str(hostile_hashlib_source),
            cfile=str(self.source / "python" / "hashlib.pyc"),
            doraise=True,
        )
        hostile = self.base / "hostile-cwd"
        (hostile / "evolution_sim").mkdir(parents=True)
        (hostile / "evolution_sim" / "__init__.py").write_text(
            "raise RuntimeError('cwd shadow executed')\n",
            encoding="utf-8",
        )
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
        authority = self._authority()
        execution = authority["execution"]
        package_init = execution["package_init"]
        guardian_entrypoint = execution["entrypoint"]
        result = subprocess.run(
            (
                str(self.launcher),
                "-I",
                "-S",
                "-c",
                runtime._ISOLATED_RUNTIME_REEXEC_CODE,
                str(self.launcher),
                runtime._RUNTIME_GUARDIAN_EXEC_CODE,
                str(self.source / package_init["relative_path"]),
                package_init["sha256"],
                str(self.source / guardian_entrypoint["relative_path"]),
                guardian_entrypoint["sha256"],
                str(self.site),
                str(self.source / "python"),
                str(self.venv),
            ),
            cwd=hostile,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        output = result.stdout.splitlines()
        self.assertEqual(
            Path(output[0]).resolve(),
            self.source / "python" / "evolution_sim" / "__init__.py",
        )
        self.assertEqual(
            Path(output[1]).resolve(),
            self.site / "torch" / "__init__.py",
        )
        self.assertEqual(Path(output[2]).resolve(), self.venv)
        self.assertEqual(output[3], "0 1 True")
        self.assertEqual(output[4], "False")

    def test_python312_real_venv_revalidates_after_isolated_no_site_relay(
        self,
    ) -> None:
        python312 = Path("/opt/homebrew/bin/python3.12")
        if not python312.is_file():
            self.skipTest("local Python 3.12 is unavailable")
        git_value = shutil.which("git")
        if git_value is None:
            self.skipTest("Git is unavailable")
        git = Path(git_value).resolve(strict=True)
        root = self.base / "python312-source"
        package = root / "python" / "evolution_sim"
        (package / "cli").mkdir(parents=True)
        (package / "io").mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cli" / "__init__.py").write_text("", encoding="utf-8")
        (package / "io" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copy2(
            Path(runtime.__file__),
            package / "io" / "open_ecology_runtime_venv_authority.py",
        )
        entrypoint = package / "cli" / "open_ecology_phase_a_guardian.py"
        entrypoint.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "from evolution_sim.io.open_ecology_runtime_venv_authority "
            "import revalidate_running_guardian_runtime\n"
            "(\n"
            " authority_path, authority_sha, source_sha, manifest_sha,\n"
            " archive_sha, ssh_target, connection_sha, git_path, git_sha,\n"
            ") = sys.argv[1:]\n"
            "revalidate_running_guardian_runtime(\n"
            " authority_path=Path(authority_path),\n"
            " expected_authority_sha256=authority_sha,\n"
            " expected_source_git_sha=source_sha,\n"
            " expected_source_manifest_sha256=manifest_sha,\n"
            " expected_archive_authority_sha256=archive_sha,\n"
            " expected_ssh_target=ssh_target,\n"
            " expected_ssh_connection_sha256=connection_sha,\n"
            " git_executable=Path(git_path),\n"
            " git_executable_sha256=git_sha,\n"
            " observed_entrypoint_path=Path(__file__),\n"
            ")\n"
            "print('REVALIDATED')\n"
            "print(sys.prefix)\n"
            "print(hash('open-ecology-python312'))\n",
            encoding="utf-8",
        )
        venv = self.base / "real-venv312"
        subprocess.run(
            (str(python312), "-m", "venv", "--without-pip", str(venv)),
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        launcher = venv / "bin" / "python"
        clean_git_environment = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        }

        def run_git(*arguments: str) -> str:
            result = subprocess.run(
                (str(git), *arguments),
                cwd=root,
                env=clean_git_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        run_git("init")
        run_git("add", "python")
        run_git(
            "-c",
            "user.name=Guardian Test",
            "-c",
            "user.email=guardian@example.invalid",
            "commit",
            "-m",
            "python312 runtime",
        )
        source_sha = run_git("rev-parse", "HEAD")
        manifest_sha = runtime._source_manifest_aggregate(root)
        archive_sha = "c" * 64
        connection_sha = "d" * 64
        authority = runtime.build_runtime_venv_authority(
            venv_python=launcher,
            source_root=root,
            source_git_sha=source_sha,
            source_manifest_sha256=manifest_sha,
            archive_authority_sha256=archive_sha,
            ssh_target="trainer-node",
            ssh_connection_sha256=connection_sha,
        )
        authority_path = self.base / "python312-runtime-authority.json"
        authority_sha = runtime.write_new_runtime_venv_authority(
            authority_path,
            authority,
        )
        execution = authority["execution"]
        package_init = execution["package_init"]
        guardian_entrypoint = execution["entrypoint"]
        git_sha = hashlib.sha256(git.read_bytes()).hexdigest()
        site_packages = authority["virtualenv"]["site_packages"]["path"]
        command = (
            str(launcher),
            "-I",
            "-S",
            "-c",
            runtime._ISOLATED_RUNTIME_REEXEC_CODE,
            str(launcher),
            runtime._RUNTIME_GUARDIAN_EXEC_CODE,
            str(root / package_init["relative_path"]),
            package_init["sha256"],
            str(root / guardian_entrypoint["relative_path"]),
            guardian_entrypoint["sha256"],
            site_packages,
            str(root / "python"),
            str(venv),
            str(authority_path),
            authority_sha,
            source_sha,
            manifest_sha,
            archive_sha,
            "trainer-node",
            connection_sha,
            str(git),
            git_sha,
        )
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        }
        outputs: list[list[str]] = []
        for _ in range(2):
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            outputs.append(result.stdout.splitlines())
        self.assertEqual(outputs[0][0], "REVALIDATED")
        self.assertEqual(Path(outputs[0][1]), venv)
        self.assertEqual(outputs[0][2], outputs[1][2])

    def test_source_manifest_changes_when_exact_source_is_tampered(self) -> None:
        before = runtime._source_manifest_aggregate(self.source)
        (self.source / "python" / "evolution_sim" / "__init__.py").write_text(
            "hostile = True\n",
            encoding="utf-8",
        )
        after = runtime._source_manifest_aggregate(self.source)
        self.assertNotEqual(before, after)

    def test_exact_source_ignores_hostile_git_config_and_detects_tamper(
        self,
    ) -> None:
        git_value = shutil.which("git")
        if git_value is None:
            self.skipTest("Git is unavailable")
        git = Path(git_value).resolve(strict=True)
        clean_git_environment = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        }

        def run_git(*arguments: str) -> str:
            completed = subprocess.run(
                (str(git), *arguments),
                cwd=self.source,
                env=clean_git_environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()

        run_git("init")
        run_git("add", "python")
        run_git(
            "-c",
            "user.name=Guardian Test",
            "-c",
            "user.email=guardian@example.invalid",
            "commit",
            "-m",
            "exact source",
        )
        expected_git_sha = run_git("rev-parse", "HEAD")
        expected_manifest = runtime._source_manifest_aggregate(self.source)
        marker = self.base / "hostile-fsmonitor-ran"
        hook = self.base / "hostile-fsmonitor"
        hook.write_text(
            f"#!/bin/sh\ntouch {str(marker)!r}\nprintf 'hostile-token\\n'\n",
            encoding="utf-8",
        )
        hook.chmod(0o700)
        hostile_home = self.base / "hostile-home"
        hostile_home.mkdir()
        (hostile_home / ".gitconfig").write_text(
            f"[core]\n\tfsmonitor = {hook}\n",
            encoding="utf-8",
        )
        run_git("config", "core.fsmonitor", str(hook))

        with mock.patch.dict(os.environ, {"HOME": str(hostile_home)}):
            runtime._verify_exact_source(
                source_root=self.source,
                expected_git_sha=expected_git_sha,
                expected_manifest_sha256=expected_manifest,
                git_executable=git,
                git_executable_sha256=hashlib.sha256(git.read_bytes()).hexdigest(),
            )
        self.assertFalse(marker.exists())

        package_init = self.source / "python" / "evolution_sim" / "__init__.py"
        hostile_source = self.base / "hostile_package_init.py"
        hostile_source.write_text(
            "HOSTILE_PACKAGE_PYC = True\n",
            encoding="utf-8",
        )
        cache_path = (
            package_init.parent
            / "__pycache__"
            / f"__init__.{sys.implementation.cache_tag}.pyc"
        )
        cache_path.parent.mkdir()
        py_compile.compile(
            str(hostile_source),
            cfile=str(cache_path),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
        cached = bytearray(cache_path.read_bytes())
        source_metadata = package_init.stat()
        cached[8:12] = int(source_metadata.st_mtime).to_bytes(4, "little")
        cached[12:16] = source_metadata.st_size.to_bytes(4, "little")
        cache_path.write_bytes(cached)
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "exact clean source",
        ):
            runtime._verify_exact_source(
                source_root=self.source,
                expected_git_sha=expected_git_sha,
                expected_manifest_sha256=expected_manifest,
                git_executable=git,
                git_executable_sha256=hashlib.sha256(git.read_bytes()).hexdigest(),
            )
        cache_path.unlink()
        cache_path.parent.rmdir()

        (self.source / "python" / "evolution_sim" / "__init__.py").write_text(
            "tampered = True\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            runtime.RuntimeVenvAuthorityError,
            "exact clean source",
        ):
            runtime._verify_exact_source(
                source_root=self.source,
                expected_git_sha=expected_git_sha,
                expected_manifest_sha256=expected_manifest,
                git_executable=git,
                git_executable_sha256=hashlib.sha256(git.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
