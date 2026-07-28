"""Stdlib-only authority and bootstrap for the remote CUDA/Torch virtualenv.

The archive authority intentionally pins the canonical base interpreter.  A
virtualenv launcher must retain its lexical invocation path for Python to apply
``pyvenv.cfg`` and site-packages.  This module separately seals that launcher,
its symlink chain, base interpreter bytes, ``pyvenv.cfg``, and a complete
site-packages content manifest.  The canonical base interpreter executes this
file, verifies the sealed authority, and only then ``execve`` replaces itself
with the lexical virtualenv launcher.

Keep this file importable with the Python standard library alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import subprocess
import sys
from typing import Final, Mapping, Sequence


RUNTIME_VENV_AUTHORITY_SCHEMA_VERSION: Final = "open_ecology_runtime_venv_authority_v1"
RUNTIME_GUARDIAN_MODULE: Final = "evolution_sim.cli.open_ecology_phase_a_guardian"
RUNTIME_GUARDIAN_ENTRYPOINT_RELATIVE_PATH: Final = (
    "python/evolution_sim/cli/open_ecology_phase_a_guardian.py"
)
RUNTIME_PACKAGE_INIT_RELATIVE_PATH: Final = "python/evolution_sim/__init__.py"
RUNTIME_EXECUTION_CONTRACT: Final = (
    "isolated_reexec_then_deterministic_safe_no_site_stdlib_site_source_v4"
)
MAX_RUNTIME_AUTHORITY_BYTES: Final = 1024 * 1024
MAX_SITE_PACKAGES_ENTRIES: Final = 2_000_000
MAX_SITE_PACKAGES_BYTES: Final = 64 * 1024**3
MAX_PYVENV_CFG_BYTES: Final = 64 * 1024
_READ_SIZE: Final = 1024 * 1024
_ISOLATED_RUNTIME_REEXEC_CODE: Final = r"""
import os
import sys

if sys.flags.isolated != 1 or sys.flags.no_site != 1:
    raise SystemExit("runtime isolated relay flags drifted")
invocation, guardian_code = sys.argv[1:3]
os.execve(
    invocation,
    (invocation, "-P", "-S", "-c", guardian_code, *sys.argv[3:]),
    dict(os.environ),
)
""".strip()
_RUNTIME_GUARDIAN_EXEC_CODE: Final = r"""
import hashlib
import importlib.util
import os
import stat
import sys

(
    package_path,
    package_sha,
    entry_path,
    entry_sha,
    site_packages,
    source_python,
    venv_root,
) = sys.argv[1:8]

def read_pinned(path, expected):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit("runtime source entry is not regular")
        digest = hashlib.sha256()
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if identity(before) != identity(after) or digest.hexdigest() != expected:
        raise SystemExit("runtime source entry identity or SHA256 drifted")
    return b"".join(chunks)

if not os.path.isdir(site_packages) or not os.path.isdir(source_python):
    raise SystemExit("sealed runtime import roots are unavailable")
if site_packages in sys.path or source_python in sys.path:
    raise SystemExit("sealed runtime import roots were injected before bootstrap")
sys.path.extend((site_packages, source_python))
# Python <=3.13 applies pyvenv.cfg in site initialization.  This no-site
# runtime restores only the already validated lexical venv identity.
sys.prefix = venv_root
sys.exec_prefix = venv_root

spec = importlib.util.find_spec("evolution_sim")
if spec is None or spec.origin is None:
    raise SystemExit("sealed evolution_sim package is unavailable")
if os.path.realpath(spec.origin) != package_path:
    raise SystemExit("evolution_sim import origin escaped sealed source")
read_pinned(package_path, package_sha)
entry_source = read_pinned(entry_path, entry_sha)
sys.argv = [entry_path, *sys.argv[8:]]
namespace = {
    "__builtins__": __builtins__,
    "__file__": entry_path,
    "__name__": "__main__",
    "__package__": "evolution_sim.cli",
    "__spec__": None,
}
exec(compile(entry_source, entry_path, "exec"), namespace, namespace)
""".strip()


class RuntimeVenvAuthorityError(RuntimeError):
    """The virtualenv launcher or its complete content authority drifted."""


def build_runtime_venv_authority(
    *,
    venv_python: Path,
    source_root: Path,
    source_git_sha: str,
    source_manifest_sha256: str,
    archive_authority_sha256: str,
    ssh_target: str,
    ssh_connection_sha256: str,
) -> dict[str, object]:
    invocation = _lexical_absolute(venv_python, field="virtualenv Python")
    source = _canonical_directory(source_root, field="source root")
    chain, resolved = _launcher_chain(invocation)
    venv_root = invocation.parent.parent
    pyvenv_cfg = _canonical_regular_file(
        venv_root / "pyvenv.cfg",
        field="pyvenv.cfg",
    )
    _require_isolated_pyvenv_cfg(pyvenv_cfg)
    site_packages = _site_packages_directory(venv_root)
    site_manifest = _tree_manifest(site_packages)
    authority: dict[str, object] = {
        "schema_version": RUNTIME_VENV_AUTHORITY_SCHEMA_VERSION,
        "archive_authority": {
            "sha256": _sha256(
                archive_authority_sha256,
                field="archive authority SHA256",
            ),
            "ssh_connection_sha256": _sha256(
                ssh_connection_sha256,
                field="SSH connection SHA256",
            ),
            "ssh_target": _ssh_target(ssh_target),
        },
        "source": {
            "git_sha": _git_sha(source_git_sha),
            "manifest_sha256": _sha256(
                source_manifest_sha256,
                field="source manifest SHA256",
            ),
            "root": str(source),
        },
        "launcher": {
            "invocation_path": str(invocation),
            "chain": chain,
            "resolved_interpreter": _file_pin(resolved),
        },
        "virtualenv": {
            "root": str(venv_root),
            "pyvenv_cfg": _file_pin(pyvenv_cfg),
            "site_packages": {
                "path": str(site_packages),
                **site_manifest,
            },
        },
        "execution": {
            "entrypoint": _source_file_pin(
                source,
                RUNTIME_GUARDIAN_ENTRYPOINT_RELATIVE_PATH,
            ),
            "package_init": _source_file_pin(
                source,
                RUNTIME_PACKAGE_INIT_RELATIVE_PATH,
            ),
            "module": RUNTIME_GUARDIAN_MODULE,
            "environment_contract": RUNTIME_EXECUTION_CONTRACT,
        },
    }
    authority["exact_digest"] = _stable_digest(authority)
    validate_runtime_venv_authority(authority)
    return authority


def validate_runtime_venv_authority(
    authority: Mapping[str, object],
    *,
    expected_authority_sha256: str | None = None,
    expected_source_git_sha: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_archive_authority_sha256: str | None = None,
    expected_ssh_target: str | None = None,
    expected_ssh_connection_sha256: str | None = None,
) -> None:
    _exact_keys(
        authority,
        {
            "schema_version",
            "archive_authority",
            "source",
            "launcher",
            "virtualenv",
            "execution",
            "exact_digest",
        },
        field="runtime venv authority",
    )
    if authority.get("schema_version") != RUNTIME_VENV_AUTHORITY_SCHEMA_VERSION:
        raise RuntimeVenvAuthorityError("runtime venv authority schema drifted")
    observed_digest = _sha256(
        authority.get("exact_digest"),
        field="runtime venv exact digest",
    )
    unsigned = dict(authority)
    unsigned.pop("exact_digest")
    if _stable_digest(unsigned) != observed_digest:
        raise RuntimeVenvAuthorityError("runtime venv authority self-digest mismatched")
    if expected_authority_sha256 is not None:
        _sha256(expected_authority_sha256, field="expected runtime authority SHA256")

    archive = _mapping(authority.get("archive_authority"), field="archive authority")
    _exact_keys(
        archive,
        {"sha256", "ssh_connection_sha256", "ssh_target"},
        field="archive authority",
    )
    archive_sha = _sha256(archive.get("sha256"), field="archive authority SHA256")
    ssh_connection_sha = _sha256(
        archive.get("ssh_connection_sha256"),
        field="SSH connection SHA256",
    )
    target = _ssh_target(archive.get("ssh_target"))
    if (
        expected_archive_authority_sha256 is not None
        and archive_sha != expected_archive_authority_sha256
    ):
        raise RuntimeVenvAuthorityError("archive authority binding drifted")
    if expected_ssh_target is not None and target != expected_ssh_target:
        raise RuntimeVenvAuthorityError("SSH target binding drifted")
    if (
        expected_ssh_connection_sha256 is not None
        and ssh_connection_sha != expected_ssh_connection_sha256
    ):
        raise RuntimeVenvAuthorityError("SSH connection binding drifted")

    source = _mapping(authority.get("source"), field="source")
    _exact_keys(
        source,
        {"git_sha", "manifest_sha256", "root"},
        field="source",
    )
    source_sha = _git_sha(source.get("git_sha"))
    source_manifest = _sha256(
        source.get("manifest_sha256"),
        field="source manifest SHA256",
    )
    source_root = _canonical_directory(Path(_text(source.get("root"))), field="source")
    if expected_source_git_sha is not None and source_sha != expected_source_git_sha:
        raise RuntimeVenvAuthorityError("source Git SHA binding drifted")
    if (
        expected_source_manifest_sha256 is not None
        and source_manifest != expected_source_manifest_sha256
    ):
        raise RuntimeVenvAuthorityError("source manifest binding drifted")

    launcher = _mapping(authority.get("launcher"), field="launcher")
    _exact_keys(
        launcher,
        {"invocation_path", "chain", "resolved_interpreter"},
        field="launcher",
    )
    invocation = _lexical_absolute(
        Path(_text(launcher.get("invocation_path"))),
        field="launcher invocation",
    )
    observed_chain, resolved = _launcher_chain(invocation)
    if launcher.get("chain") != observed_chain:
        raise RuntimeVenvAuthorityError("virtualenv launcher chain drifted")
    _validate_file_pin(
        launcher.get("resolved_interpreter"),
        expected=resolved,
        field="resolved interpreter",
    )

    virtualenv = _mapping(authority.get("virtualenv"), field="virtualenv")
    _exact_keys(
        virtualenv,
        {"root", "pyvenv_cfg", "site_packages"},
        field="virtualenv",
    )
    venv_root = _lexical_absolute(
        Path(_text(virtualenv.get("root"))),
        field="virtualenv root",
    )
    if venv_root != invocation.parent.parent:
        raise RuntimeVenvAuthorityError("virtualenv root is detached from its launcher")
    pyvenv_cfg = _canonical_regular_file(
        venv_root / "pyvenv.cfg",
        field="pyvenv.cfg",
    )
    _validate_file_pin(
        virtualenv.get("pyvenv_cfg"),
        expected=pyvenv_cfg,
        field="pyvenv.cfg",
    )
    _require_isolated_pyvenv_cfg(pyvenv_cfg)
    site = _mapping(virtualenv.get("site_packages"), field="site-packages")
    _exact_keys(
        site,
        {"path", "entry_count", "total_file_bytes", "aggregate_sha256"},
        field="site-packages",
    )
    site_path = _canonical_directory(
        Path(_text(site.get("path"))),
        field="site-packages",
    )
    if site_path != _site_packages_directory(venv_root):
        raise RuntimeVenvAuthorityError("site-packages path drifted")
    observed_manifest = _tree_manifest(site_path)
    expected_manifest = {
        "entry_count": _nonnegative_int(
            site.get("entry_count"),
            field="site-packages entry count",
        ),
        "total_file_bytes": _nonnegative_int(
            site.get("total_file_bytes"),
            field="site-packages total bytes",
        ),
        "aggregate_sha256": _sha256(
            site.get("aggregate_sha256"),
            field="site-packages aggregate SHA256",
        ),
    }
    if observed_manifest != expected_manifest:
        raise RuntimeVenvAuthorityError("site-packages content manifest drifted")

    execution = _mapping(authority.get("execution"), field="execution")
    _exact_keys(
        execution,
        {"entrypoint", "package_init", "module", "environment_contract"},
        field="runtime execution",
    )
    if (
        execution.get("module") != RUNTIME_GUARDIAN_MODULE
        or execution.get("environment_contract") != RUNTIME_EXECUTION_CONTRACT
    ):
        raise RuntimeVenvAuthorityError("runtime execution contract drifted")
    _validate_source_file_pin(
        execution.get("entrypoint"),
        source_root=source_root,
        expected_relative_path=RUNTIME_GUARDIAN_ENTRYPOINT_RELATIVE_PATH,
        field="runtime guardian entrypoint",
    )
    _validate_source_file_pin(
        execution.get("package_init"),
        source_root=source_root,
        expected_relative_path=RUNTIME_PACKAGE_INIT_RELATIVE_PATH,
        field="runtime package init",
    )
    if not (source_root / "python" / "evolution_sim").is_dir():
        raise RuntimeVenvAuthorityError("source root lacks the evolution_sim package")


def exec_guardian_from_runtime_authority(
    *,
    authority_path: Path,
    expected_authority_sha256: str,
    expected_source_git_sha: str,
    expected_source_manifest_sha256: str,
    expected_archive_authority_sha256: str,
    expected_ssh_target: str,
    expected_ssh_connection_sha256: str,
    git_executable: Path,
    git_executable_sha256: str,
    guardian_arguments: Sequence[str],
) -> None:
    authority = load_runtime_venv_authority(
        authority_path,
        expected_sha256=expected_authority_sha256,
    )
    validate_runtime_venv_authority(
        authority,
        expected_authority_sha256=expected_authority_sha256,
        expected_source_git_sha=expected_source_git_sha,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_archive_authority_sha256=expected_archive_authority_sha256,
        expected_ssh_target=expected_ssh_target,
        expected_ssh_connection_sha256=expected_ssh_connection_sha256,
    )
    source = _mapping(authority["source"], field="source")
    source_root = Path(_text(source["root"]))
    _verify_exact_source(
        source_root=source_root,
        expected_git_sha=expected_source_git_sha,
        expected_manifest_sha256=expected_source_manifest_sha256,
        git_executable=git_executable,
        git_executable_sha256=git_executable_sha256,
    )
    launcher = _mapping(authority["launcher"], field="launcher")
    invocation = _text(launcher["invocation_path"])
    execution = _mapping(authority["execution"], field="execution")
    package_init = _mapping(
        execution["package_init"],
        field="runtime package init",
    )
    entrypoint = _mapping(
        execution["entrypoint"],
        field="runtime guardian entrypoint",
    )
    source_python = source_root / "python"
    virtualenv = _mapping(authority["virtualenv"], field="virtualenv")
    site_packages = Path(
        _text(
            _mapping(
                virtualenv["site_packages"],
                field="site-packages",
            )["path"]
        )
    )
    argv = (
        invocation,
        "-I",
        "-S",
        "-c",
        _ISOLATED_RUNTIME_REEXEC_CODE,
        invocation,
        _RUNTIME_GUARDIAN_EXEC_CODE,
        str(source_root / _text(package_init["relative_path"])),
        _sha256(
            package_init["sha256"],
            field="runtime package init SHA256",
        ),
        str(source_root / _text(entrypoint["relative_path"])),
        _sha256(
            entrypoint["sha256"],
            field="runtime guardian entrypoint SHA256",
        ),
        str(site_packages),
        str(source_python),
        str(Path(_text(virtualenv["root"]))),
        *guardian_arguments,
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
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        value = os.environ.get(key)
        if value is not None:
            environment[key] = value
    os.chdir(source_root)
    os.execve(invocation, argv, environment)


def revalidate_running_guardian_runtime(
    *,
    authority_path: Path,
    expected_authority_sha256: str,
    expected_source_git_sha: str,
    expected_source_manifest_sha256: str,
    expected_archive_authority_sha256: str,
    expected_ssh_target: str,
    expected_ssh_connection_sha256: str,
    git_executable: Path,
    git_executable_sha256: str,
    observed_entrypoint_path: Path,
) -> dict[str, object]:
    """Reopen the complete runtime authority inside the launched interpreter."""

    authority = load_runtime_venv_authority(
        authority_path,
        expected_sha256=expected_authority_sha256,
    )
    validate_runtime_venv_authority(
        authority,
        expected_authority_sha256=expected_authority_sha256,
        expected_source_git_sha=expected_source_git_sha,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_archive_authority_sha256=expected_archive_authority_sha256,
        expected_ssh_target=expected_ssh_target,
        expected_ssh_connection_sha256=expected_ssh_connection_sha256,
    )
    source = _mapping(authority["source"], field="source")
    source_root = Path(_text(source["root"]))
    _verify_exact_source(
        source_root=source_root,
        expected_git_sha=expected_source_git_sha,
        expected_manifest_sha256=expected_source_manifest_sha256,
        git_executable=git_executable,
        git_executable_sha256=git_executable_sha256,
    )
    launcher = _mapping(authority["launcher"], field="launcher")
    invocation = Path(_text(launcher["invocation_path"]))
    virtualenv = _mapping(authority["virtualenv"], field="virtualenv")
    venv_root = Path(_text(virtualenv["root"]))
    site_packages = Path(
        _text(
            _mapping(
                virtualenv["site_packages"],
                field="site-packages",
            )["path"]
        )
    )
    execution = _mapping(authority["execution"], field="execution")
    entrypoint = _validate_source_file_pin(
        execution["entrypoint"],
        source_root=source_root,
        expected_relative_path=RUNTIME_GUARDIAN_ENTRYPOINT_RELATIVE_PATH,
        field="runtime guardian entrypoint",
    )
    package_init = _validate_source_file_pin(
        execution["package_init"],
        source_root=source_root,
        expected_relative_path=RUNTIME_PACKAGE_INIT_RELATIVE_PATH,
        field="runtime package init",
    )
    imported_package = sys.modules.get("evolution_sim")
    imported_package_path = getattr(imported_package, "__file__", None)
    required_environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    source_python = str(source_root / "python")
    sealed_site_packages = str(site_packages)
    if (
        Path(sys.executable) != invocation
        or Path(sys.prefix) != venv_root
        or Path(sys.exec_prefix) != venv_root
        or sys.flags.isolated != 0
        or sys.flags.ignore_environment != 0
        or sys.flags.no_site != 1
        or not bool(getattr(sys.flags, "safe_path", False))
        or not sys.dont_write_bytecode
        or Path.cwd().resolve(strict=True) != source_root
        or "PYTHONPATH" in os.environ
        or any(
            os.environ.get(key) != value for key, value in required_environment.items()
        )
        or imported_package_path is None
        or Path(imported_package_path).resolve(strict=True) != package_init
        or observed_entrypoint_path.resolve(strict=True) != entrypoint
        or sys.path[-2:] != [sealed_site_packages, source_python]
    ):
        raise RuntimeVenvAuthorityError(
            "running guardian interpreter escaped its sealed runtime contract"
        )
    return authority


def load_runtime_venv_authority(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_git_sha: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_archive_authority_sha256: str | None = None,
    expected_ssh_target: str | None = None,
    expected_ssh_connection_sha256: str | None = None,
) -> dict[str, object]:
    expected = _sha256(expected_sha256, field="runtime authority file SHA256")
    canonical = _canonical_regular_file(path, field="runtime authority file")
    metadata_before = canonical.stat()
    if metadata_before.st_size > MAX_RUNTIME_AUTHORITY_BYTES:
        raise RuntimeVenvAuthorityError("runtime authority file is oversized")
    payload = canonical.read_bytes()
    metadata_after = canonical.stat()
    if _metadata_identity(metadata_before) != _metadata_identity(metadata_after):
        raise RuntimeVenvAuthorityError("runtime authority file changed while read")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeVenvAuthorityError("runtime authority file SHA256 mismatched")
    authority = _strict_json_mapping(payload)
    validate_runtime_venv_authority(
        authority,
        expected_authority_sha256=expected,
        expected_source_git_sha=expected_source_git_sha,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_archive_authority_sha256=(expected_archive_authority_sha256),
        expected_ssh_target=expected_ssh_target,
        expected_ssh_connection_sha256=expected_ssh_connection_sha256,
    )
    return authority


def write_new_runtime_venv_authority(
    path: Path,
    authority: Mapping[str, object],
) -> str:
    validate_runtime_venv_authority(authority)
    payload = _canonical_json(authority) + b"\n"
    parent = path.parent.resolve(strict=True)
    if path.is_symlink() or path.exists():
        raise RuntimeVenvAuthorityError("runtime authority output must be new")
    temporary = parent / f".{path.name}.pending-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeVenvAuthorityError(
            "runtime authority output appeared during publication"
        ) from error
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeVenvAuthorityError(
            "runtime authority no-replace publication failed"
        ) from error
    temporary.unlink()
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(payload).hexdigest()


def _verify_exact_source(
    *,
    source_root: Path,
    expected_git_sha: str,
    expected_manifest_sha256: str,
    git_executable: Path,
    git_executable_sha256: str,
) -> None:
    root = _canonical_directory(source_root, field="source root")
    git = _canonical_regular_file(git_executable, field="pinned Git")
    expected_git = _git_sha(expected_git_sha)
    expected_manifest = _sha256(
        expected_manifest_sha256,
        field="expected source manifest SHA256",
    )
    expected_git_bytes = _sha256(
        git_executable_sha256,
        field="pinned Git SHA256",
    )
    if _sha256_file(git) != expected_git_bytes:
        raise RuntimeVenvAuthorityError("pinned Git bytes drifted")
    before_manifest, before_inventory = _source_manifest_state(root)
    head = _run_pinned_git(git, root, ("rev-parse", "HEAD"))
    status = _run_pinned_git(
        git,
        root,
        ("status", "--porcelain=v1", "--untracked-files=normal"),
    )
    tracked_inventory = frozenset(
        _run_pinned_git(
            git,
            root,
            ("ls-files", "--", "python/evolution_sim"),
        ).splitlines()
    )
    after_manifest, after_inventory = _source_manifest_state(root)
    if _sha256_file(git) != expected_git_bytes:
        raise RuntimeVenvAuthorityError("pinned Git changed during source check")
    if (
        head != expected_git
        or status
        or before_inventory != tracked_inventory
        or after_inventory != tracked_inventory
        or before_manifest != expected_manifest
        or after_manifest != expected_manifest
    ):
        raise RuntimeVenvAuthorityError(
            "runtime bootstrap requires the exact clean source checkout"
        )


def _run_pinned_git(
    executable: Path,
    source_root: Path,
    arguments: Sequence[str],
) -> str:
    try:
        completed = subprocess.run(
            (
                str(executable),
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ),
            cwd=source_root,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeVenvAuthorityError("pinned Git source check failed") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > MAX_RUNTIME_AUTHORITY_BYTES
    ):
        raise RuntimeVenvAuthorityError("pinned Git source check failed")
    try:
        return completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeVenvAuthorityError(
            "pinned Git source output was not UTF-8"
        ) from error


def _source_manifest_aggregate(root: Path) -> str:
    return _source_manifest_state(root)[0]


def _source_manifest_state(root: Path) -> tuple[str, frozenset[str]]:
    package_root = root / "python" / "evolution_sim"
    package = _canonical_directory(package_root, field="source package")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    files: dict[str, str] = {}
    inventory: set[str] = set()

    def hash_fd(descriptor: int) -> str:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while block := os.read(descriptor, _READ_SIZE):
            digest.update(block)
        after = os.fstat(descriptor)
        if _metadata_identity(before) != _metadata_identity(after):
            raise RuntimeVenvAuthorityError(
                "source file changed while descriptor-hashed"
            )
        return digest.hexdigest()

    def visit(directory_fd: int, relative: PurePosixPath) -> None:
        before = os.fstat(directory_fd)
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as error:
            raise RuntimeVenvAuthorityError(
                "source package could not be scanned"
            ) from error
        for name in names:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            child_relative = relative / name
            if stat.S_ISDIR(metadata.st_mode):
                child_fd = os.open(
                    name,
                    directory_flags,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if _opened_identity(opened) != _opened_identity(metadata):
                        raise RuntimeVenvAuthorityError(
                            "source directory identity changed"
                        )
                    visit(child_fd, child_relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                inventory.add(
                    (PurePosixPath("python/evolution_sim") / child_relative).as_posix()
                )
                if not name.endswith(".py"):
                    continue
                file_fd = os.open(name, file_flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(file_fd)
                    if _opened_identity(opened) != _opened_identity(metadata):
                        raise RuntimeVenvAuthorityError("source file identity changed")
                    files[
                        (
                            PurePosixPath("python/evolution_sim") / child_relative
                        ).as_posix()
                    ] = hash_fd(file_fd)
                finally:
                    os.close(file_fd)
            elif stat.S_ISLNK(metadata.st_mode):
                raise RuntimeVenvAuthorityError(
                    "source package contains a symbolic link"
                )
            else:
                raise RuntimeVenvAuthorityError(
                    "source package contains a non-regular entry"
                )
        after = os.fstat(directory_fd)
        if _metadata_identity(before) != _metadata_identity(after):
            raise RuntimeVenvAuthorityError("source directory changed while scanned")

    package_fd = os.open(package, directory_flags)
    try:
        visit(package_fd, PurePosixPath())
    finally:
        os.close(package_fd)
    for name in ("package.json", "requirements-mind-ml.txt"):
        candidate = root / name
        if not candidate.exists():
            continue
        canonical = _canonical_regular_file(candidate, field=f"source {name}")
        descriptor = os.open(canonical, file_flags)
        try:
            files[name] = hash_fd(descriptor)
        finally:
            os.close(descriptor)
    if not files:
        raise RuntimeVenvAuthorityError("source manifest found no files")
    return hashlib.sha256(_canonical_json(files)).hexdigest(), frozenset(inventory)


def _launcher_chain(invocation: Path) -> tuple[list[dict[str, object]], Path]:
    current = invocation
    visited: set[Path] = set()
    chain: list[dict[str, object]] = []
    for _ in range(16):
        if current in visited:
            raise RuntimeVenvAuthorityError("virtualenv launcher symlink looped")
        visited.add(current)
        try:
            metadata = current.lstat()
        except OSError as error:
            raise RuntimeVenvAuthorityError(
                "virtualenv launcher path is unavailable"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(current)
            if not target:
                raise RuntimeVenvAuthorityError(
                    "virtualenv launcher has an empty symlink target"
                )
            chain.append(
                {
                    "path": str(current),
                    "type": "symlink",
                    "target": target,
                }
            )
            raw_next = Path(target)
            current = (
                raw_next
                if raw_next.is_absolute()
                else _lexically_normal(current.parent / raw_next)
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeVenvAuthorityError(
                "virtualenv launcher did not resolve to a regular interpreter"
            )
        resolved = current.resolve(strict=True)
        chain.append(
            {
                "path": str(current),
                "type": "regular",
                "sha256": _sha256_file(resolved),
                "size": metadata.st_size,
            }
        )
        return chain, resolved
    raise RuntimeVenvAuthorityError("virtualenv launcher chain is too deep")


def _site_packages_directory(venv_root: Path) -> Path:
    lib = venv_root / "lib"
    if not lib.is_dir() or lib.is_symlink():
        raise RuntimeVenvAuthorityError("virtualenv lib directory is invalid")
    matches = sorted(
        candidate
        for candidate in lib.glob("python*/site-packages")
        if candidate.is_dir() and not candidate.is_symlink()
    )
    if len(matches) != 1:
        raise RuntimeVenvAuthorityError(
            "virtualenv must contain exactly one real site-packages directory"
        )
    return matches[0].resolve(strict=True)


def _tree_manifest(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    entry_count = 0
    total_file_bytes = 0
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)

    def visit(directory_fd: int, relative: PurePosixPath) -> None:
        nonlocal entry_count, total_file_bytes
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as error:
            raise RuntimeVenvAuthorityError("failed to scan site-packages") from error
        for name in names:
            if "/" in name or name in {".", ".."}:
                raise RuntimeVenvAuthorityError(
                    "site-packages contains a non-canonical entry"
                )
            child_relative = relative / name
            try:
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as error:
                raise RuntimeVenvAuthorityError(
                    "site-packages entry could not be inspected"
                ) from error
            entry_count += 1
            if entry_count > MAX_SITE_PACKAGES_ENTRIES:
                raise RuntimeVenvAuthorityError("site-packages entry ceiling exceeded")
            if stat.S_ISDIR(metadata.st_mode):
                record = {
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "path": child_relative.as_posix(),
                    "type": "directory",
                }
                digest.update(_canonical_json(record) + b"\n")
                try:
                    child_fd = os.open(
                        name,
                        directory_flags,
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise RuntimeVenvAuthorityError(
                        "site-packages directory identity changed"
                    ) from error
                try:
                    opened = os.fstat(child_fd)
                    if _opened_identity(opened) != _opened_identity(metadata):
                        raise RuntimeVenvAuthorityError(
                            "site-packages directory identity changed"
                        )
                    visit(child_fd, child_relative)
                    after = os.fstat(child_fd)
                    if _metadata_identity(after) != _metadata_identity(opened):
                        raise RuntimeVenvAuthorityError(
                            "site-packages directory changed while scanned"
                        )
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(metadata.st_mode):
                total_file_bytes += metadata.st_size
                if total_file_bytes > MAX_SITE_PACKAGES_BYTES:
                    raise RuntimeVenvAuthorityError(
                        "site-packages byte ceiling exceeded"
                    )
                try:
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                except OSError as error:
                    raise RuntimeVenvAuthorityError(
                        "site-packages file identity changed"
                    ) from error
                try:
                    opened = os.fstat(child_fd)
                    if _opened_identity(opened) != _opened_identity(metadata):
                        raise RuntimeVenvAuthorityError(
                            "site-packages file identity changed"
                        )
                    file_digest = hashlib.sha256()
                    while block := os.read(child_fd, _READ_SIZE):
                        file_digest.update(block)
                    after = os.fstat(child_fd)
                    if _metadata_identity(after) != _metadata_identity(opened):
                        raise RuntimeVenvAuthorityError(
                            "site-packages file changed while hashed"
                        )
                finally:
                    os.close(child_fd)
                record = {
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "path": child_relative.as_posix(),
                    "sha256": file_digest.hexdigest(),
                    "size": metadata.st_size,
                    "type": "file",
                }
                digest.update(_canonical_json(record) + b"\n")
            elif stat.S_ISLNK(metadata.st_mode):
                raise RuntimeVenvAuthorityError(
                    "site-packages symbolic links are forbidden"
                )
            else:
                raise RuntimeVenvAuthorityError(
                    "site-packages contains a special filesystem entry"
                )

    try:
        root_fd = os.open(root, directory_flags)
    except OSError as error:
        raise RuntimeVenvAuthorityError(
            "site-packages root could not be descriptor-opened"
        ) from error
    try:
        before = os.fstat(root_fd)
        visit(root_fd, PurePosixPath())
        after = os.fstat(root_fd)
        if _metadata_identity(before) != _metadata_identity(after):
            raise RuntimeVenvAuthorityError("site-packages root changed while scanned")
    finally:
        os.close(root_fd)
    return {
        "entry_count": entry_count,
        "total_file_bytes": total_file_bytes,
        "aggregate_sha256": digest.hexdigest(),
    }


def _opened_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _file_pin(path: Path) -> dict[str, object]:
    canonical = _canonical_regular_file(path, field="pinned file")
    metadata = canonical.stat()
    return {
        "path": str(canonical),
        "sha256": _sha256_file(canonical),
        "size": metadata.st_size,
    }


def _source_file_pin(
    source_root: Path,
    relative_path: str,
) -> dict[str, object]:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeVenvAuthorityError("runtime source pin path is invalid")
    candidate = source_root / Path(relative)
    canonical = _canonical_regular_file(
        candidate,
        field="runtime source pin",
    )
    if canonical != candidate:
        raise RuntimeVenvAuthorityError("runtime source pin escaped source root")
    metadata = canonical.stat()
    return {
        "relative_path": relative.as_posix(),
        "sha256": _sha256_file(canonical),
        "size": metadata.st_size,
    }


def _validate_source_file_pin(
    value: object,
    *,
    source_root: Path,
    expected_relative_path: str,
    field: str,
) -> Path:
    pin = _mapping(value, field=field)
    _exact_keys(pin, {"relative_path", "sha256", "size"}, field=field)
    if pin.get("relative_path") != expected_relative_path:
        raise RuntimeVenvAuthorityError(f"{field} relative path drifted")
    candidate = source_root / Path(PurePosixPath(expected_relative_path))
    canonical = _canonical_regular_file(candidate, field=field)
    if canonical != candidate:
        raise RuntimeVenvAuthorityError(f"{field} escaped source root")
    metadata_before = canonical.stat()
    if _nonnegative_int(
        pin.get("size"), field=f"{field} size"
    ) != metadata_before.st_size or _sha256(
        pin.get("sha256"), field=f"{field} SHA256"
    ) != _sha256_file(canonical):
        raise RuntimeVenvAuthorityError(f"{field} bytes drifted")
    metadata_after = canonical.stat()
    if _metadata_identity(metadata_before) != _metadata_identity(metadata_after):
        raise RuntimeVenvAuthorityError(f"{field} changed while verified")
    return canonical


def _require_isolated_pyvenv_cfg(path: Path) -> None:
    payload = _read_regular_file_bytes(
        path,
        byte_ceiling=MAX_PYVENV_CFG_BYTES,
        field="pyvenv.cfg",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeVenvAuthorityError("pyvenv.cfg is not UTF-8") from error
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        normalized_key = key.strip().casefold()
        if not separator or not normalized_key or normalized_key in values:
            raise RuntimeVenvAuthorityError("pyvenv.cfg is ambiguous")
        values[normalized_key] = value.strip()
    if values.get("include-system-site-packages", "").casefold() != "false":
        raise RuntimeVenvAuthorityError("pyvenv.cfg must disable system site-packages")


def _read_regular_file_bytes(
    path: Path,
    *,
    byte_ceiling: int,
    field: str,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeVenvAuthorityError(f"{field} could not be opened") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeVenvAuthorityError(f"{field} is not regular")
        payload = bytearray()
        while block := os.read(
            descriptor,
            min(_READ_SIZE, byte_ceiling + 1 - len(payload)),
        ):
            payload.extend(block)
            if len(payload) > byte_ceiling:
                raise RuntimeVenvAuthorityError(f"{field} is oversized")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _metadata_identity(before) != _metadata_identity(after):
        raise RuntimeVenvAuthorityError(f"{field} changed while read")
    return bytes(payload)


def _validate_file_pin(value: object, *, expected: Path, field: str) -> None:
    pin = _mapping(value, field=field)
    _exact_keys(pin, {"path", "sha256", "size"}, field=field)
    canonical = _canonical_regular_file(
        Path(_text(pin.get("path"))),
        field=field,
    )
    if canonical != expected.resolve(strict=True):
        raise RuntimeVenvAuthorityError(f"{field} path drifted")
    metadata_before = canonical.stat()
    if _nonnegative_int(
        pin.get("size"), field=f"{field} size"
    ) != metadata_before.st_size or _sha256(
        pin.get("sha256"), field=f"{field} SHA256"
    ) != _sha256_file(canonical):
        raise RuntimeVenvAuthorityError(f"{field} bytes drifted")
    metadata_after = canonical.stat()
    if _metadata_identity(metadata_before) != _metadata_identity(metadata_after):
        raise RuntimeVenvAuthorityError(f"{field} changed while verified")


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(_READ_SIZE):
                digest.update(block)
    except OSError as error:
        raise RuntimeVenvAuthorityError("failed to hash pinned file") from error
    return digest.hexdigest()


def _canonical_directory(path: Path, *, field: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeVenvAuthorityError(f"{field} must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeVenvAuthorityError(f"{field} is unavailable") from error
    if resolved != path or not resolved.is_dir():
        raise RuntimeVenvAuthorityError(f"{field} must be one canonical directory")
    return resolved


def _canonical_regular_file(path: Path, *, field: str) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeVenvAuthorityError(f"{field} must be absolute and non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeVenvAuthorityError(f"{field} is unavailable") from error
    if resolved != path or not resolved.is_file():
        raise RuntimeVenvAuthorityError(f"{field} must be one canonical regular file")
    return resolved


def _lexical_absolute(path: Path, *, field: str) -> Path:
    if not path.is_absolute() or str(path) != os.path.normpath(str(path)):
        raise RuntimeVenvAuthorityError(f"{field} must be one lexical absolute path")
    return path


def _lexically_normal(path: Path) -> Path:
    normalized = Path(os.path.normpath(str(path)))
    if not normalized.is_absolute():
        raise RuntimeVenvAuthorityError("launcher symlink resolved non-absolutely")
    return normalized


def _stable_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeVenvAuthorityError(
            "runtime authority is not canonical JSON"
        ) from error


def _strict_json_mapping(payload: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeVenvAuthorityError(
                    "runtime authority contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: _reject_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeVenvAuthorityError(
            "runtime authority is not strict JSON"
        ) from error
    return dict(_mapping(parsed, field="runtime authority"))


def _reject_constant(value: str) -> object:
    raise RuntimeVenvAuthorityError(
        f"runtime authority JSON constant {value!r} is forbidden"
    )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeVenvAuthorityError(f"{field} is not a string-key mapping")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(value) != expected:
        raise RuntimeVenvAuthorityError(f"{field} fields differ")


def _text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeVenvAuthorityError("runtime authority text is malformed")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeVenvAuthorityError(f"{field} is not a lowercase SHA256")
    return value


def _git_sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeVenvAuthorityError("source Git SHA is malformed")
    return value


def _ssh_target(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise RuntimeVenvAuthorityError("SSH target is malformed")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeVenvAuthorityError(f"{field} must be a non-negative integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--venv-python", type=Path, required=True)
    seal.add_argument("--source-root", type=Path, required=True)
    seal.add_argument("--source-git-sha", required=True)
    seal.add_argument("--source-manifest-sha256", required=True)
    seal.add_argument("--archive-authority-sha256", required=True)
    seal.add_argument("--ssh-target", required=True)
    seal.add_argument("--ssh-connection-sha256", required=True)
    seal.add_argument("--output", type=Path, required=True)

    execute = subparsers.add_parser("exec")
    execute.add_argument("--authority", type=Path, required=True)
    execute.add_argument("--expected-authority-sha256", required=True)
    execute.add_argument("--source-git-sha", required=True)
    execute.add_argument("--source-manifest-sha256", required=True)
    execute.add_argument("--archive-authority-sha256", required=True)
    execute.add_argument("--ssh-target", required=True)
    execute.add_argument("--ssh-connection-sha256", required=True)
    execute.add_argument("--git-executable", type=Path, required=True)
    execute.add_argument("--git-executable-sha256", required=True)
    execute.add_argument("guardian_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "seal":
        authority = build_runtime_venv_authority(
            venv_python=args.venv_python,
            source_root=args.source_root,
            source_git_sha=args.source_git_sha,
            source_manifest_sha256=args.source_manifest_sha256,
            archive_authority_sha256=args.archive_authority_sha256,
            ssh_target=args.ssh_target,
            ssh_connection_sha256=args.ssh_connection_sha256,
        )
        digest = write_new_runtime_venv_authority(args.output, authority)
        print(
            json.dumps(
                {
                    "authority_file_sha256": digest,
                    "exact_digest": authority["exact_digest"],
                    "output": str(args.output),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "exec":
        guardian_arguments = list(args.guardian_arguments)
        if guardian_arguments and guardian_arguments[0] == "--":
            guardian_arguments.pop(0)
        if not guardian_arguments or guardian_arguments[0] != "serve":
            raise RuntimeVenvAuthorityError(
                "runtime bootstrap may execute only the guardian serve command"
            )
        exec_guardian_from_runtime_authority(
            authority_path=args.authority,
            expected_authority_sha256=args.expected_authority_sha256,
            expected_source_git_sha=args.source_git_sha,
            expected_source_manifest_sha256=args.source_manifest_sha256,
            expected_archive_authority_sha256=args.archive_authority_sha256,
            expected_ssh_target=args.ssh_target,
            expected_ssh_connection_sha256=args.ssh_connection_sha256,
            git_executable=args.git_executable,
            git_executable_sha256=args.git_executable_sha256,
            guardian_arguments=guardian_arguments,
        )
        raise AssertionError("execve unexpectedly returned")
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
