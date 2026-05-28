#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_HOST = os.environ.get("TRAINER_HOST")
DEFAULT_REPO = os.environ.get("TRAINER_REPO")
DEFAULT_WHEELHOUSE = os.environ.get("TRAINER_WHEELHOUSE")
DEFAULT_SESSION = os.environ.get("TRAINER_SESSION", "train")


def _require_config(value: str | None, *, option: str, env: str) -> str:
    if value:
        return value
    raise SystemExit(f"missing {option}; pass {option} or set {env}")


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    if argv[:1] == ["ssh"] and len(argv) >= 3:
        printable = f"ssh {argv[-2]} <remote-command>"
    elif argv[:1] == ["rsync"] and len(argv) >= 3:
        printable = f"rsync {argv[-2]} {argv[-1]}"
    else:
        printable = shlex.join(argv)
    print(f"+ {printable}", flush=True)
    return subprocess.run(argv, check=check, text=True)


def _ssh(host: str, command: str, *, tty: bool = False) -> list[str]:
    argv = ["ssh"]
    if tty:
        argv.append("-t")
    argv.extend([host, shlex.join(["bash", "-lc", command])])
    return argv


def _repo_command(repo: str, command: str, *, activate: bool = True) -> str:
    parts = [
        "set -euo pipefail",
        f"cd {shlex.quote(repo)}",
        "export PYTHONHASHSEED=0",
        "export PYTHONPATH=python",
    ]
    if activate:
        parts.append(f". {shlex.quote(repo)}/.venv/bin/activate")
    parts.append(command)
    return "; ".join(parts)


def _command_from_remainder(command: list[str]) -> str:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing remote command")
    return shlex.join(command)


def cmd_status(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    repo = _require_config(args.repo, option="--repo", env="TRAINER_REPO")
    command = _repo_command(
        repo,
        "\n".join(
            [
                'echo "host=$(hostname)"',
                'echo "repo=$(pwd)"',
                'echo "branch=$(git branch --show-current)"',
                'echo "head=$(git rev-parse --short HEAD)"',
                'echo "upstream=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"',
                'git status --short',
                'echo "--- gpu ---"',
                'nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu --format=csv,noheader',
                'echo "--- python ---"',
                'python --version',
                'python - <<\'PY\'\nimport torch\nprint("torch", torch.__version__)\nprint("cuda_available", torch.cuda.is_available())\nprint("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")\nPY',
            ]
        ),
    )
    _run(_ssh(host, command))


def cmd_pull(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    repo = _require_config(args.repo, option="--repo", env="TRAINER_REPO")
    command = _repo_command(
        repo,
        "\n".join(
            [
                "git fetch --prune origin",
                "git pull --ff-only",
                "git status --short",
                'echo "head=$(git rev-parse --short HEAD)"',
            ]
        ),
        activate=False,
    )
    _run(_ssh(host, command))


def cmd_deps(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    repo = _require_config(args.repo, option="--repo", env="TRAINER_REPO")
    pip_install = "python -m pip install -r requirements-mind-ml.txt"
    if args.wheelhouse:
        wheelhouse = shlex.quote(args.wheelhouse)
        pip_install = (
            f"if [ -d {wheelhouse} ]; then "
            f"python -m pip install --no-index --find-links={wheelhouse} "
            "-r requirements-mind-ml.txt; "
            "else "
            "python -m pip install -r requirements-mind-ml.txt; "
            "fi"
        )
    command = _repo_command(
        repo,
        "\n".join(
            [
                "npm install",
                pip_install,
                'echo "dependencies_ok=yes"',
            ]
        ),
    )
    _run(_ssh(host, command))


def cmd_run(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    repo = _require_config(args.repo, option="--repo", env="TRAINER_REPO")
    remote_command = _command_from_remainder(args.command)
    _run(_ssh(host, _repo_command(repo, remote_command)))


def cmd_start(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    repo = _require_config(args.repo, option="--repo", env="TRAINER_REPO")
    # argparse.REMAINDER captures options after the session name, so accept the
    # documented `start SESSION --pull -- command` shape here as well.
    command = list(args.command)
    while command:
        token = command[0]
        if token == "--":
            command = command[1:]
            break
        if token == "--pull":
            args.pull = True
            command = command[1:]
            continue
        if token == "--replace":
            args.replace = True
            command = command[1:]
            continue
        break
    args.command = command
    remote_command = _command_from_remainder(args.command)
    session = shlex.quote(args.session)
    if args.pull:
        cmd_pull(args)
    if args.replace:
        _run(_ssh(host, f"tmux kill-session -t {session} 2>/dev/null || true"))
    runner = (
        f"cd {shlex.quote(repo)}; "
        f". {shlex.quote(repo)}/.venv/bin/activate; "
        "export PYTHONHASHSEED=0 PYTHONPATH=python; "
        'echo "[trainer] started $(date -Is) on $(hostname)"; '
        f"echo {shlex.quote('[trainer] command: ' + remote_command)}; "
        f"{remote_command}; "
        'status=$?; echo; echo "[trainer] exit=$status $(date -Is)"; '
        "exec bash"
    )
    command = "\n".join(
        [
            f"tmux has-session -t {session} 2>/dev/null && "
            f"(echo 'session already exists: {args.session}' >&2; exit 2) || true",
            f"tmux new-session -d -s {session} {shlex.quote(runner)}",
            f"tmux display-message -p -t {session} '#S started'",
        ]
    )
    _run(_ssh(host, command))


def cmd_attach(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    _run(_ssh(host, f"tmux attach -t {shlex.quote(args.session)}", tty=True))


def cmd_logs(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    command = f"tmux capture-pane -pt {shlex.quote(args.session)} -S -{int(args.lines)}"
    _run(_ssh(host, command))


def cmd_sessions(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    _run(_ssh(host, "tmux list-sessions || true"))


def cmd_stop(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    _run(_ssh(host, f"tmux kill-session -t {shlex.quote(args.session)}"))


def cmd_gpu(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    _run(_ssh(host, "nvidia-smi"))


def cmd_doctor(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    repo = _require_config(args.repo, option="--repo", env="TRAINER_REPO")
    command = _repo_command(
        repo,
        "\n".join(
            [
                'echo "--- host ---"',
                'hostnamectl --static 2>/dev/null || hostname',
                'systemctl get-default',
                'systemctl is-enabled ssh',
                'systemctl is-enabled sleep.target 2>/dev/null || true',
                'systemctl is-enabled suspend.target 2>/dev/null || true',
                'echo "--- repo ---"',
                'git branch --show-current',
                'git rev-parse --short HEAD',
                'git status --short',
                'echo "--- runtimes ---"',
                'node --version',
                'npm --version',
                'python --version',
                'echo "--- cuda ---"',
                'python - <<\'PY\'\nimport torch\nprint("torch", torch.__version__)\nprint("cuda_available", torch.cuda.is_available())\nprint("cuda_version", torch.version.cuda)\nprint("device_count", torch.cuda.device_count())\nprint("device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")\nPY',
                'nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader',
                'echo "--- network ---"',
                "ip -brief link 2>/dev/null || true",
            ]
        ),
    )
    _run(_ssh(host, command))


def cmd_fetch(args: argparse.Namespace) -> None:
    host = _require_config(args.host, option="--host", env="TRAINER_HOST")
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    remote = f"{host}:{args.path}"
    _run(["rsync", "-av", "--partial", "--progress", remote, str(destination)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control a configured remote trainer over SSH.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Remote trainer SSH host, or TRAINER_HOST.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Remote repository path, or TRAINER_REPO.")

    subparsers = parser.add_subparsers(dest="command_name", required=True)

    status = subparsers.add_parser("status", help="Show remote git/GPU/Python status.")
    status.set_defaults(func=cmd_status)

    pull = subparsers.add_parser("pull", help="Fast-forward the remote repo from origin.")
    pull.set_defaults(func=cmd_pull)

    deps = subparsers.add_parser("deps", help="Install remote npm/Python dependencies.")
    deps.add_argument("--wheelhouse", default=DEFAULT_WHEELHOUSE)
    deps.set_defaults(func=cmd_deps)

    run = subparsers.add_parser("run", help="Run a command synchronously in the remote repo.")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    start = subparsers.add_parser("start", help="Start a long command in a remote tmux session.")
    start.add_argument("session")
    start.add_argument("--pull", action="store_true", help="Pull latest origin first.")
    start.add_argument("--replace", action="store_true", help="Replace an existing session.")
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(func=cmd_start)

    attach = subparsers.add_parser("attach", help="Attach to a remote tmux session.")
    attach.add_argument("session", nargs="?", default=DEFAULT_SESSION)
    attach.set_defaults(func=cmd_attach)

    logs = subparsers.add_parser("logs", help="Print recent output from a remote tmux session.")
    logs.add_argument("session", nargs="?", default=DEFAULT_SESSION)
    logs.add_argument("--lines", type=int, default=160)
    logs.set_defaults(func=cmd_logs)

    sessions = subparsers.add_parser("sessions", help="List remote tmux sessions.")
    sessions.set_defaults(func=cmd_sessions)

    stop = subparsers.add_parser("stop", help="Kill a remote tmux session.")
    stop.add_argument("session")
    stop.set_defaults(func=cmd_stop)

    gpu = subparsers.add_parser("gpu", help="Show nvidia-smi.")
    gpu.set_defaults(func=cmd_gpu)

    doctor = subparsers.add_parser("doctor", help="Run a broader trainer health check.")
    doctor.set_defaults(func=cmd_doctor)

    fetch = subparsers.add_parser("fetch", help="Fetch a remote artifact directory or file.")
    fetch.add_argument("path", help="Remote artifact path.")
    fetch.add_argument("destination", nargs="?", default="output/remote-trainer")
    fetch.set_defaults(func=cmd_fetch)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
