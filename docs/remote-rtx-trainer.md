# Remote Trainer

The remote trainer is an optional execution target, not a second source of
truth. Keep source edits, review, local quick checks, and Git operations on the
local workstation. Use GitHub to move source code to a configured trainer when
long simulator runs or accelerator-backed training would block local work.

All trainer connection details are intentionally private local configuration.
Do not commit real SSH aliases, account names, hostnames, local network details,
or absolute remote paths.

## Configuration

Configure the trainer with environment variables or pass equivalent CLI flags:

```bash
export TRAINER_HOST="<trainer-host>"
export TRAINER_REPO="<remote-repo>"
export TRAINER_WHEELHOUSE="<optional-remote-wheelhouse>"
```

`TRAINER_HOST` should be an SSH target known to the local machine.
`TRAINER_REPO` should point at the checked-out repository on that trainer.
`TRAINER_WHEELHOUSE` is optional and is used only when an offline Python package
cache is available.

## Daily Workflow

Check trainer health:

```bash
npm run trainer:status
npm run trainer:doctor
```

After pushing local changes to GitHub, fast-forward the trainer:

```bash
npm run trainer:pull
```

For a review branch, switch the trainer to that branch before running remote
checks:

```bash
npm run trainer -- checkout codex/<branch-name>
```

Install or refresh the optional Mind ML stack:

```bash
npm run trainer:deps
```

Run a short command synchronously on the trainer:

```bash
npm run trainer -- run npm run sim:bench:quick
```

Start a long command in a remote `tmux` session:

```bash
npm run trainer -- start mind-gate --pull -- npm run sim:mind:gate:extended
```

Watch or attach to the long run:

```bash
npm run trainer -- logs mind-gate
npm run trainer -- attach mind-gate
```

Stop a session if needed:

```bash
npm run trainer -- stop mind-gate
```

Fetch result artifacts back only when needed:

```bash
npm run trainer -- fetch "<remote-repo>/output/mind/<report>.json"
```

For Mind v3 work, fetch the smallest set of artifacts needed to interpret the
run: the trained artifact, strict slice report, train-gate report, diagnostics,
and any ledger/report JSON that will be referenced in
`docs/mind-v3-autonomous-evolution.md`. Do not bulk-transfer trajectory
directories unless the next local step needs them.

Validate the optional PyTorch/CUDA Mind stack on the NVIDIA trainer with:

```bash
npm run trainer -- run npm run sim:mind:torch:validate:cuda
```

This runs the torch-gated Mind unit tests, requires CUDA visibility, runs a tiny
CUDA-backed training smoke, and writes
`output/mind/mind-torch-validation-report.json` in the trainer checkout. Fetch
that report only when it will be referenced in a review or artifact ledger:

```bash
npm run trainer -- fetch "<remote-repo>/output/mind/mind-torch-validation-report.json"
```

This is an optional ML-stack validation path, not promotion evidence and not a
replacement for strict held-out Mind gates.

For complex shell commands with pipes, redirects, or multiple steps, wrap the
remote command in `bash -lc`:

```bash
npm run trainer -- start experiment --replace -- bash -lc 'npm run sim:bench && nvidia-smi'
```

## What Should Run Where

Prefer the trainer for:

- Accelerator-backed `sim:mind:train` runs.
- Long `sim:mind:gate:*` runs.
- Mind v3 strict candidate runs that need many seed/fixture combinations.
- Long seed sweeps and benchmark runs.
- Any CPU-bound simulator run that would block local development.

Prefer the local workstation for:

- Editing and code review.
- Fast local checks before pushing.
- Viewer/browser work.
- Cross-platform replay/golden checks unless the task is specifically about the
  trainer platform.

## Operational Notes

Keep the trainer reachable only through private local infrastructure. Avoid
committing operational topology, network names, host aliases, usernames, or
hardware identifiers. Treat generated trainer outputs as reproducible artifacts:
fetch and document only the files needed for the current review.
