# Remote RTX Trainer

The RTX trainer is a remote execution target, not a second source of truth.

Use the Mac for editing, review, local quick checks, and Git operations. Use
GitHub to move source code to the trainer. Use the trainer for CUDA training,
long sweeps, and heavier CPU-bound simulator runs.

## Machine

- SSH alias: `gpu4070`
- Remote repo: `/home/train/Projects/evolution-sim`
- Remote venv: `/home/train/Projects/evolution-sim/.venv`
- GPU: NVIDIA GeForce RTX 4070 SUPER

## Daily Workflow

Check trainer health:

```bash
npm run trainer:status
npm run trainer:doctor
```

After pushing Mac changes to GitHub, fast-forward the trainer:

```bash
npm run trainer:pull
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

Fetch result artifacts back to the Mac only when needed:

```bash
npm run trainer -- fetch /home/train/Projects/evolution-sim/output/mind/mind-v1-gate-extended-report.json
```

For Mind v3 work, fetch the smallest set of artifacts needed to interpret the
run: the trained artifact, strict slice report, train-gate report, diagnostics,
and any ledger/report JSON that will be referenced in
`docs/mind-v3-autonomous-evolution.md`. Do not bulk-transfer trajectory
directories unless the next local step needs them.

For complex shell commands with pipes, redirects, or multiple steps, wrap the
remote command in `bash -lc`:

```bash
npm run trainer -- start experiment --replace -- bash -lc 'npm run sim:bench && nvidia-smi'
```

## What Should Run Where

Prefer the trainer for:

- CUDA-backed `sim:mind:train` runs.
- Long `sim:mind:gate:*` runs.
- Mind v3 strict candidate runs that need PyTorch, CUDA, or many seed/fixture
  combinations.
- Long seed sweeps and benchmark runs.
- Any CPU-bound simulator run that would block local development.

Prefer the Mac for:

- Editing and code review.
- Fast local checks before pushing.
- Viewer/browser work.
- Cross-platform golden checks until Linux/macOS replay float normalization is fixed.

## Known Caveats

The trainer currently negotiates Ethernet at `100Mb/s`. Training is local to the
RTX box, so this mostly affects package downloads, dataset transfer, and artifact
fetches. Fix the cable, switch port, or router port when convenient; target
`1000Mb/s` or `2500Mb/s`.

`sim:golden:quick` currently differs between Apple Silicon and Linux x86_64 by a
last-bit viewer float in `seed7_ticks20`. Treat that as a repo determinism issue,
not a trainer setup failure.

Do not expose SSH directly to the public internet. For remote access outside the
LAN, put the trainer behind WireGuard, Tailscale, or an equivalent VPN first.
