# Open-Ecology Host Health Probe

`evolution_sim.cli.open_ecology_health` is the external, fail-closed resource
probe consumed by the persistent campaign coordinator. It binds one immutable
baseline to the exact campaign ID, canonical active root, source Git SHA,
source-manifest SHA256, Linux boot ID, initial swap use, initial kernel Xid/OOM
counts, and descriptor-safe SHA-256 authorities for the canonical
`nvidia-smi` and `journalctl` executables before any campaign worker starts.
Discovery ignores inherited `PATH` and uses a fixed system search path; every
check revalidates the exact absolute executable identities and runs them with a
constrained environment. Stdout and stderr are drained concurrently under one
combined byte ceiling and wall-clock deadline, and a breach terminates the
isolated command process group.

The coordinator invokes `check` immediately before every 5,000-tick advance and
again while all started tasks are quiescent at the frontier. A snapshot is
healthy only when the host has not rebooted, swap has not grown, RAM use remains
below 80%, the campaign filesystem retains at least 100 GiB and 20% free, the
kernel Xid and OOM counts are unchanged, every GPU remains below 80% memory
use, and every GPU is below its driver-reported slowdown temperature. Missing `/proc`,
`journalctl`, `nvidia-smi`, detailed slowdown temperature, malformed output, or
baseline integrity is a blocker rather than an assumed pass. The baseline is
not self-authorizing: `init` prints its whole-file SHA-256, the operator seals
that digest into the externally hashed launch specification, and every `check`
requires the same digest. A valid-looking replacement baseline is rejected.
A durable
`.open-ecology-fatal-worker-pids.json` marker is also an unconditional blocker:
it means a prior launcher could not prove that all worker PIDs stopped after
both terminate and kill deadlines. The probe does not remove or waive that
marker.

The baseline belongs outside the source checkout and active campaign tree. A
fresh exact-SHA launch creates it once:

```bash
PYTHONPATH=python /absolute/venv/bin/python \
  -m evolution_sim.cli.open_ecology_health init \
  --baseline /absolute/control/health-baseline.json \
  --campaign-root /absolute/active-campaign \
  --campaign-id <campaign-id> \
  --source-git-sha <40-char-sha> \
  --source-manifest-sha256 <64-char-sha256>
```

The `init` output includes `health_baseline_sha256`. That digest is computed
from the exact bytes written and descriptor-validated by `init`; it is not
obtained by reopening the path. A manual check supplies it explicitly with
`--expected-baseline-sha256`.

Create the one-file campaign probe with the fail-if-present builder:

```bash
PYTHONPATH=python /absolute/venv/bin/python \
  -m evolution_sim.cli.open_ecology_health build-wrapper \
  --output /absolute/control/open-ecology-health-probe \
  --python-executable /absolute/venv/bin/python \
  --health-script /absolute/exact-checkout/python/evolution_sim/cli/open_ecology_health.py \
  --expected-health-script-sha256 <externally-retained-source-file-sha256> \
  --baseline /absolute/control/health-baseline.json \
  --expected-baseline-sha256 <health-baseline-file-sha256> \
  --campaign-root /absolute/active-campaign \
  --campaign-id <campaign-id> \
  --source-git-sha <40-char-sha> \
  --source-manifest-sha256 <64-char-sha256>
```

The builder descriptor-hashes the exact Python executable and health source,
copies the complete health implementation into the generated executable, and
binds the fixed check arguments into its terminal entry point. The source file
is not imported at runtime. The generated shebang includes the canonical
Python path and an embedded interpreter SHA-256 that the coordinator
revalidates. The builder uses exclusive creation and refuses an existing
output.

The launch specification stores that digest as `health_baseline_sha256`, stores
the builder's `wrapper_sha256` as `health_probe_file_sha256`, and names exactly
one self-contained probe executable in `health_probe_command`.
It must not express `python -m`, `python -c`, or a replaceable script/module as
additional argv. A native executable is pinned directly. An executable script
is snapshotted byte-for-byte and its one absolute native shebang interpreter is
independently pinned through the mandatory
`# evosim_shebang_sha256=<digest>` second line. The probe executable therefore embeds its fixed
campaign/baseline arguments and emits the check JSON directly; the coordinator
supplies only `EVOSIM_OPEN_ECOLOGY_HEALTH_PHASE` and
`EVOSIM_OPEN_ECOLOGY_FRONTIER_TICK` in a constrained environment. Its output
must bind the externally retained digest as `baseline_file_sha256`. The probe
writes one bounded JSON snapshot to stdout and never mutates the baseline,
campaign, source, or GPU.
