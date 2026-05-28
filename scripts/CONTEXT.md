# Trainer Operations

Trainer Operations cover remote execution for heavy simulator, CUDA, and long
gate work. The remote trainer is an execution target, not a second source of
truth.

## Language

**Remote trainer**:
The RTX machine used for CUDA training, long gates, long sweeps, and heavier
CPU-bound simulator runs.
_Avoid_: production server

**Local workspace**:
The Mac workspace used for editing, review, quick validation, Git operations,
and viewer work.
_Avoid_: client copy

**Trainer session**:
A named remote long-running command, usually managed through the trainer CLI and
remote tmux.
_Avoid_: job unless referring generally

**Artifact fetch**:
Copying the smallest useful result artifacts from the remote trainer back to the
local workspace for interpretation or documentation.
_Avoid_: bulk sync

**Long sweep**:
A high-cost multi-seed, multi-tick, fixture, benchmark, or gate run that should
be opt-in and usually remote.
_Avoid_: fast local check

**Trainer health**:
The status, connectivity, environment, GPU, and repo sync checks that should be
verified before remote work.
_Avoid_: deployment health

## Example Dialogue

Developer: "Can I edit files directly on the trainer and pull them back?"

Domain expert: "No. Edit locally, review locally, move source through GitHub,
and use the trainer for execution."

Developer: "Should I fetch the whole trajectory directory after a v3 run?"

Domain expert: "Only if the next local step needs it. Fetch the smallest set of
reports and artifacts needed to interpret the run."
