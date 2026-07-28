# Phase-A terminal-matrix closure and Drive archive

Run this only after the live two-party guardian has reported
`matrix_complete=true` for the fixed 16-cell Phase-A matrix. The close command
is a stage boundary, not selection authority: it reopens every terminal,
checkpoint prefix, and artifact through the CPU reconstruction validator while
holding all 16 run locks. It also verifies the exact preregistration, launch
authorization, evidence references, guardian transcript, source binding, and
sealed CUDA runtime authority. It also reconstructs the four capture-only
guardian verifiers and requires their digests to equal the live guardian
transcript. A signed transcript digest without that evidence provenance is not
accepted.

The command copies the active run tree and its authority tree into a new bundle
outside both source trees. The copy is descriptor-relative, refuses links,
special files, mount crossings, and concurrent mutation, and never replaces an
existing bundle. The authority copy is an exact reference-graph allowlist:
unreferenced files and directories fail closure instead of being published.
A failure can leave a partial attempt directory; preserve it for diagnosis and
retry with a new `--bundle-id`. Do not delete or reuse the failed attempt.

From the clean detached GPU checkout at the source SHA, after copying the final
Mac guardian transcript into the remote authority root:

```bash
: "${SOURCE_SHA:?export the exact 40-character pushed source SHA}"
: "${REMOTE_REPO:?export the private exact-SHA checkout path}"
: "${RUNTIME_PYTHON:?export the sealed trainer Python path}"
: "${REMOTE_ACTIVE_OUTPUT_ROOT:?export the private active campaign path}"
: "${REMOTE_CLOSED_BUNDLE_PARENT:?export the private closed-bundle parent}"
: "${REMOTE_CLOSURE_RECEIPT:?export the private closure-receipt path}"
cd "$REMOTE_REPO"
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH="$REMOTE_REPO/python" \
  "$RUNTIME_PYTHON" -m evolution_sim.cli.open_ecology_phase_a_archive close \
  --preregistration /absolute/authority/preregistration.json \
  --launch-authorization /absolute/authority/launch-authorization.json \
  --guardian-transcript /absolute/authority/guardian-transcript.json \
  --runtime-venv-authority /absolute/authority/runtime-venv-authority.json \
  --active-output-root "$REMOTE_ACTIVE_OUTPUT_ROOT" \
  --authority-root /absolute/authority \
  --closed-bundle-parent "$REMOTE_CLOSED_BUNDLE_PARENT" \
  --bundle-id phase-a-SOURCE_SHA-ATTEMPT \
  --closure-receipt "$REMOTE_CLOSURE_RECEIPT"
```

Before transfer, record the receipt's whole-file SHA-256 on the GPU host:

```bash
CLOSURE_RECEIPT="$REMOTE_CLOSURE_RECEIPT"
CLOSURE_RECEIPT_SHA256=$(/usr/bin/sha256sum "$CLOSURE_RECEIPT" | /usr/bin/cut -d' ' -f1)
printf '%s\n' "$CLOSURE_RECEIPT_SHA256"
```

Copy only that compact closure receipt to the Mac over the pinned SSH endpoint,
then independently verify that the Mac bytes have the recorded hash. From the
matching clean source checkout, reuse the already sealed archive-tool authority
and its separately recorded whole-file SHA-256:

```bash
MAC_CLOSURE_RECEIPT=/absolute/mac/receipts/phase-a-SOURCE_SHA-ATTEMPT.json
: "${REMOTE_ARCHIVE_STAGING_DIRECTORY:?export the private remote staging path}"
test "$(/usr/bin/shasum -a 256 "$MAC_CLOSURE_RECEIPT" | /usr/bin/cut -d' ' -f1)" = "$CLOSURE_RECEIPT_SHA256"

PYTHONPATH=python .venv/bin/python -m evolution_sim.cli.open_ecology_phase_a_archive archive \
  --closure-receipt "$MAC_CLOSURE_RECEIPT" \
  --expected-closure-receipt-sha256 "$CLOSURE_RECEIPT_SHA256" \
  --archive-tool-authority /absolute/mac/authority/archive-tool-authority.json \
  --expected-archive-tool-authority-sha256 ARCHIVE_AUTHORITY_SHA256 \
  --remote-staging-directory "$REMOTE_ARCHIVE_STAGING_DIRECTORY" \
  --drive-receipt /absolute/mac/receipts/phase-a-SOURCE_SHA-ATTEMPT-drive.json
```

The verified uploader targets
`gdrive:evolution-sim-backups/archives/open-ecology/<campaign-id>/<bundle-id>/`.
It reopens the closed bundle and requires its marker SHA-256 and exact
entry/file/directory/byte counts to equal the externally transferred closure
receipt. The receipt also pins the exact preregistered archive-tool authority,
not merely the Git SHA and endpoint.
The large payload stays on the GPU host during streaming; only compact receipts
need to remain on the Mac. A successful upload does not authorize deleting or
pruning the active matrix, closed bundle, authority evidence, or Drive objects.
