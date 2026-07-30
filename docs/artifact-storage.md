# Artifact storage and verified archival

Generated simulator evidence is intentionally ignored by Git. Git preserves the
source and experiment contract; an object store preserves large reports,
trajectories, checkpoints, and replay bundles.

The default cold-storage destination is:

`gdrive:evolution-sim-backups/archives`

The remote name and root are configurable. Never commit the local rclone
configuration because it contains credentials.

## Safety contract

`scripts/archive_evolution_outputs.py` accepts exactly one explicit input
directory. It does not accept globs, multiple roots, or implicit workspace-wide
cleanup. It:

1. walks the directory without following symlinks;
2. rejects dangling symlinks and symlinks whose resolved target escapes the
   input directory;
3. writes a stable, path-sorted SHA256 manifest for every regular file;
4. creates a normalized, deterministic `tar.zst` with stable tar metadata;
5. tests the local zstd stream and rescans the input for drift;
6. captures the exact identity, metadata, size, and SHA-256 of each local
   evidence object, then uploads through an already-open file descriptor rather
   than reopening a mutable pathname;
7. streams every remote object back through SHA256 and compares it with the
   originally captured local digest and size;
8. never deletes or prunes local inputs or partially written evidence.

Remote failure, hash disagreement, local identity drift, source drift,
unsupported filesystem entries, or missing tools all fail closed. Failed
exclusive writes can leave a non-authoritative local object behind; the tool
does not remove it by pathname because a stat-then-unlink cleanup can delete a
racing replacement. No success receipt is emitted for such an object, and the
same immutable name cannot be reused.

`--prune-after-verify` is retained only as a fail-closed compatibility
sentinel. It always stops before archival. Remove local source data only as a
separate, explicitly reviewed offline operation after checking the remote
archive, manifest, and SHA sidecar.

This generic command is retained for maintenance and historical evidence only.
It is not an authorized open-ecology campaign publication route and cannot
produce the sealed campaign receipt. Production open-ecology archival must use
`scripts/archive_open_ecology_campaign.py`.

## Operating modes

Inspect and hash one exact directory without writing anything:

```bash
python3 scripts/archive_evolution_outputs.py \
  --input-dir /Users/njm/Projects/evolution-sim/output \
  --dry-run
```

Create local evidence without uploading or deleting:

```bash
python3 scripts/archive_evolution_outputs.py \
  --input-dir /Users/njm/Projects/evolution-sim/output \
  --output-dir /Users/njm/evolution-sim-staging \
  --archive-name 20260727T120000Z-evolution-output.tar.zst \
  --archive-only
```

Upload and verify while retaining the local source:

```bash
python3 scripts/archive_evolution_outputs.py \
  --input-dir /Users/njm/Projects/evolution-sim/output \
  --output-dir /Users/njm/evolution-sim-staging \
  --archive-name 20260727T120000Z-evolution-output.tar.zst
```

Requests for automatic pruning are rejected:

```bash
python3 scripts/archive_evolution_outputs.py \
  --input-dir /Users/njm/Projects/evolution-sim/output \
  --output-dir /Users/njm/evolution-sim-staging \
  --archive-name 20260727T120000Z-evolution-output.tar.zst \
  --prune-after-verify
```

The command above exits nonzero without writing, uploading, or deleting.

Use `--remote REMOTE:ROOT` and `--remote-subdir PATH` to select another
pre-existing rclone directory. The defaults are
`gdrive:evolution-sim-backups` and `archives`.

The output directory must be outside the input. Local compression needs space
for the compressed archive but never another uncompressed copy. When the Mac is
critically full, first remove only already-verified expanded backup copies or
place `--output-dir` on another volume.

## Deprecated legacy GPU-host streaming example

`scripts/stream_remote_evolution_archive.py` is not authorized for new
open-ecology campaigns because it predates the sealed tool/endpoint authority,
closed-bundle gate, exact Drive inventory, and immutable receipt. The command
below is retained only to explain historical archives; do not use it for a new
campaign:

```bash
python3 scripts/stream_remote_evolution_archive.py \
  --ssh-target "$TRAINER_SSH_TARGET" \
  --remote-repository-root "$REMOTE_REPOSITORY_ROOT" \
  --remote-input-dir "$REMOTE_CAMPAIGN_ROOT" \
  --remote-staging-dir "$REMOTE_ARCHIVE_STAGING_ROOT" \
  --archive-name YYYYMMDDTHHMMSSZ-CAMPAIGN.tar.zst
```

The authoritative open-ecology route is
`scripts/archive_open_ecology_campaign.py`, not the older generic streaming
example above. It requires the separately SHA-sealed executable/endpoint/helper
authority described in `docs/open-ecology-campaign-storage-gate.md`, streams
each of the three objects directly from the compute host to Drive, reads every
byte back, performs the independent three-row `rclone check`, and writes only a
small receipt on the Mac. No archive payload is staged on the Mac and this
route contains no deletion or pruning operation.

This path does not put archive payload bytes on the Mac filesystem. It first
creates and validates the deterministic archive on the server, refuses any
destination-name collision, streams the archive, manifest, and SHA sidecar
directly into `gdrive:evolution-sim-backups/archives`, and streams all three
Drive objects back through SHA256 for comparison with independently hashed
server files.

The streaming tool never deletes the server's source or staging files. Treat
its successful JSON result as an archival verification gate, inspect the
recorded object paths and digests, and only then schedule a separate,
explicitly scoped server cleanup. This separation ensures a transfer bug can
never silently turn into source deletion.

The path was exercised from clean source commit
`563604b1f77db0103ada5416df690d53fc7599d0` on a privately configured trainer. The remote test
archive was streamed directly to Drive and read back with matching SHA256
`80ea6479d0aef58abe08f185969fd507f8d822d7df4457d8d6de3c783f5d1e32`;
the manifest and sidecar also matched their independently computed source
digests. Neither source nor remote staging was pruned by the test.

## Restore

Download all three objects into a new empty directory:

```bash
rclone copy \
  gdrive:evolution-sim-backups/archives/20260727T120000Z-evolution-output.tar.zst \
  /absolute/restore/directory/
rclone copy \
  gdrive:evolution-sim-backups/archives/20260727T120000Z-evolution-output.tar.zst.manifest.json \
  /Users/njm/evolution-sim-restore/
rclone copy \
  gdrive:evolution-sim-backups/archives/20260727T120000Z-evolution-output.tar.zst.sha256 \
  /Users/njm/evolution-sim-restore/
```

Verify before extraction:

```bash
cd /Users/njm/evolution-sim-restore
shasum -a 256 -c 20260727T120000Z-evolution-output.tar.zst.sha256
zstd -t 20260727T120000Z-evolution-output.tar.zst
zstd -dc 20260727T120000Z-evolution-output.tar.zst | tar -tf -
```

Extract only into the intended empty restore directory:

```bash
zstd -dc 20260727T120000Z-evolution-output.tar.zst | tar -xf -
```

After extraction, hash the restored regular files and compare them with the
JSON manifest. Do not treat a successful download alone as evidence that every
source artifact is present.

## Retention model

Keep the active experiment's compact report, policy artifact, preregistration,
runtime provenance, and short viewable replay locally. Send full trajectories,
counterfactual matrices, terminal checkpoints, historical viewer captures, and
closed-campaign bundles to cold storage.

Use a new immutable archive name for every campaign or cleanup pass. Never
overwrite an earlier evidence bundle. Record the source commit and archive
object names in the experiment report, and retain at least the remote archive,
manifest, and SHA sidecar together.

## Verified initial migration

On 2026-07-27, the pre-open-ecology `output/` tree was archived as
`20260727T123500Z-evolution-output-pre-open-ecology.tar.zst`. The source
snapshot contained 2,812 files and 10,210,110,929 bytes. Its compressed archive
is 1,271,366,930 bytes with SHA256
`113703bd1d076e85755794cb29f60aba52b017136400b8236dc0813c77846a8a`.

The archive, manifest, and SHA sidecar were uploaded to
`gdrive:evolution-sim-backups/archives`, streamed back through SHA256 by the
archival tool, and independently checked with `rclone check` as three matching
objects with zero differences. Only then were the manifested local contents
pruned. The local `output/` directory itself remains present for new runs.
