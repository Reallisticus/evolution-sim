# Artifact storage and verified pruning

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
6. uploads the archive, manifest, and archive-SHA sidecar as three immutable
   objects;
7. streams every remote object back through SHA256 and compares it with the
   local bytes;
8. prunes only when `--prune-after-verify` was explicitly supplied and every
   verification passed.

Pruning removes only entries in the verified manifest and leaves the named
input directory itself in place. A new or changed file blocks pruning. Remote
failure, hash disagreement, source drift, unsupported filesystem entries, or
missing tools all fail closed.

Quiesce every process that can write to the input before requesting pruning.
The tool rescans for drift and will stop when it observes a change, but pruning
is not a substitute for coordinating active simulator or viewer writers.

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

Prune the exact input contents only after all remote verification gates:

```bash
python3 scripts/archive_evolution_outputs.py \
  --input-dir /Users/njm/Projects/evolution-sim/output \
  --output-dir /Users/njm/evolution-sim-staging \
  --archive-name 20260727T120000Z-evolution-output.tar.zst \
  --prune-after-verify
```

Use `--remote REMOTE:ROOT` and `--remote-subdir PATH` to select another
pre-existing rclone directory. The defaults are
`gdrive:evolution-sim-backups` and `archives`.

The output directory must be outside the input. Local compression needs space
for the compressed archive but never another uncompressed copy. When the Mac is
critically full, first remove only already-verified expanded backup copies or
place `--output-dir` on another volume.

## Stream a GPU-host campaign without using Mac staging space

For campaigns generated on `gpu4070`, keep the expanded evidence and temporary
compressed archive on the server's NVMe. The coordinator can invoke the same
deterministic archiver remotely and pipe each resulting object directly from
SSH stdout into rclone stdin:

```bash
python3 scripts/stream_remote_evolution_archive.py \
  --ssh-target gpu4070 \
  --remote-repository-root /home/train/evolution-sim-open-ecology-checkouts/COMMIT \
  --remote-input-dir /home/train/evolution-sim-open-ecology-runs/CAMPAIGN \
  --remote-staging-dir /home/train/evolution-sim-open-ecology-archives \
  --archive-name YYYYMMDDTHHMMSSZ-CAMPAIGN.tar.zst
```

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
`563604b1f77db0103ada5416df690d53fc7599d0` on `gpu4070`. The remote test
archive was streamed directly to Drive and read back with matching SHA256
`80ea6479d0aef58abe08f185969fd507f8d822d7df4457d8d6de3c783f5d1e32`;
the manifest and sidecar also matched their independently computed source
digests. Neither source nor remote staging was pruned by the test.

## Restore

Download all three objects into a new empty directory:

```bash
rclone copy \
  gdrive:evolution-sim-backups/archives/20260727T120000Z-evolution-output.tar.zst \
  /Users/njm/evolution-sim-restore/
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
