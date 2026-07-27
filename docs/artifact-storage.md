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
