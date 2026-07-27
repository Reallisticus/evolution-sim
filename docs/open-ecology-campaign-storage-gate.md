# Open-ecology campaign storage gate

This gate implements the sealed storage contract in
`docs/research/open-ecology-campaign-preregistration-v1.md`. It does not make
scientific claims and it does not expose a deletion or pruning operation.

The compute host remains the active store. The active campaign root is scanned
under the same campaign/source lock used by the persistent runner, with a
200-GiB hard ceiling that cannot be raised and a free-space floor equal to the
greater of 100 GiB or 20% of the filesystem. The remote 300-GiB floor and
entry ceiling likewise cannot be weakened through CLI options. The scan rejects
raw/resolved path drift, a symlink in any ancestor, symlinks inside the tree,
hard-linked files, special files, mount crossings, and files that change while
being opened or hashed. The active root is never passed to an archive producer.

An archive input is a separate bundle directory whose basename is its
`bundle-id`. `seal_closed_bundle` is the only closure route: while holding the
campaign storage lock it removes write permissions, hashes every regular file
through a no-follow descriptor, and exclusively writes
`.open-ecology-closed.json`. The marker binds the campaign, bundle, exact
source commit, source-manifest digest, complete sorted entry set, modes, sizes,
and file hashes. Archival revalidates that marker and all bytes. A bundle inside
or above the active root, a writable entry, marker change, or content change
fails closed.

The remote helper must run from the claimed clean detached checkout. It verifies
exact `HEAD`, detached-HEAD state, an empty Git status, the runtime
source-manifest digest, the imported storage module's checkout root, and
committed-byte equality for both the helper and
`scripts/archive_evolution_outputs.py`. This extra byte comparison prevents Git
`assume-unchanged` or `skip-worktree` flags from hiding a modified producer.
That existing producer creates the deterministic `tar.zst`, canonical manifest,
and archive-SHA sidecar in a separate compute-host staging directory. The helper
strictly compares the
producer manifest, entry set, sizes, modes, and hashes to the descriptor-safe
closed-bundle preflight and revalidates both source and bundle after production.
The coordinator accepts source objects only from the exact remote staging
directory named in the request; an absolute path with merely a matching
basename is rejected.

The only Drive destination is:

`gdrive:evolution-sim-backups/archives/open-ecology/<campaign-id>/<bundle-id>/`

The default and preregistered free-space floor is 300 GiB. Exactly three
positive objects are allowed: `<name>.tar.zst`,
`<name>.tar.zst.manifest.json`, and `<name>.tar.zst.sha256`. A retry may reuse
an existing object only after a complete `rclone cat` SHA-256 and size match;
it may fill missing exact objects. A mismatched object, surplus name, duplicate
name, partial producer triple, empty Drive ID, or duplicate Drive ID stops the
operation.

After upload the coordinator streams all three objects back through SHA-256 and
size accounting. It then writes a temporary, tiny SHA256 sum file on the Mac
and runs:

```text
rclone check <sum-file> <destination> --checkfile SHA-256 --combined -
```

The command must exit zero and return exactly three canonical `= <basename>`
rows. `--one-way` is not used. A fresh `rclone lsjson` must then contain exactly
the three unique names, positive sizes, and three unique nonempty Drive IDs.
That exact inventory is sampled immediately before byte verification and again
after `rclone check`; names, sizes, and IDs must remain identical so the
receipt cannot bind verified bytes to a replacement object.
Only after all of those gates pass is a small immutable receipt written on the
Mac. The receipt records the exact pre-upload and post-upload Drive free-byte
measurements and the sealed minimum alongside all three IDs, sizes, and hashes.
Source archives and closed bundles remain on the compute host; this version
never deletes or prunes them.

The focused validation command is:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python -m unittest \
  python.tests.test_open_ecology_campaign_storage
```

Run the CLI only from a committed clean checkout:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python \
  scripts/archive_open_ecology_campaign.py archive \
  --ssh-target gpu4070 \
  --remote-repository-root /absolute/clean/checkout \
  --remote-active-campaign-root /absolute/active/campaign \
  --remote-closed-bundle-dir /absolute/closed/bundle-0001 \
  --remote-staging-dir /absolute/staging/bundle-0001 \
  --campaign-id <campaign-id> \
  --bundle-id bundle-0001 \
  --source-git-sha <40-hex-commit> \
  --source-manifest-sha256 <64-hex-runtime-manifest> \
  --receipt-path /absolute/small-receipts/bundle-0001.json
```
