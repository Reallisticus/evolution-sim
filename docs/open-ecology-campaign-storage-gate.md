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
The production CLI additionally requires an external archive-tool authority
document and its separately supplied whole-file SHA-256. That authority binds
the exact source commit and manifest, remote checkout, SSH endpoint, sealed
Drive prefix, rclone configuration digest, canonical absolute local
`ssh`/`rclone`/`git`/`zstd` binaries, canonical absolute remote
Python/Git/zstd/sha256sum/env binaries, and every helper/module involved in
source validation or archive production. Each binary and helper has an external
whole-file SHA-256. PATH is never used to select a production tool. Local
subprocesses receive only `LANG`, `LC_ALL`, and a fixed non-authoritative PATH;
rclone receives its pinned configuration via `--config`. That credential
configuration must remain a canonical regular file owned by the current UID,
have exactly one hard link, and grant no group or other permissions; those
properties are checked while sealing and every time its pin is revalidated.
The effective `ssh -G` output is also SHA-pinned so an alias cannot silently
change hosts, users, ports, or identity configuration. A real noninteractive
connection is made at seal time and around archival; its authenticated server
host-key fingerprint, resolved address, port, and required `publickey`
authentication method must exactly match the external authority.

The coordinator rehashes local tools and the rclone config around use, verifies
all remote tool/helper hashes before the build and again before the receipt,
and passes the exact remote pins into the detached helper. The helper rehashes
its interpreter, Git, zstd, and committed helper bytes before and after archive
production. A replaced executable, modified helper, changed SSH endpoint,
changed rclone config, or changed authority document therefore writes no
receipt. The receipt records the authority digest and the complete
tool/endpoint/helper identity set, but never records rclone credentials or
configuration contents.
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
  --ssh-target "$TRAINER_SSH_TARGET" \
  --remote-repository-root /absolute/clean/checkout \
  --remote-active-campaign-root /absolute/active/campaign \
  --remote-closed-bundle-dir /absolute/closed/bundle-0001 \
  --remote-staging-dir /absolute/staging/bundle-0001 \
  --campaign-id <campaign-id> \
  --bundle-id bundle-0001 \
  --source-git-sha <40-hex-commit> \
  --source-manifest-sha256 <64-hex-runtime-manifest> \
  --expected-marker-sha256 <64-hex-from-transferred-closure-receipt> \
  --expected-entry-count <closure-entry-count> \
  --expected-file-count <closure-file-count> \
  --expected-directory-count <closure-directory-count> \
  --expected-total-file-bytes <closure-total-file-bytes> \
  --tool-authority-path /absolute/evidence/archive-tool-authority.json \
  --tool-authority-sha256 <64-hex-authority-file-digest> \
  --receipt-path /absolute/small-receipts/bundle-0001.json
```

Those five closed-bundle values must come from a separately transferred,
whole-file-SHA-verified closure receipt. Measuring the current remote bundle
and supplying its current values would destroy the replacement-detection
boundary. Phase-A uses the narrower
`evolution_sim.cli.open_ecology_phase_a_archive archive` wrapper so this
binding is automatic.

The authority JSON is strict: unknown or missing keys, duplicate keys,
non-canonical paths, or malformed digests fail closed. Its shape is:

```json
{
  "endpoint": {
    "rclone_base": "gdrive:evolution-sim-backups/archives/open-ecology",
    "rclone_config": {
      "path": "/absolute/canonical/rclone.conf",
      "sha256": "<64-hex>"
    },
    "ssh_effective_config_sha256": "<64-hex>",
    "ssh_connection": {
      "address": "<authenticated-address>",
      "authenticated_host": "<authenticated-host>",
      "authentication": "publickey",
      "host_key": "<algorithm> SHA256:<fingerprint>",
      "port": 22
    },
    "ssh_target": "<sealed-target>"
  },
  "local_tools": {
    "git": {"path": "/absolute/canonical/git", "sha256": "<64-hex>"},
    "rclone": {"path": "/absolute/canonical/rclone", "sha256": "<64-hex>"},
    "ssh": {"path": "/absolute/canonical/ssh", "sha256": "<64-hex>"},
    "zstd": {"path": "/absolute/canonical/zstd", "sha256": "<64-hex>"}
  },
  "remote_helpers": {
    "python/evolution_sim/io/open_ecology_archive_authority.py": "<64-hex>",
    "python/evolution_sim/io/open_ecology_campaign_storage.py": "<64-hex>",
    "python/evolution_sim/io/source_manifest.py": "<64-hex>",
    "scripts/archive_evolution_outputs.py": "<64-hex>",
    "scripts/archive_open_ecology_campaign.py": "<64-hex>"
  },
  "remote_tools": {
    "env": {"path": "/absolute/remote/env", "sha256": "<64-hex>"},
    "git": {"path": "/absolute/remote/git", "sha256": "<64-hex>"},
    "python": {"path": "/absolute/remote/python", "sha256": "<64-hex>"},
    "sha256sum": {"path": "/absolute/remote/sha256sum", "sha256": "<64-hex>"},
    "zstd": {"path": "/absolute/remote/zstd", "sha256": "<64-hex>"}
  },
  "schema_version": "open_ecology_archive_tool_authority_v1",
  "source": {
    "git_sha": "<40-hex-commit>",
    "manifest_sha256": "<64-hex-runtime-manifest>",
    "remote_repository_root": "/absolute/clean/detached/checkout"
  }
}
```

Create it only after the source is committed, the local checkout is clean, and
the remote checkout is clean and detached at that same commit. The sealer takes
only explicit absolute tool paths, measures every byte twice, compares every
remote helper with the clean local source, verifies remote HEAD/status, writes
the authority exclusively, and prints the separate digest used by `archive`:

```bash
PYTHONHASHSEED=0 PYTHONPATH=python .venv/bin/python \
  scripts/seal_open_ecology_archive_authority.py \
  --repository-root /absolute/local/evolution-sim \
  --remote-repository-root /absolute/remote/detached-checkout \
  --ssh-target <sealed-target> \
  --rclone-config-path /absolute/canonical/rclone.conf \
  --local-ssh-path /absolute/canonical/ssh \
  --local-rclone-path /absolute/canonical/rclone \
  --local-git-path /absolute/canonical/git \
  --local-zstd-path /absolute/canonical/zstd \
  --remote-python-path /absolute/remote/python \
  --remote-git-path /absolute/remote/git \
  --remote-zstd-path /absolute/remote/zstd \
  --remote-sha256sum-path /absolute/remote/sha256sum \
  --remote-env-path /absolute/remote/env \
  --output /absolute/evidence/archive-tool-authority.json
```
