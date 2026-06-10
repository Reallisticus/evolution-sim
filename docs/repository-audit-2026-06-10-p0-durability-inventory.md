# P0 Durability Inventory

Date: 2026-06-10

This records the non-destructive P0 preservation pass after the full repository
audit. No commit, push, reset, clean, delete, or artifact move was performed.

## Source State At Intake

- Branch: `master`
- Base HEAD: `e7d7e3a02255e165b299dc5d866bd49e98479287`
- Last committed change: `2026-05-28T16:45:21+03:00`
- Tracked modified files: `13`
- Untracked files: `127`
- Dirty source files inventoried: `140`
- Dirty source size: `3,947,144` bytes
- Dirty source lines: `96,631`
- Tracked diff size: `2,758` insertions, `99` deletions

Untracked source split:

- `python/evolution_sim/cli/`: `45`
- `python/evolution_sim/mind/`: `40`
- `python/tests/`: `40`
- `docs/`: `2`

## Local Preservation Snapshot

A local preservation snapshot was written outside the repo:

`/Users/njm/evolution-sim-p0-backups/20260610T120558Z`

It contains:

- `dirty-files-current-content.tar.gz`: current contents for all dirty files.
- `tracked-changes.diff`: `git diff --binary` for tracked modifications.
- `git-status.txt`, `dirty-files.txt`, `untracked-files.txt`,
  `tracked-changes.stat`, `tracked-changes.numstat`.
- `dirty-source-manifest.tsv`: size, line count, and SHA256 for every dirty
  source file.
- `p0-artifact-manifest.json`: dirty-tree digest/path references plus full
  `output/mind/` SHA256 index.
- `output-mind/`: local copy of the full `output/mind/` tree.
- `output-mind-copy-verification.tsv`: copied artifact SHA256 verification.
- `output-trajectories/`: local copy of `output/trajectories/`.
- `output-audits/`: local copy of `output/audits/`.
- `output-non-mind-copy-verification.json`: copied trajectory/audit
  verification.
- `SHA256SUMS.txt`: checksum ledger for the snapshot contents.

The local `output/mind/` copy was verified against the original SHA256 index:

- Original files: `1,853`
- Copied files: `1,853`
- Missing files: `0`
- Extra files: `0`
- Mismatched files: `0`

This is a same-disk backup. It protects against repo-local cleanup or accidental
artifact pruning, but it does not protect against disk loss by itself. The
Google Drive archive below is the second storage target for this snapshot.

Additional ignored-output preservation:

- `output/trajectories/`: `564` files, `266,144,400` bytes, copied and verified
  with `0` missing, extra, or mismatched files.
- `output/audits/`: `1` file, `23,439` bytes, copied and verified with `0`
  missing, extra, or mismatched files.

## Google Drive Backup

The local snapshot was compressed into:

`/Users/njm/evolution-sim-p0-backups/20260610T120558Z.tar.zst`

It was uploaded with `rclone` to:

`gdrive:evolution-sim-backups/archives/20260610T120558Z.tar.zst`

The companion checksum file was uploaded to:

`gdrive:evolution-sim-backups/archives/20260610T120558Z.tar.zst.sha256`

Verification:

- `rclone check /Users/njm/evolution-sim-p0-backups gdrive:evolution-sim-backups/archives --include '20260610T120558Z.tar.zst*' --one-way`: pass, `0` differences, `2` matching files.
- Local archive SHA256:
  `89a39491d1727cb6ee1a4585cca3e3302b4700ba927fe12d4a3c7a9e2b4b3b1c`
- Local/remote archive MD5:
  `314f261917682f903d68b2434a27d732`

Restore outline:

```bash
rclone copy gdrive:evolution-sim-backups/archives/20260610T120558Z.tar.zst /Users/njm/evolution-sim-p0-restore/
rclone copy gdrive:evolution-sim-backups/archives/20260610T120558Z.tar.zst.sha256 /Users/njm/evolution-sim-p0-restore/
cd /Users/njm/evolution-sim-p0-restore
shasum -a 256 -c 20260610T120558Z.tar.zst.sha256
zstd -dc 20260610T120558Z.tar.zst | tar -xf -
```

The `rclone` config is local user state under `~/.config/rclone/rclone.conf`.
Do not commit or print it; it contains Google OAuth credentials.

## Artifact Inventory

Original `output/mind/` inventory:

- Files: `1,853`
- Bytes: `8,082,440,307`
- Reported disk use: `7.5G`

Dirty-tree references:

- Unique `output/mind/` path references: `372`
- Missing referenced paths: `21`
- Unique SHA256-looking references: `51`
- SHA references matching raw `output/mind/` file bytes: `0`
- SHA references found inside `output/mind/` artifact contents: `50`

The SHA references are mostly report-internal stable payload, dataset, exact, or
branch-evidence digests, not raw file checksums. Reproducibility therefore
depends on preserving the report files and their internal payloads, not only on
matching file bytes.

## Missing Reference Triage

The following missing references are examples or glob prefixes, not evidence
loss:

- `output/mind/seed7-bc-artifact.json`
- `output/mind/seed7-bc-eval.json`
- `output/mind/seed7-neural-artifact.json`
- `output/mind/seed7-torch-artifact.json`
- `output/mind/seed7-torch-advantage-artifact.json`
- `output/mind/seed7-torch-iql-artifact.json`
- `output/mind/mind-v3-horizon-labels.json`
- `output/mind/mind-v3-fixture-labels.json`
- `output/mind/mind-v3-neural-artifact.json`
- `output/mind/mind-v3-neural-80-eval.json`
- `output/mind/mind-v3-v88-branch-action-oracle-option-preview-`
- `output/mind/v98-broad-support-trajectories/open-mind-v3-`
- `output/mind/v138-strict-heldout-trajectories/open-mind-v3-`
- `output/mind/test-trajectories/open-mind-v3-`
- `output/mind/v146-test`

The following are historical default command paths, not the canonical preserved
v148 evidence dependency:

- `output/mind/mind-v3-v148-carrion-specific-archive-expansion-report.json`
- `output/mind/mind-v3-v148-carrion-specific-archive-expansion-dataset.jsonl`
- `output/mind/mind-v3-v148-carrion-specific-archive-expansion-chunks`

The v148 ledger entry was corrected to the sharded/merged paths:

- `output/mind/shards/v148-carrion/merged-report.json`
- `output/mind/shards/v148-carrion/merged-dataset.jsonl`
- `output/mind/shards/v148-carrion/*-chunks/`

The following chunk directories are referenced by default constants but are not
present locally:

- `output/mind/mind-v3-v154-carrion-survivor-continuation-chunks`
- `output/mind/shards/bp3-public-sequence-context-branch-evidence-chunks`

Their merged report/dataset artifacts are present. If the chunk-level evidence
is meant to be part of replayable provenance, it must be regenerated or the
ledger must state that only merged evidence is durable.

The CLI default output
`output/mind/mind-v3-autonomous-evolution-report.json` is also absent; this is
a generic default path, not a recorded evidence dependency.

## Digest Drift

One dirty-tree digest was not present inside the current artifact tree at
intake:

`6db0e409b4834f97cd54e60bff51db54317e80d80ef8b7727a2f9c510efce66e`

This appeared in `docs/mind-v3-autonomous-evolution.md` as a partial merged
v171 exact digest. The ledger was corrected to the complete v171 artifact. The
current local v171 report records the complete v171 exact digest:

`80d46c17927cdf33ebb7ea23d2e5d24d1bcf7d560ac856ae790f8d456a67904e`

That complete digest is what v172 source validation expects and what the local
v171/v172 artifacts contain. This was documentation provenance drift, not a
current source-integrity blocker.

## Validation Run

These checks were run after the snapshot:

- `git diff --check`: pass
- `node -e 'JSON.parse(package.json)'`: pass, `149` scripts
- `python3 -m compileall -q python/evolution_sim python/tests`: pass
- JSON parse for `package.json`, `package-lock.json`, and
  `p0-artifact-manifest.json`: pass
- Full local `output/mind/` copy verification: pass, `0` mismatches

No simulator gate, Mind gate, or long experiment was run in this P0 pass.

## Commit Slice Plan

Do not start new Mind v3 experiments before source/evidence durability is
complete.

Recommended source commits, if git actions are authorized:

1. Audit/orientation durability docs:
   `AGENTS.md`, `README.md`, `docs/repository-audit-2026-06-10-remediation.md`,
   this inventory doc, `docs/versioning.md`, onboarding, and local agent-skill
   updates.
2. v137-v145 route reset, sequence/transition scorer, branch intervention, and
   branch-label causal audit:
   affected CLI wrappers, `mind/` implementations, tests, core v3 policy hooks,
   support-residual changes, and corresponding package scripts.
3. v146 safe-archive/candidate-campaign and public sequence-context branch
   evidence:
   `candidate_campaign.py`, `safe_archive_*`, `public_sequence_context_*`,
   wrappers, tests, and package scripts.
4. v148-v153 carrion-specific archive and sequence-context closeout:
   carrion-specific archive expansion/train-eval, archive override autopsy,
   sequence-context archive, ablation, comparator closeout, wrappers, tests, and
   package scripts.
5. v154-v162 carrion-survivor continuation base lane:
   continuation archive/train-eval, feature sufficiency, action-value audit,
   target dataset, scorer readiness, action-value scorer, shadow eval, tie
   collapse autopsy, wrappers, tests, and package scripts.
6. v163-v176 transition/world-model diagnostic closeout:
   tied-set expansions, preterminal target expansion, source-split scorer and
   failure autopsy, public feature/temporal probes, diagnostic portfolio,
   replay expansion, v172 target dataset, v173-v175 audits, and v176 transition
   diagnostic planner.

If reviewable staging by exact historical slice is too costly, prefer a smaller
number of lane-range commits over delaying preservation. The commit message
should explicitly state any compromise and reference this P0 inventory.

## Remaining Durability Work

Post-inventory source durability was completed after this preservation pass:
the backlog was committed on `codex/p0-durability-backlog`, pushed, and merged
to `master` as `628ff1e`.

1. Decide whether missing chunk directories are acceptable merged-evidence-only
   provenance or need regeneration.
2. Keep future Mind experiment source and digest-referenced evidence durable
   before using the remote trainer or making promotion-style claims.
