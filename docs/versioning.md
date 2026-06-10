# Versioning

This repository has two different kinds of version identifiers.

Product SemVer records the release state of the simulator as a product. The
`package.json` version follows SemVer and should describe release maturity, not
the number of historical experiments that have been run.

Historical Mind experiment IDs, artifact IDs, and schema IDs are reproducibility
identifiers. Existing IDs such as `v31`, `v61`, `v143`, and strings ending in
`_v1` or `_v2` are immutable once recorded in docs, output paths, tests, replay
contracts, or serialized artifacts. Do not rename them to match product SemVer.

`v1.0.0` is reserved for the current final product goal: a release-ready
simulator with the expected deterministic replay, Foundation behavior, viewer
contracts, and autonomous-controller readiness boundaries.

New experiment IDs should prefer descriptive identifiers instead of bare
global `vNN` counters. Use formats such as:

- `m3-YYYYMMDD-slug`
- `m3-archive-expansion-001`

If an artifact or schema needs its own contract suffix, keep that suffix scoped
to the contract, for example `mind_v3_archive_expansion_report_v1`.

## Experiment Durability

Historical IDs are useful only if the source and evidence behind them are
durable. A new Mind experiment slice should be committed as a reviewable unit:
CLI entrypoint, `mind/` implementation, tests, package script, and ledger entry
together. Do not let a new slice depend on a long-lived dirty tree.

Reports may record SHA256 digests of generated artifacts, but a digest of a
gitignored local file is provenance, not backup. If a later module fails closed
on such a digest, the artifact needs a documented durable storage path or the
module needs an explicit source-integrity override for regenerated,
metric-equivalent artifacts.

Existing vNN module names are immutable once recorded. Do not rename old files
to retrofit this policy. Apply descriptive IDs to new work after the current
backlog is made durable.
