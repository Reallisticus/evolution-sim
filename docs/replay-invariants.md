# Replay Invariants

This repo treats full replay output as a compatibility contract. Refactors may
move code and remove duplicate logic, but they must not change these invariants
without an explicit contract update.

## Runtime Pin

- Python version: `3.14.3`
- Dict insertion order is part of the replay contract.
- Replay serialization must not introduce canonical key sorting.

## Full Replay Serialization

- Top-level key order is fixed:
  - `run_id`
  - `config`
  - `summary`
  - `events`
  - `viewer`
- Event order is the current emit order.
- Frame order is the current capture order.
- Species and ecotype IDs keep current assignment semantics.
- Float rounding points remain where the current builders apply them.

## Viewer Contract

- Full mode must always include:
  - `viewer.frames`
  - `viewer.map`
  - `viewer.agent_encoding`
  - `viewer.agent_catalog`
  - `viewer.species_catalog`
  - `viewer.analytics`
- `viewer.map` key order is fixed:
  - `width`
  - `height`
  - `terrain_codes`
  - `terrain_legend`
  - `hydrology_primary_legend`
  - `hydrology_support_bits`
  - `refuge_legend`
  - `hazard_legend`
  - `trophic_role_legend`
  - `meat_mode_legend`
  - `ecology_legend`
  - `terrain_counts`
  - `environment_fields`
  - `base_tile_fields`
- Legend dictionaries stay string-keyed and inserted in ascending numeric code order.
- `viewer.agent_encoding` order is fixed:
  - `agent_id`
  - `x`
  - `y`
  - `energy`
  - `energy_ratio`
  - `hydration`
  - `hydration_ratio`
  - `health`
  - `health_ratio`
  - `injury_load`
  - `age`
  - `energy_modifier`
  - `hydration_modifier`
  - `tile_vegetation`
  - `tile_recovery_debt`
  - `reproduction_ready`
  - `trophic_role`
  - `meat_mode`
  - `last_damage_source`
  - `water_access_reason`
  - `soft_refuge_reason`
  - `hydrology_support_code`
  - `refuge_score`
  - `matched_diet_ratio`
  - `ecotype_id`
  - `species_id`

## Summary Contract

- Full mode returns both shared and taxonomy-dependent summary fields.
- Shared summaries include `summary_schema_version`. The current value is
  `foundation_summary_v1`; any shared-summary key or schema change must update
  tests and documentation intentionally.
- Summary-only returns only the shared summary fields and must not be treated as
  a replay payload.
- Summary-only sets `events=None` and `viewer=None`.
- Replay taxonomy must not run in summary-only mode.
- Summary-only may skip internal replay event and tick-detail bookkeeping when
  those details are not needed for shared summary fields.
- Shared summary analytics currently include:
  - `carrying_capacity`: near-cap ticks, at-cap ticks, tick shares, and
    saturation births/deaths.
  - `resource_pressure`: plant created/removed/lost budget and energy spend by
    metabolism, movement, attack, reproduction, and signal.
  - `selection_heredity`: initial trait distributions, terminal alive trait
    distributions, and terminal-minus-initial mean deltas.
- These analytics are summary contracts, not viewer contracts. Summary-only
  runs must compute them without frame capture, viewer surface builders, replay
  taxonomy rewriting, or full replay event payloads.
