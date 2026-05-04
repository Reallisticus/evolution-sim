from __future__ import annotations

def build_species_metric_leaderboards(
    latest_species_metrics: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        "top_species_by_realized_animal_share": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_animal_share": float(metrics["realized_animal_share"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    "reproduction_success": int(metrics["reproduction_success"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if metrics["alive_count"] > 0 or metrics["reproduction_success"] > 0
            ),
            key=lambda item: (
                -item["realized_animal_share"],
                -item["reproduction_success"],
                item["species_id"],
            ),
        )[:10],
        "top_species_by_realized_fresh_kill_share": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    "fresh_kill_gained_energy": float(metrics["fresh_kill_gained_energy"]),
                    "kills": int(metrics["kills"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if metrics["alive_count"] > 0
                or metrics["fresh_kill_gained_energy"] > 0
                or metrics["kills"] > 0
            ),
            key=lambda item: (
                -item["realized_fresh_kill_share"],
                -item["fresh_kill_gained_energy"],
                -item["kills"],
                item["species_id"],
            ),
        )[:10],
        "top_species_by_realized_carcass_share": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    "carcass_gained_energy": float(metrics["carcass_gained_energy"]),
                    "carcass_energy_consumed": float(metrics["carcass_energy_consumed"]),
                    "attack_attempts": int(metrics["attack_attempts"]),
                    "kills": int(metrics["kills"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if metrics["alive_count"] > 0
                or metrics["carcass_gained_energy"] > 0
                or metrics["attack_attempts"] > 0
            ),
            key=lambda item: (
                -item["realized_carcass_share"],
                -item["carcass_gained_energy"],
                -item["attack_attempts"],
                item["species_id"],
            ),
        )[:10],
        "top_carnivore_species": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_animal_share": float(metrics["realized_animal_share"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if int(metrics["trophic_role_occupancy"]["carnivore"]) > 0
            ),
            key=lambda item: (
                -item["alive_count"],
                -item["realized_animal_share"],
                item["species_id"],
            ),
        )[:10],
        "top_hunter_species": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "kills": int(metrics["kills"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if int(metrics["meat_mode_occupancy"]["hunter"]) > 0
            ),
            key=lambda item: (
                -item["alive_count"],
                -item["kills"],
                -item["realized_fresh_kill_share"],
                item["species_id"],
            ),
        )[:10],
        "top_scavenger_species": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    "carcass_gained_energy": float(metrics["carcass_gained_energy"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if int(metrics["meat_mode_occupancy"]["scavenger"]) > 0
            ),
            key=lambda item: (
                -item["alive_count"],
                -item["realized_carcass_share"],
                -item["carcass_gained_energy"],
                item["species_id"],
            ),
        )[:10],
    }


def build_run_top_species(
    species_registry: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "species_id": species_id,
                "label": registry["label"],
                "peak_members": registry["peak_members"],
                "alive_members": registry.get("current_members", 0),
                "lineages": sorted(registry["lineages"]),
            }
            for species_id, registry in species_registry.items()
        ),
        key=lambda item: (-item["alive_members"], -item["peak_members"], item["species_id"]),
    )[:10]


def build_taxonomy_top_species(
    species_catalog: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "species_id": int(species_id),
                "label": payload["label"],
                "peak_members": payload["peak_members"],
                "alive_members": payload.get("current_members", 0),
                "lineages": payload["lineages"],
                "origin_kind": payload["taxonomy_origin"],
                "status": payload["status"],
                "parent_species_id": payload.get("parent_species_id"),
            }
            for species_id, payload in species_catalog.items()
        ),
        key=lambda item: (-item["alive_members"], -item["peak_members"], item["species_id"]),
    )[:10]
