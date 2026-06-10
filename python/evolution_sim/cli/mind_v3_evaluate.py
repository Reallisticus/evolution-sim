from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution_sim.mind.evaluation_harness import (
    MIND_V3_EVALUATION_SCHEMA_VERSION,
    MIND_V3_REPRODUCTION_ATTRIBUTION_POLICY,
    MIND_V3_TEMPORAL_READINESS_ATTRIBUTION_POLICY,
    MIND_V3_CONTROLLED_FIXTURE_SUITE_POLICY,
    MIND_V3_CONTROLLED_FIXTURE_GATE_POLICY,
    MIND_V3_NEURAL_ANCHOR_DIAGNOSTICS_POLICY,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_RUNTIME_DIAGNOSTICS_POLICY,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_READY_CLASSIFICATION,
    MIND_V3_TRANSITION_VALUE_RUNTIME_DIAGNOSTICS_POLICY,
    MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
    MIND_V3_TRANSITION_VALUE_READY_CLASSIFICATION,
    MIND_V3_V97_PLANNER_DISTILLED_PROMOTION_POLICY,
    V97_BROAD_SEEDS,
    V97_CARRION_FIXTURE_SEEDS,
    V97_TICKS,
    V97_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
    CONTROLLED_FIXTURE_NAMES,
    BIOLOGICAL_BLOCKER_REASONS,
    _comparison_delta,
    _dominant_action_summary,
    _dominant_count_key,
    _heuristic_action_source_count,
    _int_counter,
    _json_ready,
    _mean,
    _mind_v3_policy,
    _round,
    _run_once,
    _run_world,
    _safe_path_part,
    _share,
    _write_trajectory_records,
    _trajectory_records_with_optional_metadata,
    _trajectory_safe_policy_decision_diagnostics,
    _flatten_sequence_history_shadow_diagnostics,
    _flatten_transition_value_diagnostics,
    _is_scalar_safe_diagnostic_value,
    _trajectory_output_path,
    run_mind_v3_fixture_suite,
    run_controlled_fixture_policy_suite,
    mind_v3_fixture_gate_config,
    mind_v3_fixture_gate_status,
    _fixture_gate_comparison_delta,
    _load_json_mapping,
    _load_sequence_history_shadow_scorer_artifact,
    _sequence_history_shadow_scorer_source_integrity,
    _load_transition_value_scorer_artifact,
    _transition_value_scorer_source_integrity,
    _planner_distilled_source_reload_actions_match,
    _v97_planner_distilled_promotion_report,
    _attach_v97_promotion_ledger_metrics,
    _fixture_report_by_name,
    _fixture_blocker_count,
    _per_seed_deltas,
    _append_exact_sequence_blocker,
    _append_nonnegative_blocker,
    _v97_blocker,
    _fixture_gate_metrics,
    _fixture_gate_blockers,
    _fixture_gate_blocker,
    _float_metric,
    _int,
    _mapping,
    _list_of_mappings,
    _run_fixture_once,
    _fixture_names,
    _parse_fixture_names,
    _fixture_world,
    _fixture_resources,
    _fixture_scenario_config,
    _configure_uniform_arena,
    _set_water_border,
    _set_water_tile,
    _archetype_genomes,
    _seed_archetypes,
    _add_fixture_agent,
    _aggregate_runs,
    _sequence_history_shadow_scorer_diagnostics,
    _aggregate_sequence_history_shadow_scorer_diagnostics,
    _empty_sequence_history_shadow_scorer_stats,
    _update_sequence_history_shadow_scorer_stats,
    _finalize_sequence_history_shadow_scorer_stats,
    _transition_value_scorer_diagnostics,
    _aggregate_transition_value_scorer_diagnostics,
    _empty_transition_value_scorer_stats,
    _update_transition_value_scorer_stats,
    _finalize_transition_value_scorer_stats,
    _neural_anchor_diagnostics,
    _aggregate_neural_anchor_diagnostics,
    _empty_neural_anchor_diagnostic_stats,
    _update_neural_anchor_diagnostic_stats,
    _finalize_neural_anchor_diagnostic_stats,
    _diagnostic_action,
    _diagnostic_float,
    _safe_mean_total,
    _reproduction_failure_attribution,
    _aggregate_reproduction_failure_attribution,
    _temporal_readiness_attribution,
    _aggregate_temporal_readiness_attribution,
    _core_readiness_blockers,
    _primary_core_readiness_blocker,
    _core_fields,
    _core_blocker_name,
    _sum_grouped_int_counts,
    _terminal_energy_readiness,
    _balanced_readiness_score,
    _float_value,
    _nonnegative_int,
    _nonnegative_float,
    _founder_template_count,
    _founder_template_specialization_profile_counts,
    _parse_seeds,
)
from evolution_sim.mind.evolution import (
    load_mind_v3_founder_template,
    require_mind_v3_founder_template_promotion_eligible,
)
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_MODEL_TYPE,
    load_mind_v3_neural_artifact,
)
from evolution_sim.mind.v3_planner_distilled import (
    load_mind_v3_planner_distilled_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Mind v3 autonomous evolution against the heuristic "
            "baseline without heuristic fallback inside the v3 policy."
        )
    )
    parser.add_argument(
        "--seeds",
        default="5,13,19,29",
        help="Comma-separated deterministic seed list.",
    )
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--founder-template",
        type=Path,
        help=(
            "Optional raw Mind v3 controller metadata or evolution-search "
            "report whose best candidate initializes founders."
        ),
    )
    parser.add_argument(
        "--neural-artifact",
        type=Path,
        help=(
            "Optional frozen Mind v3 neural policy artifact. The artifact "
            "uses ecological policy inputs and does not mutate neural weights "
            "inside a run."
        ),
    )
    parser.add_argument(
        "--anchored-neural-artifact",
        type=Path,
        help=(
            "Optional current anchored neural artifact to evaluate beside the "
            "primary --neural-artifact. This is intended for comparing an "
            "experimental direct artifact against the existing linear-anchor "
            "neural path in the same report."
        ),
    )
    parser.add_argument(
        "--planner-distilled-artifact",
        type=Path,
        help=(
            "Optional frozen v96 planner-distilled Mind v3 action scorer "
            "artifact or v96 report containing distilled_artifact. This is a "
            "runtime-feasibility artifact and is evaluated without planner "
            "outcome tables or global assignment."
        ),
    )
    parser.add_argument(
        "--sequence-history-shadow-scorer",
        type=Path,
        help=(
            "Optional diagnostics-only v139 sequence-history shadow scorer "
            "artifact. The scorer logs what it would predict during Mind v3 "
            "evaluation without changing requested actions."
        ),
    )
    parser.add_argument(
        "--sequence-history-action-override",
        action="store_true",
        help=(
            "Explicitly opt into v141 sequence-history action selection. "
            "Requires --sequence-history-shadow-scorer with a full v139 report "
            "whose source-integrity and support floors passed."
        ),
    )
    parser.add_argument(
        "--transition-value-scorer",
        type=Path,
        help=(
            "Optional v142 public transition-value scorer artifact. The "
            "scorer logs transition utility predictions unless the explicit "
            "--transition-value-action-override flag is also set."
        ),
    )
    parser.add_argument(
        "--transition-value-action-override",
        action="store_true",
        help=(
            "Explicitly opt into v142 transition-value action selection. "
            "Requires --transition-value-scorer with a full v142 report whose "
            "source-integrity, support floors, and JSON score roundtrip passed."
        ),
    )
    parser.add_argument(
        "--compare-linear-baseline",
        action="store_true",
        help=(
            "When --neural-artifact or --planner-distilled-artifact is set, "
            "also evaluate the current linear Mind v3 controller on the same "
            "seeds/ticks and report primary-minus-linear deltas."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-autonomous-evolution-report.json"),
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        help=(
            "Optional directory for per-run trajectory JSONL.gz files. "
            "This preserves broad and controlled-fixture runs for horizon "
            "label generation without building full replay payloads."
        ),
    )
    parser.add_argument(
        "--fixture-suite",
        choices=["none", "basic"],
        default="none",
        help=(
            "Optionally run controlled ecology fixtures through the same "
            "simulator and policies. Defaults to disabled."
        ),
    )
    parser.add_argument(
        "--fixture-seeds",
        help=(
            "Comma-separated deterministic seed list for --fixture-suite. "
            "Defaults to --seeds."
        ),
    )
    parser.add_argument(
        "--fixture-names",
        help=(
            "Optional comma-separated fixture subset for --fixture-suite. "
            "Defaults to all fixtures in the selected suite."
        ),
    )
    parser.add_argument(
        "--fixture-ticks",
        type=int,
        help="Tick horizon for --fixture-suite. Defaults to --ticks.",
    )
    parser.add_argument(
        "--fixture-min-alive",
        type=float,
        default=1.0,
        help="Minimum v3 alive-agent mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-births",
        type=float,
        default=0.0,
        help="Minimum v3 births mean required for every enabled fixture.",
    )
    parser.add_argument(
        "--fixture-min-mixed-stable-births",
        type=float,
        default=0.0,
        help="Minimum v3 births mean required specifically on mixed_stable.",
    )
    parser.add_argument(
        "--fixture-min-energy-viability",
        type=float,
        default=0.0,
        help="Minimum terminal energy viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-hydration-viability",
        type=float,
        default=0.0,
        help="Minimum terminal hydration viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-health-viability",
        type=float,
        default=0.0,
        help="Minimum terminal health viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-matched-diet-viability",
        type=float,
        default=0.0,
        help="Minimum terminal matched-diet viability required for every fixture.",
    )
    parser.add_argument(
        "--fixture-min-biologically-ready",
        type=float,
        default=0.0,
        help=(
            "Minimum biologically-ready terminal agent mean required for "
            "every fixture."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    founder_template = (
        load_mind_v3_founder_template(args.founder_template)
        if args.founder_template is not None
        else None
    )
    if founder_template is not None:
        try:
            require_mind_v3_founder_template_promotion_eligible(founder_template)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    neural_artifact = (
        load_mind_v3_neural_artifact(args.neural_artifact)
        if args.neural_artifact is not None
        else None
    )
    planner_distilled_source_payload = (
        _load_json_mapping(args.planner_distilled_artifact)
        if args.planner_distilled_artifact is not None
        else None
    )
    planner_distilled_artifact = (
        load_mind_v3_planner_distilled_artifact(
            planner_distilled_source_payload
            if planner_distilled_source_payload is not None
            else args.planner_distilled_artifact
        )
        if args.planner_distilled_artifact is not None
        else None
    )
    anchored_neural_artifact = (
        load_mind_v3_neural_artifact(args.anchored_neural_artifact)
        if args.anchored_neural_artifact is not None
        else None
    )
    if args.sequence_history_action_override and args.sequence_history_shadow_scorer is None:
        raise SystemExit(
            "--sequence-history-action-override requires "
            "--sequence-history-shadow-scorer"
        )
    if args.transition_value_action_override and args.transition_value_scorer is None:
        raise SystemExit(
            "--transition-value-action-override requires --transition-value-scorer"
        )
    if args.sequence_history_action_override and args.transition_value_action_override:
        raise SystemExit(
            "--sequence-history-action-override and "
            "--transition-value-action-override are mutually exclusive"
        )
    sequence_history_shadow_payload = (
        _load_json_mapping(args.sequence_history_shadow_scorer)
        if args.sequence_history_shadow_scorer is not None
        else None
    )
    sequence_history_shadow_source_integrity = (
        _sequence_history_shadow_scorer_source_integrity(
            sequence_history_shadow_payload
        )
        if sequence_history_shadow_payload is not None
        else {
            "passed": False,
            "failures": ["missing_sequence_history_shadow_scorer"],
        }
    )
    if (
        args.sequence_history_action_override
        and sequence_history_shadow_source_integrity.get("passed") is not True
    ):
        raise SystemExit(
            "--sequence-history-action-override requires a valid full v139 "
            "sequence-history scorer report; failures="
            f"{sequence_history_shadow_source_integrity.get('failures')}"
        )
    sequence_history_shadow_scorer = (
        _load_sequence_history_shadow_scorer_artifact(
            sequence_history_shadow_payload
        )
        if sequence_history_shadow_payload is not None
        else None
    )
    transition_value_payload = (
        _load_json_mapping(args.transition_value_scorer)
        if args.transition_value_scorer is not None
        else None
    )
    transition_value_source_integrity = (
        _transition_value_scorer_source_integrity(transition_value_payload)
        if transition_value_payload is not None
        else {
            "passed": False,
            "failures": ["missing_transition_value_scorer"],
        }
    )
    if (
        args.transition_value_action_override
        and transition_value_source_integrity.get("passed") is not True
    ):
        raise SystemExit(
            "--transition-value-action-override requires a valid full v142 "
            "transition-value scorer report; failures="
            f"{transition_value_source_integrity.get('failures')}"
        )
    transition_value_scorer = (
        _load_transition_value_scorer_artifact(transition_value_payload)
        if transition_value_payload is not None
        else None
    )
    if planner_distilled_artifact is not None and neural_artifact is not None:
        raise SystemExit(
            "--planner-distilled-artifact and --neural-artifact are mutually exclusive"
        )
    if (
        planner_distilled_artifact is not None
        and anchored_neural_artifact is not None
    ):
        raise SystemExit(
            "--planner-distilled-artifact cannot be combined with "
            "--anchored-neural-artifact"
        )
    if (
        anchored_neural_artifact is not None
        and anchored_neural_artifact.get("model_type") != MIND_V3_NEURAL_MODEL_TYPE
    ):
        raise SystemExit(
            "--anchored-neural-artifact must use the current anchored neural "
            f"model_type {MIND_V3_NEURAL_MODEL_TYPE}"
        )
    if (
        args.compare_linear_baseline
        and neural_artifact is None
        and planner_distilled_artifact is None
    ):
        raise SystemExit(
            "--compare-linear-baseline requires --neural-artifact or "
            "--planner-distilled-artifact"
        )
    heuristic_runs = [
        _run_once(
            seed=seed,
            ticks=args.ticks,
            policy=None,
            trajectory_output_path=_trajectory_output_path(
                args.trajectory_output_dir,
                "open",
                "heuristic",
                seed,
                args.ticks,
            ),
            trajectory_split_id="mind_v3_evaluate_open_heuristic",
        )
        for seed in seeds
    ]
    linear_runs = (
        [
            _run_once(
                seed=seed,
                ticks=args.ticks,
                policy=_mind_v3_policy(
                    seed=seed,
                    founder_template=founder_template,
                    sequence_history_shadow_scorer=(
                        sequence_history_shadow_scorer
                    ),
                    sequence_history_action_override=(
                        args.sequence_history_action_override
                    ),
                    sequence_history_action_override_source_integrity_passed=(
                        sequence_history_shadow_source_integrity.get("passed") is True
                    ),
                    transition_value_scorer=transition_value_scorer,
                    transition_value_action_override=(
                        args.transition_value_action_override
                    ),
                    transition_value_action_override_source_integrity_passed=(
                        transition_value_source_integrity.get("passed") is True
                    ),
                ),
                trajectory_output_path=_trajectory_output_path(
                    args.trajectory_output_dir,
                    "open",
                    "mind_v3_linear",
                    seed,
                    args.ticks,
                ),
                trajectory_split_id="mind_v3_evaluate_open_linear",
            )
            for seed in seeds
        ]
        if args.compare_linear_baseline
        else None
    )
    mind_v3_runs = [
        _run_once(
            seed=seed,
            ticks=args.ticks,
            policy=_mind_v3_policy(
                seed=seed,
                founder_template=founder_template,
                neural_artifact=neural_artifact,
                planner_distilled_artifact=planner_distilled_artifact,
                sequence_history_shadow_scorer=sequence_history_shadow_scorer,
                sequence_history_action_override=(
                    args.sequence_history_action_override
                ),
                sequence_history_action_override_source_integrity_passed=(
                    sequence_history_shadow_source_integrity.get("passed") is True
                ),
                transition_value_scorer=transition_value_scorer,
                transition_value_action_override=(
                    args.transition_value_action_override
                ),
                transition_value_action_override_source_integrity_passed=(
                    transition_value_source_integrity.get("passed") is True
                ),
            ),
            trajectory_output_path=_trajectory_output_path(
                args.trajectory_output_dir,
                "open",
                "mind_v3",
                seed,
                args.ticks,
            ),
            trajectory_split_id="mind_v3_evaluate_open_mind_v3",
        )
        for seed in seeds
    ]
    anchored_neural_runs = (
        [
            _run_once(
                seed=seed,
                ticks=args.ticks,
                policy=_mind_v3_policy(
                    seed=seed,
                    founder_template=founder_template,
                    neural_artifact=anchored_neural_artifact,
                    sequence_history_shadow_scorer=(
                        sequence_history_shadow_scorer
                    ),
                    sequence_history_action_override=(
                        args.sequence_history_action_override
                    ),
                    sequence_history_action_override_source_integrity_passed=(
                        sequence_history_shadow_source_integrity.get("passed") is True
                    ),
                    transition_value_scorer=transition_value_scorer,
                    transition_value_action_override=(
                        args.transition_value_action_override
                    ),
                    transition_value_action_override_source_integrity_passed=(
                        transition_value_source_integrity.get("passed") is True
                    ),
                ),
                trajectory_output_path=_trajectory_output_path(
                    args.trajectory_output_dir,
                    "open",
                    "mind_v3_anchored_neural",
                    seed,
                    args.ticks,
                ),
                trajectory_split_id="mind_v3_evaluate_open_anchored_neural",
            )
            for seed in seeds
        ]
        if anchored_neural_artifact is not None
        else None
    )
    report = {
        "schema_version": MIND_V3_EVALUATION_SCHEMA_VERSION,
        "policy": {
            "policy_id": "mind_v3_autonomous_evolution_policy",
            "policy_version": "mind_v3_autonomous_evolution_policy_v1",
            "heuristic_free": True,
            "founder_template_source": (
                str(args.founder_template)
                if args.founder_template is not None
                else None
            ),
            "founder_template_count": _founder_template_count(founder_template),
            "founder_template_specialization_profile_counts": (
                _founder_template_specialization_profile_counts(founder_template)
            ),
            "neural_artifact_source": (
                str(args.neural_artifact)
                if args.neural_artifact is not None
                else None
            ),
            "neural_artifact_schema_version": (
                neural_artifact.get("schema_version")
                if isinstance(neural_artifact, dict)
                else None
            ),
            "neural_model_type": (
                neural_artifact.get("model_type")
                if isinstance(neural_artifact, dict)
                else None
            ),
            "neural_input_policy": (
                neural_artifact.get("input_policy")
                if isinstance(neural_artifact, dict)
                else None
            ),
            "anchored_neural_artifact_source": (
                str(args.anchored_neural_artifact)
                if args.anchored_neural_artifact is not None
                else None
            ),
            "anchored_neural_model_type": (
                anchored_neural_artifact.get("model_type")
                if isinstance(anchored_neural_artifact, dict)
                else None
            ),
            "planner_distilled_artifact_source": (
                str(args.planner_distilled_artifact)
                if args.planner_distilled_artifact is not None
                else None
            ),
            "planner_distilled_artifact_schema_version": (
                planner_distilled_artifact.get("schema_version")
                if isinstance(planner_distilled_artifact, dict)
                else None
            ),
            "planner_distilled_model_type": (
                planner_distilled_artifact.get("model_type")
                if isinstance(planner_distilled_artifact, dict)
                else None
            ),
            "planner_distilled_source_reload_actions_match": (
                _planner_distilled_source_reload_actions_match(
                    planner_distilled_source_payload
                )
            ),
            "sequence_history_shadow_scorer_source": (
                str(args.sequence_history_shadow_scorer)
                if args.sequence_history_shadow_scorer is not None
                else None
            ),
            "sequence_history_shadow_scorer_runtime_effect": (
                "explicit_opt_in_action_override"
                if args.sequence_history_action_override
                else "diagnostics_only_no_action_selection_change"
                if sequence_history_shadow_scorer is not None
                else None
            ),
            "sequence_history_action_override_enabled": bool(
                args.sequence_history_action_override
            ),
            "sequence_history_shadow_scorer_source_integrity": (
                sequence_history_shadow_source_integrity
            ),
            "transition_value_scorer_source": (
                str(args.transition_value_scorer)
                if args.transition_value_scorer is not None
                else None
            ),
            "transition_value_scorer_runtime_effect": (
                "explicit_opt_in_action_override"
                if args.transition_value_action_override
                else "diagnostics_only_no_action_selection_change"
                if transition_value_scorer is not None
                else None
            ),
            "transition_value_action_override_enabled": bool(
                args.transition_value_action_override
            ),
            "transition_value_scorer_source_integrity": (
                transition_value_source_integrity
            ),
            "linear_baseline_compared": bool(args.compare_linear_baseline),
        },
        "comparison": {
            "heuristic": {
                "runs": heuristic_runs,
                "aggregate": _aggregate_runs(heuristic_runs),
            },
            "mind_v3": {
                "runs": mind_v3_runs,
                "aggregate": _aggregate_runs(mind_v3_runs),
            },
        },
    }
    if linear_runs is not None:
        report["comparison"]["mind_v3_linear"] = {
            "runs": linear_runs,
            "aggregate": _aggregate_runs(linear_runs),
        }
    if anchored_neural_runs is not None:
        report["comparison"]["mind_v3_anchored_neural"] = {
            "runs": anchored_neural_runs,
            "aggregate": _aggregate_runs(anchored_neural_runs),
        }
    report["comparison"]["delta"] = _comparison_delta(
        heuristic=report["comparison"]["heuristic"]["aggregate"],
        mind_v3=report["comparison"]["mind_v3"]["aggregate"],
    )
    if linear_runs is not None:
        primary_vs_linear_delta = _comparison_delta(
            heuristic=report["comparison"]["mind_v3_linear"]["aggregate"],
            mind_v3=report["comparison"]["mind_v3"]["aggregate"],
        )
        if neural_artifact is not None:
            report["comparison"]["neural_vs_linear_delta"] = (
                primary_vs_linear_delta
            )
        if planner_distilled_artifact is not None:
            report["comparison"]["planner_vs_linear_delta"] = (
                primary_vs_linear_delta
            )
    if anchored_neural_runs is not None:
        report["comparison"]["primary_vs_anchored_neural_delta"] = (
            _comparison_delta(
                heuristic=report["comparison"]["mind_v3_anchored_neural"][
                    "aggregate"
                ],
                mind_v3=report["comparison"]["mind_v3"]["aggregate"],
            )
        )
    if args.fixture_suite != "none":
        fixture_seeds = (
            _parse_seeds(args.fixture_seeds)
            if args.fixture_seeds is not None
            else seeds
        )
        fixture_ticks = (
            int(args.fixture_ticks)
            if args.fixture_ticks is not None
            else int(args.ticks)
        )
        fixture_names = _parse_fixture_names(
            args.fixture_names,
            suite=args.fixture_suite,
        )
        fixture_config = mind_v3_fixture_gate_config(
            suite=args.fixture_suite,
            seeds=fixture_seeds,
            ticks=fixture_ticks,
            min_alive=float(args.fixture_min_alive),
            min_births=float(args.fixture_min_births),
            min_mixed_stable_births=float(args.fixture_min_mixed_stable_births),
            min_energy_viability=float(args.fixture_min_energy_viability),
            min_hydration_viability=float(args.fixture_min_hydration_viability),
            min_health_viability=float(args.fixture_min_health_viability),
            min_matched_diet_viability=float(args.fixture_min_matched_diet_viability),
            min_biologically_ready=float(args.fixture_min_biologically_ready),
        )
        report["fixture_suite"] = run_mind_v3_fixture_suite(
            suite=args.fixture_suite,
            fixture_names=fixture_names,
            seeds=fixture_seeds,
            ticks=fixture_ticks,
            founder_template=founder_template,
            neural_artifact=neural_artifact,
            planner_distilled_artifact=planner_distilled_artifact,
            sequence_history_shadow_scorer=sequence_history_shadow_scorer,
            sequence_history_action_override=args.sequence_history_action_override,
            sequence_history_action_override_source_integrity_passed=(
                sequence_history_shadow_source_integrity.get("passed") is True
            ),
            transition_value_scorer=transition_value_scorer,
            transition_value_action_override=args.transition_value_action_override,
            transition_value_action_override_source_integrity_passed=(
                transition_value_source_integrity.get("passed") is True
            ),
            trajectory_output_dir=args.trajectory_output_dir,
            trajectory_prefix="fixture",
        )
        report["fixture_gate"] = mind_v3_fixture_gate_status(
            fixture_suite=report["fixture_suite"],
            fixture_config=fixture_config,
        )
        if args.compare_linear_baseline:
            report["linear_baseline_fixture_suite"] = run_mind_v3_fixture_suite(
                suite=args.fixture_suite,
                fixture_names=fixture_names,
                seeds=fixture_seeds,
                ticks=fixture_ticks,
                founder_template=founder_template,
                sequence_history_shadow_scorer=sequence_history_shadow_scorer,
                sequence_history_action_override=(
                    args.sequence_history_action_override
                ),
                sequence_history_action_override_source_integrity_passed=(
                    sequence_history_shadow_source_integrity.get("passed") is True
                ),
                transition_value_scorer=transition_value_scorer,
                transition_value_action_override=(
                    args.transition_value_action_override
                ),
                transition_value_action_override_source_integrity_passed=(
                    transition_value_source_integrity.get("passed") is True
                ),
                trajectory_output_dir=args.trajectory_output_dir,
                trajectory_prefix="fixture_linear_baseline",
            )
            report["linear_baseline_fixture_gate"] = mind_v3_fixture_gate_status(
                fixture_suite=report["linear_baseline_fixture_suite"],
                fixture_config=fixture_config,
            )
            primary_vs_linear_fixture_delta = _fixture_gate_comparison_delta(
                linear_gate=report["linear_baseline_fixture_gate"],
                neural_gate=report["fixture_gate"],
            )
            if neural_artifact is not None:
                report["neural_vs_linear_fixture_delta"] = (
                    primary_vs_linear_fixture_delta
                )
            if planner_distilled_artifact is not None:
                report["planner_vs_linear_fixture_delta"] = (
                    primary_vs_linear_fixture_delta
                )
        if anchored_neural_artifact is not None:
            report["anchored_neural_fixture_suite"] = run_mind_v3_fixture_suite(
                suite=args.fixture_suite,
                fixture_names=fixture_names,
                seeds=fixture_seeds,
                ticks=fixture_ticks,
                founder_template=founder_template,
                neural_artifact=anchored_neural_artifact,
                sequence_history_shadow_scorer=sequence_history_shadow_scorer,
                sequence_history_action_override=args.sequence_history_action_override,
                sequence_history_action_override_source_integrity_passed=(
                    sequence_history_shadow_source_integrity.get("passed") is True
                ),
                transition_value_scorer=transition_value_scorer,
                transition_value_action_override=args.transition_value_action_override,
                transition_value_action_override_source_integrity_passed=(
                    transition_value_source_integrity.get("passed") is True
                ),
                trajectory_output_dir=args.trajectory_output_dir,
                trajectory_prefix="fixture_anchored_neural",
            )
            report["anchored_neural_fixture_gate"] = mind_v3_fixture_gate_status(
                fixture_suite=report["anchored_neural_fixture_suite"],
                fixture_config=fixture_config,
            )
            report["primary_vs_anchored_neural_fixture_delta"] = (
                _fixture_gate_comparison_delta(
                    linear_gate=report["anchored_neural_fixture_gate"],
                    neural_gate=report["fixture_gate"],
                )
            )
    if planner_distilled_artifact is not None:
        report["v97_planner_distilled_promotion"] = (
            _v97_planner_distilled_promotion_report(report)
        )
        _attach_v97_promotion_ledger_metrics(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"mind_v3_report={args.output}")
    print(
        "mind_v3_heuristic_action_source_count="
        f"{report['comparison']['mind_v3']['aggregate']['heuristic_action_source_count']}"
    )
    print(
        "mind_v3_alive_agents_mean="
        f"{report['comparison']['mind_v3']['aggregate']['alive_agents_mean']}"
    )
    print(
        "mind_v3_births_mean="
        f"{report['comparison']['mind_v3']['aggregate']['births_mean']}"
    )
    if "fixture_suite" in report:
        print(
            "mind_v3_fixture_count="
            f"{len(report['fixture_suite']['fixtures'])}"
        )
    if "fixture_gate" in report:
        print(f"mind_v3_fixture_gate_passed={report['fixture_gate']['passed']}")
    if "comparison" in report and "neural_vs_linear_delta" in report["comparison"]:
        delta = report["comparison"]["neural_vs_linear_delta"]
        print(f"mind_v3_neural_minus_linear_alive={delta['alive_agents_mean']}")
        print(f"mind_v3_neural_minus_linear_births={delta['births_mean']}")
    if "comparison" in report and "planner_vs_linear_delta" in report["comparison"]:
        delta = report["comparison"]["planner_vs_linear_delta"]
        print(f"mind_v3_planner_minus_linear_alive={delta['alive_agents_mean']}")
        print(f"mind_v3_planner_minus_linear_births={delta['births_mean']}")
    if "v97_planner_distilled_promotion" in report:
        promotion = report["v97_planner_distilled_promotion"]
        print(
            "mind_v3_v97_promotion_candidate_passed="
            f"{promotion['promotion_candidate_passed']}"
        )


if __name__ == "__main__":
    main()
