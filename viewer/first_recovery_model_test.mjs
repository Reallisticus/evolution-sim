import assert from "node:assert/strict";
import {
  buildFirstRecoveryInspectorModel,
  classificationChainModel,
  diagnosticsOnlyFlagModel,
  sourceIntegrityModel,
  topBranchExamples,
} from "./first_recovery_model.mjs";

const bundle = {
  schema_version: "mind_v3_first_recovery_inspector_bundle_v1",
  contract: {
    diagnostics_only: true,
    runtime_policy_effect: "none",
    runtime_policy_implemented: false,
    trained_artifact_effect: "none",
    training_executed: false,
    bundle_is_replay_contract: false,
    bundle_is_training_manifest: false,
    v113_readiness_rerun_allowed: false,
    downstream_shadow_scorer_allowed: false,
  },
  boundary_labels: {
    audit_metadata_not_trainable_input: true,
    bundle_is_not_replay_contract: true,
    bundle_is_not_training_manifest: true,
  },
  source_integrity: {
    passed: true,
    failures: [],
    archive_row_count: 530,
    manifest_row_count: 106,
    manifest_unique_branch_count: 106,
    manifest_digest_matches_v119: true,
    manifest_digest_matches_v120: true,
    manifest_digest_matches_v121: true,
  },
  classification_chain: {
    v116: { primary: "stay_oracle_dominance_detected", labels: ["readiness_rerun_blocked"] },
    v117: { primary: "tie_break_artifact_likely", labels: ["readiness_rerun_blocked"] },
    v118: { primary: "tie_aware_repair_clears_action_collapse", labels: ["readiness_rerun_blocked"] },
    v119: { primary: "repaired_label_contract_support_limited", labels: ["readiness_rerun_blocked"] },
    v120: { primary: "split_support_feasibility_limited_by_rare_actions", labels: ["readiness_rerun_blocked"] },
    v121: { primary: "rare_action_coverage_not_available_in_existing_archive", labels: ["readiness_rerun_blocked"] },
  },
  blocker_summary: {
    v118_repaired_distribution: {
      repaired_action_counts: { stay: 16, eat: 17, attack_east: 3 },
    },
    v120_rare_action_support_limitation: {
      rare_action_additional_needed: { attack_east: 1, attack_west: 1 },
    },
    v121_no_recoverable_rare_attack_candidates: {
      valid_candidate_counts: { attack_east: 0, attack_west: 0 },
    },
  },
  branch_rows: {
    b002: {
      branch_id: "b002",
      current_oracle_action: "eat",
      repaired_action: "eat",
      changed: false,
      unique_objective_best: true,
      objective_equivalence_verified: true,
      selected_resolution_legal: true,
      legal_tied_candidate_actions: ["eat"],
      trainable_public_input: { clean: true },
    },
    b001: {
      branch_id: "b001",
      current_oracle_action: "stay",
      repaired_action: "attack_east",
      changed: true,
      unique_objective_best: false,
      objective_equivalence_verified: true,
      selected_resolution_legal: true,
      legal_tied_candidate_actions: ["attack_east", "stay"],
      trainable_public_input: { clean: true },
    },
  },
  recommendation: {
    v113_readiness_rerun_allowed: false,
    downstream_shadow_scorer_allowed: false,
  },
};

assert.deepEqual(sourceIntegrityModel(bundle.source_integrity), {
  passed: true,
  failures: [],
  archiveRowCount: 530,
  manifestRowCount: 106,
  manifestUniqueBranchCount: 106,
  manifestDigestMatchesV119: true,
  manifestDigestMatchesV120: true,
  manifestDigestMatchesV121: true,
});

assert.equal(classificationChainModel(bundle.classification_chain).at(-1).version, "v121");
assert.equal(
  classificationChainModel(bundle.classification_chain).at(-1).primary,
  "rare_action_coverage_not_available_in_existing_archive",
);

const flags = diagnosticsOnlyFlagModel(bundle);
assert.equal(flags.diagnosticsOnly, true);
assert.equal(flags.noRuntimePolicy, true);
assert.equal(flags.noTraining, true);
assert.equal(flags.readinessBlocked, true);
assert.equal(flags.shadowScorerBlocked, true);
assert.equal(flags.notReplayContract, true);
assert.equal(flags.notTrainingManifest, true);
assert.equal(flags.auditMetadataNotTrainable, true);

const examples = topBranchExamples(bundle.branch_rows, 1);
assert.equal(examples.length, 1);
assert.equal(examples[0].branchId, "b001");
assert.equal(examples[0].repairedAction, "attack_east");
assert.equal(examples[0].objectiveEquivalenceVerified, true);
assert.equal(examples[0].trainablePublicInputClean, true);

const model = buildFirstRecoveryInspectorModel(bundle);
assert.equal(model.schemaVersion, "mind_v3_first_recovery_inspector_bundle_v1");
assert.equal(model.sourceIntegrity.passed, true);
assert.deepEqual(model.repairedActionCounts, { attack_east: 3, eat: 17, stay: 16 });
assert.deepEqual(model.rareSupportGaps, { attack_east: 1, attack_west: 1 });
assert.deepEqual(model.candidateCounts, { attack_east: 0, attack_west: 0 });
assert.equal(model.branchCount, 2);
assert(model.summary.includes("readiness and shadow scoring blocked"));

const fallback = buildFirstRecoveryInspectorModel({ branch_rows: bundle.branch_rows });
assert.deepEqual(fallback.repairedActionCounts, { attack_east: 1, eat: 1 });
assert.equal(fallback.diagnosticsOnlyFlags.diagnosticsOnly, false);

const malformedCountsBundle = {
  ...bundle,
  blocker_summary: {
    v118_repaired_distribution: {
      repaired_action_counts: "not-an-object",
    },
    v120_rare_action_support_limitation: {
      rare_action_additional_needed: "not-an-object",
    },
    v121_no_recoverable_rare_attack_candidates: {
      valid_candidate_counts: ["not", "an", "object"],
    },
  },
};
const malformedCountsModel = buildFirstRecoveryInspectorModel(malformedCountsBundle);
assert.deepEqual(malformedCountsModel.repairedActionCounts, { attack_east: 1, eat: 1 });
assert.deepEqual(malformedCountsModel.rareSupportGaps, {});
assert.deepEqual(malformedCountsModel.candidateCounts, {});
assert(malformedCountsModel.summary.includes("rare_gaps=none"));
assert(malformedCountsModel.summary.includes("candidates=none"));

console.log("first_recovery_model_test_ok");
