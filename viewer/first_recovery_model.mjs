export const FIRST_RECOVERY_CLASSIFICATION_ORDER = [
  "v115",
  "v116",
  "v117",
  "v118",
  "v119",
  "v120",
  "v121",
];

export function buildFirstRecoveryInspectorModel(bundle, options = {}) {
  const payload = bundle && typeof bundle === "object" ? bundle : {};
  const branchRows = normalizeBranchRows(payload.branch_rows);
  const blockerSummary = objectValue(payload.blocker_summary);
  const sourceIntegrity = sourceIntegrityModel(payload.source_integrity);
  const classificationChain = classificationChainModel(payload.classification_chain);
  const repairedActionCountPayload = objectValue(
    blockerSummary.v118_repaired_distribution,
  ).repaired_action_counts;
  const repairedActionCounts = isObjectRecord(repairedActionCountPayload)
    ? repairedActionCountPayload
    : actionCountsFromBranches(branchRows);
  const rareSupportGaps = sortedObject(
    objectValue(blockerSummary.v120_rare_action_support_limitation).rare_action_additional_needed,
  );
  const candidateCounts = sortedObject(
    objectValue(blockerSummary.v121_no_recoverable_rare_attack_candidates).valid_candidate_counts,
  );
  const normalizedRepairedActionCounts = sortedObject(repairedActionCounts);
  const diagnosticsOnlyFlags = diagnosticsOnlyFlagModel(payload);
  return {
    schemaVersion: payload.schema_version ?? null,
    sourceIntegrity,
    classificationChain,
    repairedActionCounts: normalizedRepairedActionCounts,
    rareSupportGaps,
    candidateCounts,
    topBranchExamples: topBranchExamples(branchRows, options.exampleLimit ?? 8),
    diagnosticsOnlyFlags,
    branchCount: branchRows.length,
    summary: summaryText({
      sourceIntegrity,
      classificationChain,
      rareSupportGaps,
      candidateCounts,
      diagnosticsOnlyFlags,
    }),
  };
}

export function sourceIntegrityModel(sourceIntegrity) {
  const source = objectValue(sourceIntegrity);
  const failures = Array.isArray(source.failures) ? source.failures.map(String) : [];
  return {
    passed: source.passed === true,
    failures,
    archiveRowCount: numberOrZero(source.archive_row_count),
    manifestRowCount: numberOrZero(source.manifest_row_count),
    manifestUniqueBranchCount: numberOrZero(source.manifest_unique_branch_count),
    manifestDigestMatchesV119: source.manifest_digest_matches_v119 === true,
    manifestDigestMatchesV120: source.manifest_digest_matches_v120 === true,
    manifestDigestMatchesV121: source.manifest_digest_matches_v121 === true,
  };
}

export function classificationChainModel(classificationChain) {
  const chain = objectValue(classificationChain);
  return FIRST_RECOVERY_CLASSIFICATION_ORDER.filter((version) => chain[version]).map((version) => {
    const entry = objectValue(chain[version]);
    return {
      version,
      primary: entry.primary ?? null,
      labels: Array.isArray(entry.labels) ? entry.labels.map(String) : [],
      nextStep: entry.next_step ?? null,
    };
  });
}

export function diagnosticsOnlyFlagModel(bundle) {
  const payload = bundle && typeof bundle === "object" ? bundle : {};
  const contract = objectValue(payload.contract);
  const boundaries = objectValue(payload.boundary_labels);
  const recommendation = objectValue(payload.recommendation);
  return {
    diagnosticsOnly: contract.diagnostics_only === true,
    noRuntimePolicy: contract.runtime_policy_effect === "none" && contract.runtime_policy_implemented === false,
    noTraining: contract.trained_artifact_effect === "none" && contract.training_executed === false,
    readinessBlocked:
      contract.v113_readiness_rerun_allowed === false &&
      recommendation.v113_readiness_rerun_allowed === false,
    shadowScorerBlocked:
      contract.downstream_shadow_scorer_allowed === false &&
      recommendation.downstream_shadow_scorer_allowed === false,
    notReplayContract:
      contract.bundle_is_replay_contract === false &&
      boundaries.bundle_is_not_replay_contract === true,
    notTrainingManifest:
      contract.bundle_is_training_manifest === false &&
      boundaries.bundle_is_not_training_manifest === true,
    auditMetadataNotTrainable: boundaries.audit_metadata_not_trainable_input === true,
  };
}

export function topBranchExamples(branchRows, limit = 8) {
  const rows = Array.isArray(branchRows) ? branchRows : normalizeBranchRows(branchRows);
  const max = Math.max(1, Number.isFinite(Number(limit)) ? Number(limit) : 8);
  return [...rows]
    .sort((left, right) => branchSortKey(left).localeCompare(branchSortKey(right)))
    .slice(0, max)
    .map((row) => ({
      branchId: row.branch_id ?? null,
      currentOracleAction: row.current_oracle_action ?? null,
      repairedAction: row.repaired_action ?? null,
      changed: row.changed === true,
      uniqueObjectiveBest: row.unique_objective_best === true,
      objectiveEquivalenceVerified: row.objective_equivalence_verified === true,
      selectedResolutionLegal: row.selected_resolution_legal === true,
      legalTiedCandidateActions: Array.isArray(row.legal_tied_candidate_actions)
        ? row.legal_tied_candidate_actions.map(String)
        : [],
      trainablePublicInputClean: objectValue(row.trainable_public_input).clean === true,
    }));
}

function normalizeBranchRows(branchRows) {
  if (Array.isArray(branchRows)) return branchRows.filter((row) => row && typeof row === "object");
  const rows = objectValue(branchRows);
  return Object.values(rows).filter((row) => row && typeof row === "object");
}

function actionCountsFromBranches(branchRows) {
  const counts = {};
  for (const row of branchRows) {
    const action = row?.repaired_action;
    if (typeof action !== "string" || action.length === 0) continue;
    counts[action] = (counts[action] ?? 0) + 1;
  }
  return counts;
}

function branchSortKey(row) {
  const changedRank = row?.changed === true ? "0" : "1";
  const action = String(row?.repaired_action ?? "");
  const branchId = String(row?.branch_id ?? "");
  return `${changedRank}:${action}:${branchId}`;
}

function summaryText({ sourceIntegrity, classificationChain, rareSupportGaps, candidateCounts, diagnosticsOnlyFlags }) {
  const latest = classificationChain[classificationChain.length - 1]?.primary ?? "unknown";
  const rareActions = Object.entries(rareSupportGaps)
    .map(([action, count]) => `${action}:${count}`)
    .join(", ");
  const candidates = Object.entries(candidateCounts)
    .map(([action, count]) => `${action}:${count}`)
    .join(", ");
  const boundary = diagnosticsOnlyFlags.readinessBlocked && diagnosticsOnlyFlags.shadowScorerBlocked
    ? "readiness and shadow scoring blocked"
    : "authorization flags require review";
  return `First-recovery inspector ${sourceIntegrity.passed ? "source-clean" : "source-failed"}; latest=${latest}; rare_gaps=${rareActions || "none"}; candidates=${candidates || "none"}; ${boundary}.`;
}

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function isObjectRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function numberOrZero(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function sortedObject(value) {
  const payload = objectValue(value);
  return Object.fromEntries(
    Object.entries(payload)
      .map(([key, item]) => [key, Number.isFinite(Number(item)) ? Number(item) : item])
      .sort(([left], [right]) => left.localeCompare(right)),
  );
}
