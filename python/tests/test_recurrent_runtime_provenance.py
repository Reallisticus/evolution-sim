from __future__ import annotations

import copy
import json
from unittest import mock
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

from evolution_sim.mind.provenance import stable_payload_digest

if torch is not None:
    from evolution_sim.mind.recurrent_runtime_provenance import (
        RecurrentRuntimeProvenanceError,
        assert_recurrent_scale_runtime_provenance_match,
        build_recurrent_scale_runtime_provenance,
        installed_dependency_freeze,
        validate_recurrent_scale_runtime_provenance,
    )


class _Distribution:
    def __init__(self, name: str, version: str) -> None:
        self.metadata = {"Name": name}
        self.version = version


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentRuntimeProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.freeze = installed_dependency_freeze(
            [
                _Distribution("Torch", "2.11.0+cu130"),
                _Distribution("numpy", "2.4.1"),
                _Distribution("Foo_Bar", "1.0"),
            ]
        )

    def _build(self, **overrides: object) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "source_commit": "a" * 40,
            "source_manifest_sha256": "b" * 64,
            "preregistration_digest": "c" * 64,
            "repository_clean": True,
            "device": torch.device("cpu"),
            "rollout_workers": 16,
            "counterfactual_workers": 12,
            "evaluation_workers": 8,
            "dependency_freeze": self.freeze,
        }
        kwargs.update(overrides)
        return build_recurrent_scale_runtime_provenance(**kwargs)  # type: ignore[arg-type]

    def test_dependency_freeze_is_sorted_canonical_path_free_and_hashed(
        self,
    ) -> None:
        self.assertEqual(
            self.freeze["packages"],
            [
                {"name": "foo-bar", "version": "1.0"},
                {"name": "numpy", "version": "2.4.1"},
                {"name": "torch", "version": "2.11.0+cu130"},
            ],
        )
        serialized = json.dumps(self.freeze, sort_keys=True)
        self.assertNotIn("/", serialized)
        self.assertNotIn("@", serialized)
        self.assertEqual(
            self.freeze["packages_sha256"],
            stable_payload_digest(self.freeze["packages"]),
        )

    def test_provenance_binds_source_runtime_dependencies_and_workers(self) -> None:
        provenance = self._build()

        validate_recurrent_scale_runtime_provenance(provenance)
        self.assertEqual(provenance, self._build())
        self.assertEqual(provenance["device"]["type"], "cpu")  # type: ignore[index]
        self.assertEqual(  # type: ignore[index]
            provenance["contract_binding"]["source_commit"],
            "a" * 40,
        )
        self.assertEqual(  # type: ignore[index]
            provenance["contract_binding"]["preregistration_schema_version"],
            "mind_v3_public_recurrent_ippo_scale_campaign_preregistration_v2",
        )
        self.assertEqual(  # type: ignore[index]
            provenance["contract_binding"]["seed_registry_sha256"],
            "1531a9e782dc414ae43c37346a45d581cb00a29a8348e0bd902c106166ba5ee7",
        )
        self.assertEqual(  # type: ignore[index]
            provenance["contract_binding"]["counterfactual_rng_retape_boundary"],
            "after_focal_natural_policy_draw_before_action_resolution_v1",
        )
        self.assertEqual(  # type: ignore[index]
            provenance["contract_binding"][
                "counterfactual_source_branch_schema_version"
            ],
            "mind_v3_recurrent_counterfactual_boundary_source_branch_row_v5",
        )
        self.assertEqual(  # type: ignore[index]
            provenance["workers"]["counterfactual_workers"],
            12,
        )
        self.assertEqual(  # type: ignore[index]
            provenance["workers"]["concurrent_cuda_arm_processes"],
            3,
        )
        self.assertFalse(provenance["nvidia"]["required"])  # type: ignore[index]
        self.assertEqual(  # type: ignore[index]
            provenance["dependency_freeze"]["package_count"],
            3,
        )
        privacy = provenance["privacy_contract"]
        self.assertFalse(privacy["hostname_recorded"])  # type: ignore[index]
        self.assertFalse(privacy["filesystem_paths_recorded"])  # type: ignore[index]

    def test_dirty_source_and_unavailable_cuda_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            RecurrentRuntimeProvenanceError,
            "clean repository",
        ):
            self._build(repository_clean=False)

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(
                RecurrentRuntimeProvenanceError,
                "CUDA is unavailable",
            ):
                self._build(device=torch.device("cuda:0"))

    def test_digest_tampering_and_resume_runtime_drift_fail_closed(self) -> None:
        provenance = self._build()
        tampered = copy.deepcopy(provenance)
        tampered["workers"]["rollout_workers"] = 15  # type: ignore[index]
        with self.assertRaisesRegex(
            RecurrentRuntimeProvenanceError,
            "digest mismatched",
        ):
            validate_recurrent_scale_runtime_provenance(tampered)

        drifted = self._build(rollout_workers=15)
        with self.assertRaisesRegex(
            RecurrentRuntimeProvenanceError,
            "drifted",
        ):
            assert_recurrent_scale_runtime_provenance_match(
                provenance,
                drifted,
            )

        unexpected = copy.deepcopy(provenance)
        unexpected["hostname"] = "should-never-be-recorded"
        unexpected["exact_digest"] = stable_payload_digest(
            {key: value for key, value in unexpected.items() if key != "exact_digest"}
        )
        with self.assertRaisesRegex(
            RecurrentRuntimeProvenanceError,
            "field set drifted",
        ):
            validate_recurrent_scale_runtime_provenance(unexpected)

        stale_v1_binding = copy.deepcopy(provenance)
        stale_v1_binding["contract_binding"]["seed_registry_sha256"] = "0" * 64  # type: ignore[index]
        stale_v1_binding["exact_digest"] = stable_payload_digest(
            {
                key: value
                for key, value in stale_v1_binding.items()
                if key != "exact_digest"
            }
        )
        with self.assertRaisesRegex(
            RecurrentRuntimeProvenanceError,
            "v2 campaign contract binding drifted",
        ):
            validate_recurrent_scale_runtime_provenance(stale_v1_binding)

    def test_dependency_freeze_rejects_conflicting_duplicate_versions(self) -> None:
        with self.assertRaisesRegex(
            RecurrentRuntimeProvenanceError,
            "conflicting versions",
        ):
            installed_dependency_freeze(
                [
                    _Distribution("Foo.Bar", "1"),
                    _Distribution("foo-bar", "2"),
                ]
            )


if __name__ == "__main__":
    unittest.main()
