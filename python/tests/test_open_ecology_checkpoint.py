from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution_sim.io.open_ecology_checkpoint import (
    OPEN_ECOLOGY_CHECKPOINT_SCHEMA_VERSION,
    REQUIRED_CHECKPOINT_COMPONENTS,
    OpenEcologyCheckpointError,
    VersionedCheckpointState,
    build_open_ecology_checkpoint,
    checkpoint_generation_identity_sha256,
    load_open_ecology_checkpoint,
    validate_open_ecology_checkpoint,
    write_open_ecology_checkpoint,
)


class OpenEcologyCheckpointTests(unittest.TestCase):
    def test_complete_checkpoint_is_deterministic_restartable_and_digest_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first_path = Path(tmpdir) / "first.json"
            second_path = Path(tmpdir) / "second.json"
            first = write_open_ecology_checkpoint(
                first_path,
                **self._complete_kwargs(),
            )
            second = write_open_ecology_checkpoint(
                second_path,
                **self._complete_kwargs(),
            )
            loaded = load_open_ecology_checkpoint(
                first_path,
                expected_source_git_sha=self._source_sha(),
                expected_generation_identity_sha256=(
                    checkpoint_generation_identity_sha256(first)
                ),
                require_restartable=True,
            )

            first_bytes = first_path.read_bytes()
            self.assertEqual(first_bytes, second_path.read_bytes())

        self.assertEqual(first, second)
        self.assertEqual(
            loaded["schema_version"],
            OPEN_ECOLOGY_CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertEqual(loaded["tick"], 1_250)
        self.assertTrue(loaded["restartability"]["restartable"])
        self.assertEqual(
            loaded["restartability"]["validated_components"],
            list(REQUIRED_CHECKPOINT_COMPONENTS),
        )
        self.assertEqual(loaded["restartability"]["missing_components"], [])
        self.assertEqual(
            loaded["components"]["world_state"]["payload"]["agents"][0]["id"],
            7,
        )
        self.assertEqual(
            loaded["components"]["environment_rng_state"]["payload"]["state"],
            [11, 29, 47],
        )
        self.assertEqual(
            loaded["components"]["evidence_writer_continuation_state"]["payload"][
                "record_stream_sha256"
            ],
            "3" * 64,
        )
        self.assertEqual(
            first_bytes,
            self._canonical(first) + b"\n",
        )

    def test_validation_result_does_not_alias_caller_input(self) -> None:
        checkpoint = build_open_ecology_checkpoint(**self._complete_kwargs())
        validated = validate_open_ecology_checkpoint(checkpoint)

        checkpoint["components"]["world_state"]["payload"]["agents"][0]["id"] = 99
        self.assertEqual(
            validated["components"]["world_state"]["payload"]["agents"][0]["id"],
            7,
        )

        validated["components"]["environment_rng_state"]["payload"]["state"][0] = 999
        self.assertEqual(
            checkpoint["components"]["environment_rng_state"]["payload"]["state"],
            [11, 29, 47],
        )

    def test_partial_observational_checkpoint_cannot_masquerade_as_restartable(
        self,
    ) -> None:
        kwargs = self._complete_kwargs()
        kwargs["sampling_rng_state"] = None
        checkpoint = build_open_ecology_checkpoint(**kwargs)

        self.assertFalse(checkpoint["restartability"]["restartable"])
        self.assertEqual(
            checkpoint["restartability"]["missing_components"],
            ["sampling_rng_state"],
        )
        validate_open_ecology_checkpoint(checkpoint)
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "not restartable"):
            validate_open_ecology_checkpoint(
                checkpoint,
                require_restartable=True,
            )

        forged = copy.deepcopy(checkpoint)
        forged["restartability"]["restartable"] = True
        forged["restartability"]["missing_components"] = []
        self._redigest(forged)
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "truth gate"):
            validate_open_ecology_checkpoint(forged)

    def test_component_source_and_generation_tampering_fail_closed(self) -> None:
        checkpoint = build_open_ecology_checkpoint(**self._complete_kwargs())
        cases: list[tuple[str, dict[str, object], str]] = []

        component = copy.deepcopy(checkpoint)
        component["components"]["world_state"]["payload"]["agents"][0]["id"] = 99
        self._redigest(component)
        cases.append(("component", component, "state SHA256 mismatch"))

        source = copy.deepcopy(checkpoint)
        source["source"]["config_contract"]["payload"]["width"] = 999
        self._redigest(source)
        cases.append(("source", source, "state SHA256 mismatch"))

        identity = copy.deepcopy(checkpoint)
        identity["generation_identity"]["generation_index"] = 12
        self._redigest(identity)
        cases.append(("identity", identity, "generation identity SHA256 mismatch"))

        checkpoint_digest = copy.deepcopy(checkpoint)
        checkpoint_digest["tick"] = 1_251
        cases.append(("checkpoint", checkpoint_digest, "checkpoint SHA256 mismatch"))

        for label, tampered, expected_error in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    OpenEcologyCheckpointError,
                    expected_error,
                ):
                    validate_open_ecology_checkpoint(tampered)

    def test_exact_container_and_envelope_keys_are_required(self) -> None:
        checkpoint = build_open_ecology_checkpoint(**self._complete_kwargs())

        extra = copy.deepcopy(checkpoint)
        extra["resume_command"] = "python arbitrary.py"
        self._redigest(extra)
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "keys mismatch"):
            validate_open_ecology_checkpoint(extra)

        missing = copy.deepcopy(checkpoint)
        del missing["components"]["sampling_rng_state"]
        self._redigest(missing)
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "keys mismatch"):
            validate_open_ecology_checkpoint(missing)

        envelope = copy.deepcopy(checkpoint)
        envelope["components"]["world_state"]["callable"] = "danger"
        self._redigest(envelope)
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "keys mismatch"):
            validate_open_ecology_checkpoint(envelope)

        boolean_type_confusion = copy.deepcopy(checkpoint)
        boolean_type_confusion["format_contract"]["duplicate_json_keys_allowed"] = 0
        self._redigest(boolean_type_confusion)
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "format_contract"):
            validate_open_ecology_checkpoint(boolean_type_confusion)

    def test_duplicate_noncanonical_and_nonfinite_json_are_rejected(self) -> None:
        checkpoint = build_open_ecology_checkpoint(**self._complete_kwargs())
        with tempfile.TemporaryDirectory() as tmpdir:
            duplicate_path = Path(tmpdir) / "duplicate.json"
            duplicate_path.write_text(
                '{"schema_version":"first","schema_version":"second"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OpenEcologyCheckpointError, "duplicate"):
                load_open_ecology_checkpoint(duplicate_path)

            pretty_path = Path(tmpdir) / "pretty.json"
            pretty_path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OpenEcologyCheckpointError, "not canonical"):
                load_open_ecology_checkpoint(pretty_path)
            self.assertEqual(
                load_open_ecology_checkpoint(
                    pretty_path,
                    require_canonical=False,
                ),
                checkpoint,
            )

            nonfinite_path = Path(tmpdir) / "nonfinite.json"
            nonfinite_path.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(OpenEcologyCheckpointError, "non-finite"):
                load_open_ecology_checkpoint(nonfinite_path)

    def test_write_size_failure_and_replace_failure_preserve_previous_destination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "checkpoint.json"
            previous = b'{"previous":"complete"}\n'
            output_path.write_bytes(previous)

            with self.assertRaisesRegex(
                OpenEcologyCheckpointError,
                "max_checkpoint_bytes",
            ):
                write_open_ecology_checkpoint(
                    output_path,
                    max_checkpoint_bytes=64,
                    **self._complete_kwargs(),
                )
            self.assertEqual(output_path.read_bytes(), previous)

            with patch(
                "evolution_sim.io.open_ecology_checkpoint.os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated"):
                    write_open_ecology_checkpoint(
                        output_path,
                        **self._complete_kwargs(),
                    )
            self.assertEqual(output_path.read_bytes(), previous)
            self.assertEqual(
                list(output_path.parent.glob(f".{output_path.name}.*.tmp")),
                [],
            )

    def test_safe_load_rejects_symlinks_oversize_and_non_json_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            real_path = Path(tmpdir) / "real.json"
            write_open_ecology_checkpoint(real_path, **self._complete_kwargs())
            link_path = Path(tmpdir) / "link.json"
            link_path.symlink_to(real_path)
            with self.assertRaisesRegex(OpenEcologyCheckpointError, "safely open"):
                load_open_ecology_checkpoint(link_path)

            with self.assertRaisesRegex(
                OpenEcologyCheckpointError,
                "max_checkpoint_bytes",
            ):
                load_open_ecology_checkpoint(
                    real_path,
                    max_checkpoint_bytes=32,
                )

            binary_path = Path(tmpdir) / "pickle.bin"
            binary_path.write_bytes(pickle.dumps(os.system))
            with self.assertRaisesRegex(
                OpenEcologyCheckpointError,
                "UTF-8 JSON|valid JSON",
            ):
                load_open_ecology_checkpoint(binary_path)

    def test_expected_source_and_generation_identity_are_fail_closed(self) -> None:
        checkpoint = build_open_ecology_checkpoint(**self._complete_kwargs())
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "source git SHA"):
            validate_open_ecology_checkpoint(
                checkpoint,
                expected_source_git_sha="b" * 40,
            )
        with self.assertRaisesRegex(
            OpenEcologyCheckpointError,
            "generation identity SHA256",
        ):
            validate_open_ecology_checkpoint(
                checkpoint,
                expected_generation_identity_sha256="f" * 64,
            )

    def test_types_identifiers_and_state_payloads_are_strict(self) -> None:
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "non-empty"):
            VersionedCheckpointState("empty_v1", {})
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "non-finite"):
            VersionedCheckpointState("bad_v1", {"value": float("nan")})
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "non-string"):
            VersionedCheckpointState("bad_key_v1", {1: "not-json-object-key"})
        with self.assertRaisesRegex(
            OpenEcologyCheckpointError, "unsupported JSON type"
        ):
            VersionedCheckpointState("tuple_v1", {"values": (1, 2, 3)})

        invalid_tick = self._complete_kwargs()
        invalid_tick["tick"] = True
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "nonnegative"):
            build_open_ecology_checkpoint(**invalid_tick)

        invalid_sha = self._complete_kwargs()
        invalid_sha["source_git_sha"] = "A" * 40
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "40-hex"):
            build_open_ecology_checkpoint(**invalid_sha)

        invalid_identity = self._complete_kwargs()
        invalid_identity["run_generation_id"] = "../escape"
        with self.assertRaisesRegex(OpenEcologyCheckpointError, "identifier"):
            build_open_ecology_checkpoint(**invalid_identity)

    def _complete_kwargs(self) -> dict[str, object]:
        return {
            "source_git_sha": self._source_sha(),
            "config_contract": self._state(
                "open_ecology_world_config_v1",
                {
                    "width": 48,
                    "height": 32,
                    "island_count": 4,
                },
            ),
            "seed_contract": self._state(
                "open_ecology_seed_contract_v1",
                {
                    "environment_seed": 101,
                    "policy_sampling_seed": 202,
                    "genome_founder_seed": 303,
                },
            ),
            "run_generation_id": "open-ecology-generation-0009",
            "island_id": "island-03",
            "generation_index": 9,
            "tick": 1_250,
            "world_state": self._state(
                "simulation_world_state_adapter_v1",
                {
                    "agents": [{"id": 7, "x": 4, "y": 8, "alive": True}],
                    "resources": [{"x": 4, "y": 9, "energy": 0.75}],
                    "climate": {"season": "wet", "phase": 0.2},
                },
            ),
            "environment_rng_state": self._state(
                "python_random_state_json_v1",
                {"algorithm": "MT19937", "state": [11, 29, 47]},
            ),
            "recurrent_policy_state": self._state(
                "frozen_recurrent_policy_state_v1",
                {
                    "artifact_sha256": "1" * 64,
                    "hidden_state_by_agent": {"7": [0.1, -0.2]},
                },
            ),
            "public_feedback_history": self._state(
                "public_feedback_history_v1",
                {
                    "previous_feedback_by_agent": {"7": [0.02, 1.0]},
                    "history_prefix_by_agent": {"7": [[0.0, 0.02]]},
                },
            ),
            "sampling_rng_state": self._state(
                "torch_generator_state_json_v1",
                {"device": "cpu", "state_bytes_hex": "00a1ff"},
            ),
            "genome_population_snapshot": self._state(
                "recurrent_genome_population_v1",
                {
                    "agent_genomes": {"7": "2" * 64},
                    "population_sha256": "4" * 64,
                },
            ),
            "evidence_writer_continuation_state": self._state(
                "bounded_open_ecology_writer_continuation_v1",
                {
                    "period_index": 12,
                    "record_count": 50_000,
                    "record_stream_sha256": "3" * 64,
                    "replay_rows_written": 256,
                    "replay_bytes_written": 1_048_576,
                },
            ),
        }

    def _state(
        self,
        schema_version: str,
        payload: dict[str, object],
    ) -> VersionedCheckpointState:
        return VersionedCheckpointState(
            schema_version=schema_version,
            payload=payload,
        )

    def _source_sha(self) -> str:
        return "a" * 40

    def _redigest(self, checkpoint: dict[str, object]) -> None:
        without_digest = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        checkpoint["checkpoint_sha256"] = hashlib.sha256(
            self._canonical(without_digest)
        ).hexdigest()

    def _canonical(self, payload: object) -> bytes:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")


if __name__ == "__main__":
    unittest.main()
