from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from evolution_sim.env.runtime.action_contract import ACTION_NAMES
    from evolution_sim.mind.recurrent_actor_critic import (
        ACTION_COUNT,
        PREVIOUS_PUBLIC_FEEDBACK_SIZE,
        PUBLIC_INPUT_SIZE,
        PublicRecurrentActorCritic,
    )
    from evolution_sim.mind.recurrent_artifact import (
        FROZEN_RECURRENT_POLICY_ARTIFACT_KIND,
        FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
        FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
        RECURRENT_ARTIFACT_SCHEMA_VERSION,
        RECURRENT_REPLAY_PROBE_CONTRACT_VERSION,
        RECURRENT_TRAINING_CRASH_CHECKPOINT_KIND,
        RECURRENT_TRAINING_CRASH_CHECKPOINT_SCHEMA_VERSION,
        RecurrentArtifactError,
        build_frozen_recurrent_policy_artifact,
        build_recurrent_artifact,
        build_recurrent_training_crash_checkpoint,
        load_frozen_recurrent_policy_artifact,
        load_recurrent_artifact,
        load_recurrent_training_crash_checkpoint,
        model_from_frozen_recurrent_policy_artifact,
        model_from_recurrent_artifact,
        validate_frozen_recurrent_policy_artifact,
        validate_recurrent_artifact,
        validate_recurrent_training_crash_checkpoint,
        write_frozen_recurrent_policy_artifact,
        write_recurrent_artifact,
        write_recurrent_training_crash_checkpoint,
    )


@unittest.skipIf(torch is None, "optional Mind ML dependency torch is not installed")
class RecurrentArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        self.model = PublicRecurrentActorCritic(initialization_seed=77)
        self.model.eval()

    def test_json_tensor_artifact_records_full_contract_and_digests(self) -> None:
        artifact = self._artifact()

        self.assertEqual(
            artifact["schema_version"],
            RECURRENT_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(artifact["model"]["public_input_size"], 541)
        self.assertEqual(artifact["model"]["learned_encoder_input_size"], 604)
        self.assertEqual(artifact["model"]["action_ordering"], list(ACTION_NAMES))
        self.assertEqual(artifact["serialization"]["tensor_dtype"], "float32_le")
        self.assertEqual(artifact["serialization"]["tensor_byte_order"], "little")
        self.assertRegex(
            artifact["serialization"]["whole_model_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(artifact["artifact_sha256"], r"^[0-9a-f]{64}$")
        tensor_names = [record["name"] for record in artifact["tensors"]]
        self.assertEqual(tensor_names, sorted(self.model.state_dict()))
        for record in artifact["tensors"]:
            self.assertEqual(record["dtype"], "float32_le")
            self.assertEqual(record["encoding"], "base64_raw")
            raw = base64.b64decode(record["data"], validate=True)
            self.assertEqual(len(raw), record["byte_length"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
        validate_recurrent_artifact(artifact)
        json.dumps(artifact, allow_nan=False)

    def test_round_trip_reconstructs_exact_cpu_eval_inference(self) -> None:
        artifact = self._artifact()
        loaded_model = model_from_recurrent_artifact(artifact)
        observations, masks, feedback = self._inputs(batch_size=8)

        with torch.no_grad():
            original = self.model.act(
                observations,
                masks,
                feedback,
                deterministic=True,
            )
            loaded = loaded_model.act(
                observations,
                masks,
                feedback,
                deterministic=True,
            )

        self.assertFalse(loaded_model.training)
        self.assertEqual(next(loaded_model.parameters()).device.type, "cpu")
        self.assertEqual(next(loaded_model.parameters()).dtype, torch.float32)
        torch.testing.assert_close(
            original.raw_logits, loaded.raw_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(original.values, loaded.values, rtol=0, atol=0)
        torch.testing.assert_close(original.actions, loaded.actions, rtol=0, atol=0)
        torch.testing.assert_close(
            original.next_state, loaded.next_state, rtol=0, atol=0
        )

    def test_seeded_stochastic_inference_replays_after_load(self) -> None:
        artifact = self._artifact()
        loaded_model = model_from_recurrent_artifact(artifact)
        observations, masks, feedback = self._inputs(batch_size=64)

        first = self.model.act(
            observations,
            masks,
            feedback,
            deterministic=False,
            generator=torch.Generator().manual_seed(9876),
        )
        second = loaded_model.act(
            observations,
            masks,
            feedback,
            deterministic=False,
            generator=torch.Generator().manual_seed(9876),
        )

        torch.testing.assert_close(first.actions, second.actions, rtol=0, atol=0)
        torch.testing.assert_close(first.log_probs, second.log_probs, rtol=0, atol=0)

    def test_atomic_write_and_strict_load_replace_existing_json(self) -> None:
        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "policy.json"
            path.parent.mkdir(parents=True)
            path.write_text("stale", encoding="utf-8")

            written = write_recurrent_artifact(path, artifact)
            loaded = load_recurrent_artifact(written)

            self.assertEqual(loaded.artifact, artifact)
            self.assertFalse(loaded.model.training)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
            with self.assertRaises(RecurrentArtifactError):
                load_recurrent_training_crash_checkpoint(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), artifact)

    def test_tensor_digest_whole_model_digest_and_artifact_digest_fail_closed(
        self,
    ) -> None:
        artifact = self._artifact()
        tensor_tamper = copy.deepcopy(artifact)
        record = tensor_tamper["tensors"][0]
        raw = bytearray(base64.b64decode(record["data"], validate=True))
        raw[0] ^= 1
        record["data"] = base64.b64encode(raw).decode("ascii")
        self._refresh_artifact_digest(tensor_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "tensor .* SHA256"):
            validate_recurrent_artifact(tensor_tamper)

        model_tamper = copy.deepcopy(artifact)
        model_tamper["serialization"]["whole_model_sha256"] = "0" * 64
        self._refresh_artifact_digest(model_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "whole-model"):
            validate_recurrent_artifact(model_tamper)

        artifact_tamper = copy.deepcopy(artifact)
        artifact_tamper["provenance"]["learner_device"] = "cpu"
        with self.assertRaisesRegex(RecurrentArtifactError, "artifact SHA256"):
            validate_recurrent_artifact(artifact_tamper)

    def test_exact_schema_shape_dtype_and_action_order_fail_closed(self) -> None:
        artifact = self._artifact()
        cases: list[dict[str, object]] = []
        extra_key = copy.deepcopy(artifact)
        extra_key["unexpected"] = True
        cases.append(extra_key)
        bad_shape = copy.deepcopy(artifact)
        bad_shape["tensors"][0]["shape"] = [1]
        self._refresh_artifact_digest(bad_shape)
        cases.append(bad_shape)
        bad_dtype = copy.deepcopy(artifact)
        bad_dtype["tensors"][0]["dtype"] = "float64_le"
        self._refresh_artifact_digest(bad_dtype)
        cases.append(bad_dtype)
        bad_actions = copy.deepcopy(artifact)
        bad_actions["model"]["action_ordering"] = list(reversed(ACTION_NAMES))
        self._refresh_artifact_digest(bad_actions)
        cases.append(bad_actions)

        for tampered in cases:
            with self.subTest(tamper=tampered):
                with self.assertRaises(RecurrentArtifactError):
                    validate_recurrent_artifact(tampered)

    def test_nonfinite_weights_are_rejected_on_build_and_verified_load(self) -> None:
        with torch.no_grad():
            first_parameter = next(self.model.parameters())
            first_parameter.view(-1)[0] = float("nan")
        with self.assertRaisesRegex(RecurrentArtifactError, "non-finite"):
            self._artifact()

        clean_model = PublicRecurrentActorCritic(initialization_seed=77)
        artifact = self._artifact(model=clean_model)
        record = artifact["tensors"][0]
        raw = bytearray(base64.b64decode(record["data"], validate=True))
        raw[:4] = struct.pack("<f", float("nan"))
        record["data"] = base64.b64encode(raw).decode("ascii")
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        artifact["serialization"]["whole_model_sha256"] = self._whole_model_sha256(
            artifact["tensors"]
        )
        self._refresh_artifact_digest(artifact)

        with self.assertRaisesRegex(RecurrentArtifactError, "non-finite"):
            validate_recurrent_artifact(artifact)

    def test_provenance_values_must_be_canonical_and_source_pinned(self) -> None:
        invalid_overrides = (
            {"seed_registry_digest": "bad"},
            {"source_commit": "dirty"},
            {"learner_seed": True},
            {"learner_device": " cuda "},
            {"training_config": {"learning_rate": float("nan")}},
        )
        for overrides in invalid_overrides:
            kwargs = self._metadata()
            kwargs.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaises(RecurrentArtifactError):
                    build_recurrent_artifact(self.model, **kwargs)

    def test_strict_json_loader_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "duplicate.json"
            path.write_text(
                '{"schema_version":"first","schema_version":"second"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RecurrentArtifactError, "duplicate"):
                load_recurrent_artifact(path)

    def test_frozen_policy_v2_binds_exact_hashes_and_executes_cpu_probe(self) -> None:
        artifact = self._frozen_artifact()

        self.assertEqual(
            artifact["schema_version"],
            FROZEN_RECURRENT_POLICY_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(
            artifact["artifact_kind"], FROZEN_RECURRENT_POLICY_ARTIFACT_KIND
        )
        integrity = artifact["integrity"]
        self.assertEqual(
            integrity["parameters_sha256"],
            artifact["serialization"]["whole_model_sha256"],
        )
        for key in (
            "parameters_sha256",
            "model_config_sha256",
            "training_config_sha256",
            "experiment_config_sha256",
            "configuration_sha256",
            "source_commit_sha256",
            "source_manifest_sha256",
            "seed_registry_digest",
        ):
            self.assertRegex(integrity[key], r"^[0-9a-f]{64}$")
        verification = artifact["verification"]
        self.assertEqual(
            verification["replay_probe_contract_version"],
            RECURRENT_REPLAY_PROBE_CONTRACT_VERSION,
        )
        self.assertEqual(verification["verified_on_device"], "cpu")
        self.assertIs(
            verification["full_world_replay_manifest"]["all_replays_exact"],
            True,
        )
        validate_frozen_recurrent_policy_artifact(artifact)
        with self.assertRaises(RecurrentArtifactError):
            validate_recurrent_artifact(artifact)

        loaded = model_from_frozen_recurrent_policy_artifact(artifact)
        observations, masks, feedback = self._inputs(batch_size=8)
        with torch.no_grad():
            original = self.model.act(observations, masks, feedback, deterministic=True)
            restored = loaded.act(observations, masks, feedback, deterministic=True)
        self.assertFalse(loaded.training)
        self.assertEqual(next(loaded.parameters()).device.type, "cpu")
        torch.testing.assert_close(
            original.raw_logits, restored.raw_logits, rtol=0, atol=0
        )
        torch.testing.assert_close(original.values, restored.values, rtol=0, atol=0)
        torch.testing.assert_close(original.actions, restored.actions, rtol=0, atol=0)

    def test_frozen_policy_atomic_round_trip_preserves_verification(self) -> None:
        artifact = self._frozen_artifact()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "frozen-policy.json"
            path.parent.mkdir(parents=True)
            path.write_text("stale", encoding="utf-8")

            write_frozen_recurrent_policy_artifact(path, artifact)
            loaded = load_frozen_recurrent_policy_artifact(path)

            self.assertEqual(loaded.artifact, artifact)
            self.assertFalse(loaded.model.training)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
            with self.assertRaises(RecurrentArtifactError):
                load_recurrent_artifact(path)
            with self.assertRaises(RecurrentArtifactError):
                load_recurrent_training_crash_checkpoint(path)

    def test_frozen_policy_hash_probe_and_world_manifest_tamper_fail_closed(
        self,
    ) -> None:
        training_config_tamper = self._frozen_artifact()
        training_config_tamper["provenance"]["training_config"]["learning_rate"] = 0.9
        self._refresh_artifact_digest(training_config_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "hashes drifted"):
            validate_frozen_recurrent_policy_artifact(training_config_tamper)

        probe_tamper = self._frozen_artifact()
        probe = probe_tamper["verification"]["probe"]
        probe["expected_actions"][0] = (probe["expected_actions"][0] + 1) % ACTION_COUNT
        probe_tamper["verification"]["probe_output_sha256"] = hashlib.sha256(
            self._canonical_json(
                {
                    "expected_raw_logits": probe["expected_raw_logits"],
                    "expected_values": probe["expected_values"],
                    "expected_actions": probe["expected_actions"],
                    "expected_next_state": probe["expected_next_state"],
                }
            )
        ).hexdigest()
        self._refresh_artifact_digest(probe_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "CPU replay probe actions"):
            validate_frozen_recurrent_policy_artifact(probe_tamper)

        manifest_tamper = self._frozen_artifact()
        manifest = manifest_tamper["verification"]["full_world_replay_manifest"]
        manifest["replay_verified_world_count"] -= 1
        manifest_tamper["verification"][
            "full_world_replay_manifest_metadata_sha256"
        ] = hashlib.sha256(self._canonical_json(manifest)).hexdigest()
        self._refresh_artifact_digest(manifest_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "verify every world"):
            validate_frozen_recurrent_policy_artifact(manifest_tamper)

    def test_frozen_policy_rejects_validation_or_lockbox_replay_provenance(
        self,
    ) -> None:
        for role, expected in (
            ("validation", "validation"),
            ("lockbox", "lockbox"),
            ("scale_validation_shadow", "validation"),
            ("promotion_lockbox_probe", "lockbox"),
        ):
            metadata = self._frozen_metadata()
            metadata["full_world_replay_manifest"]["environment_seed_roles"] = [
                "scale_train",
                role,
            ]
            with self.subTest(role=role):
                with self.assertRaisesRegex(RecurrentArtifactError, expected):
                    build_frozen_recurrent_policy_artifact(self.model, **metadata)

    def test_frozen_policy_malformed_probe_mask_fails_as_artifact_error(self) -> None:
        artifact = self._frozen_artifact()
        probe = artifact["verification"]["probe"]
        probe["action_masks"][0] = True
        artifact["verification"]["probe_input_sha256"] = hashlib.sha256(
            self._canonical_json(
                {
                    "deterministic": probe["deterministic"],
                    "observations": probe["observations"],
                    "action_masks": probe["action_masks"],
                    "previous_feedback": probe["previous_feedback"],
                }
            )
        ).hexdigest()
        self._refresh_artifact_digest(artifact)
        with self.assertRaisesRegex(RecurrentArtifactError, "fixed-width boolean"):
            validate_frozen_recurrent_policy_artifact(artifact)

    def test_crash_checkpoint_round_trips_optimizer_rng_and_never_loads_as_policy(
        self,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3.0e-4)
        observations, masks, feedback = self._inputs(batch_size=2)
        output = self.model.act(observations, masks, feedback, deterministic=True)
        output.values.sum().backward()
        optimizer.step()
        self.model.train()
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": __import__("numpy").random.get_state(),
            "policy_generator": torch.Generator().manual_seed(6789).get_state(),
        }
        checkpoint = self._checkpoint(
            optimizer_state=optimizer.state_dict(), rng_state=rng_state
        )

        self.assertEqual(
            checkpoint["schema_version"],
            RECURRENT_TRAINING_CRASH_CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertEqual(
            checkpoint["checkpoint_kind"],
            RECURRENT_TRAINING_CRASH_CHECKPOINT_KIND,
        )
        self.assertIs(checkpoint["runtime_policy_eligible"], False)
        self.assertIs(checkpoint["resumable_training_state"], True)
        self.assertNotIn("artifact_sha256", checkpoint)
        validate_recurrent_training_crash_checkpoint(checkpoint)
        with self.assertRaises(RecurrentArtifactError):
            validate_frozen_recurrent_policy_artifact(checkpoint)
        with self.assertRaises(RecurrentArtifactError):
            validate_recurrent_artifact(checkpoint)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            write_recurrent_training_crash_checkpoint(path, checkpoint)
            loaded = load_recurrent_training_crash_checkpoint(path)
            self.assertEqual(loaded.checkpoint, checkpoint)
            self.assertTrue(loaded.model.training)
            torch.testing.assert_close(
                loaded.rng_state["torch_cpu"], rng_state["torch_cpu"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                loaded.rng_state["policy_generator"],
                rng_state["policy_generator"],
                rtol=0,
                atol=0,
            )
            self.assertEqual(
                loaded.optimizer_state["param_groups"],
                optimizer.state_dict()["param_groups"],
            )
            for parameter_id, state in optimizer.state_dict()["state"].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(
                        loaded.optimizer_state["state"][parameter_id][name],
                        tensor,
                        rtol=0,
                        atol=0,
                    )
            with self.assertRaises(RecurrentArtifactError):
                load_frozen_recurrent_policy_artifact(path)

    def test_crash_checkpoint_tamper_and_unsupported_state_fail_closed(self) -> None:
        checkpoint = self._checkpoint(
            optimizer_state={"state": {}, "param_groups": []},
            rng_state={"torch_cpu": torch.get_rng_state()},
        )
        eligibility_tamper = copy.deepcopy(checkpoint)
        eligibility_tamper["runtime_policy_eligible"] = True
        self._refresh_checkpoint_digest(eligibility_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "cannot be runtime-policy"):
            validate_recurrent_training_crash_checkpoint(eligibility_tamper)

        optimizer_tamper = copy.deepcopy(checkpoint)
        optimizer_tamper["optimizer_state_sha256"] = "0" * 64
        self._refresh_checkpoint_digest(optimizer_tamper)
        with self.assertRaisesRegex(RecurrentArtifactError, "optimizer state SHA256"):
            validate_recurrent_training_crash_checkpoint(optimizer_tamper)

        with self.assertRaisesRegex(RecurrentArtifactError, "unsupported"):
            self._checkpoint(
                optimizer_state={"opaque": object()},
                rng_state={},
            )

    def _artifact(
        self,
        *,
        model: PublicRecurrentActorCritic | None = None,
    ) -> dict[str, object]:
        return build_recurrent_artifact(model or self.model, **self._metadata())

    def _frozen_artifact(self) -> dict[str, object]:
        return build_frozen_recurrent_policy_artifact(
            self.model, **self._frozen_metadata()
        )

    def _checkpoint(
        self,
        *,
        optimizer_state: dict[object, object],
        rng_state: dict[object, object],
    ) -> dict[str, object]:
        return build_recurrent_training_crash_checkpoint(
            self.model,
            optimizer_state=optimizer_state,
            rng_state=rng_state,
            optimizer_type="torch.optim.Adam",
            training_config={"algorithm": "recurrent_ppo", "learning_rate": 0.0003},
            seed_registry_digest="a" * 64,
            source_commit="b" * 40,
            source_manifest_sha256="c" * 64,
            learner_seed=12345,
            completed_updates=7,
            run_id="scale-development-run-01",
        )

    @staticmethod
    def _metadata() -> dict[str, object]:
        return {
            "training_config": {
                "algorithm": "recurrent_ppo",
                "learning_rate": 0.0003,
                "rollout_steps": 2048,
            },
            "seed_registry_digest": "a" * 64,
            "source_commit": "b" * 40,
            "data_metadata": {
                "policy_induced": True,
                "agent_steps": 4096,
                "digest": "c" * 64,
            },
            "run_metadata": {
                "run_id": "recurrent-ppo-canary",
                "episodes": 32,
            },
            "learner_seed": 12345,
            "learner_device": "cuda:0",
        }

    @classmethod
    def _frozen_metadata(cls) -> dict[str, object]:
        metadata = cls._metadata()
        metadata.update(
            {
                "experiment_config": {
                    "arm": "exact_counterfactual",
                    "updates": 32,
                    "independent_rng_tapes": 4,
                },
                "source_manifest_sha256": "d" * 64,
                "full_world_replay_manifest": {
                    "schema_version": FULL_WORLD_REPLAY_MANIFEST_SCHEMA_VERSION,
                    "manifest_sha256": "e" * 64,
                    "replay_engine_contract_sha256": "f" * 64,
                    "environment_seed_registry_sha256": "1" * 64,
                    "environment_seed_roles": ["scale_train", "scale_selection"],
                    "scenario_names": ["broad", "carrion_only"],
                    "tick_horizons": [120, 180],
                    "world_count": 64,
                    "replay_verified_world_count": 64,
                    "policy_sampling_stream_count": 4,
                    "all_replays_exact": True,
                    "verification_runner": "full-world-replay-verifier-v1",
                    "verification_runner_sha256": "2" * 64,
                },
            }
        )
        return metadata

    @staticmethod
    def _inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = torch.linspace(-1.0, 1.0, PUBLIC_INPUT_SIZE).repeat(
            batch_size, 1
        )
        masks = torch.zeros(batch_size, ACTION_COUNT, dtype=torch.bool)
        for action in ("stay", "eat", "drink", "move_east"):
            masks[:, ACTION_NAMES.index(action)] = True
        feedback = torch.zeros(batch_size, PREVIOUS_PUBLIC_FEEDBACK_SIZE)
        return observations, masks, feedback

    @classmethod
    def _whole_model_sha256(cls, records: list[dict[str, object]]) -> str:
        digest = hashlib.sha256()
        for record in records:
            raw = base64.b64decode(record["data"], validate=True)
            descriptor = {
                key: record[key]
                for key in (
                    "name",
                    "shape",
                    "dtype",
                    "encoding",
                    "byte_length",
                    "sha256",
                )
            }
            descriptor_bytes = cls._canonical_json(descriptor)
            digest.update(len(descriptor_bytes).to_bytes(8, "big"))
            digest.update(descriptor_bytes)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    @classmethod
    def _refresh_artifact_digest(cls, artifact: dict[str, object]) -> None:
        payload = {
            key: value for key, value in artifact.items() if key != "artifact_sha256"
        }
        artifact["artifact_sha256"] = hashlib.sha256(
            cls._canonical_json(payload)
        ).hexdigest()

    @classmethod
    def _refresh_checkpoint_digest(cls, checkpoint: dict[str, object]) -> None:
        payload = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        checkpoint["checkpoint_sha256"] = hashlib.sha256(
            cls._canonical_json(payload)
        ).hexdigest()

    @staticmethod
    def _canonical_json(value: object) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
