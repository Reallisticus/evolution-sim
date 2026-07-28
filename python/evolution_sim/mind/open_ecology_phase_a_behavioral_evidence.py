"""Executable D1-D4 evidence for the open-ecology Phase-A launch.

The readiness authority deliberately does not accept a semantic report merely
because the report hashes itself.  This module produces sealed raw probe
bundles and independently reconstructs their facts from the captured bytes.
The probes exercise production contracts and implementations; no scientific
selection or validation seeds are accessed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import statistics
import stat
import subprocess
import time
from typing import Any

from evolution_sim.config import (
    CombatConfig,
    DietMatchingConfig,
    ReproductionConfig,
    SignalConfig,
    WorldConfig,
)
from evolution_sim.env import SimulationWorld
from evolution_sim.env.runtime.action_contract import ACTION_NAMES, action_contract
from evolution_sim.env.runtime.action_space import build_action_mask
from evolution_sim.env.runtime.observations import observation_contract
from evolution_sim.env.runtime import reproduction as runtime_reproduction
from evolution_sim.env.runtime.state import Agent
from evolution_sim.env.runtime.trajectory import trajectory_contract
from evolution_sim.env.viewer_contracts import viewer_contract_payload
from evolution_sim.genome import Genome, ReproductiveGenome
from evolution_sim.genome.species import genome_vector
from evolution_sim.io.open_ecology_bounded_subprocess import (
    OpenEcologyProcessGroupError,
    leader_exit_observed_without_reaping,
    terminate_process_group_before_reap,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    CRITIC_GENOME_CONDITIONING_FILM_V1,
    CRITIC_GENOME_CONDITIONING_NONE,
    GENOME_CONDITIONING_ACTOR_FILM_V1,
    PREVIOUS_PUBLIC_FEEDBACK_SIZE,
    VALUE_SHARED_TRUNK_GRADIENT_SHARED,
    VALUE_SHARED_TRUNK_GRADIENT_STOP_V1,
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_artifact import (
    build_recurrent_artifact,
    build_recurrent_training_crash_checkpoint,
)
from evolution_sim.mind.recurrent_experiment import (
    OPEN_ECOLOGY_PHASE_A,
    OPEN_ECOLOGY_PROOF_SEED_ROLE,
    OpenEcologyBroadWorldTreatment,
    OpenEcologyProofRolloutTask,
    OpenEcologySignalTreatment,
    RecurrentRolloutTask,
    _open_ecology_proof_policy_sampling_identity,
    _open_ecology_proof_task_id,
    collect_recurrent_rollout_batch,
)
from evolution_sim.mind.recurrent_genome import (
    RECURRENT_CONTROLLER_GENOME_SIZE,
    founder_recurrent_genome,
    zero_recurrent_genome,
)
from evolution_sim.mind.recurrent_genome_population import (
    RecurrentGenomePopulationManager,
    RecurrentGenomePopulationMode,
)
from evolution_sim.mind.open_ecology_seed_registry import (
    OPEN_ECOLOGY_CANONICAL_SHA256,
    OPEN_ECOLOGY_SEED_REGISTRY,
    OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
)
from evolution_sim.mind.recurrent_rollout import (
    OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
    RECURRENT_ROLLOUT_ACTION_SOURCE,
    PreviousPublicFeedback,
    RecurrentCoreOutput,
    RecurrentOnPolicyCollector,
    RecurrentRolloutBuffer,
    RecurrentRolloutError,
    RecurrentRolloutStep,
    TorchRecurrentPolicyCore,
    derive_recurrent_policy_sampling_seed,
)


D1_KIND = "cross_surface_and_self_echo"
D2_KIND = "runtime_genome_and_action_source"
D3_KIND = "critic_gradient_and_density_schedule"
D4_KIND = "fixed_batch_equivalence_and_speed"

_PRODUCER_CONTRACTS = {
    D1_KIND: "open_ecology_cross_surface_self_echo_raw_producer_v1",
    D2_KIND: "open_ecology_runtime_genome_action_source_raw_producer_v1",
    D3_KIND: "open_ecology_critic_gradient_density_raw_producer_v1",
    D4_KIND: "open_ecology_fixed_batch_equivalence_speed_raw_producer_v2",
}
_PRODUCER_NAMES = {
    D1_KIND: "produce_cross_surface_and_self_echo_report",
    D2_KIND: "produce_runtime_genome_and_action_source_report",
    D3_KIND: "produce_critic_gradient_and_density_schedule_report",
    D4_KIND: "produce_fixed_batch_equivalence_and_speed_report",
}
_D4_DECLARED_SHAPE = {
    "worlds": 16,
    "rollout_ticks": 128,
    "initial_agents": 64,
}
_D4_REPEAT_COUNT = 2
_D4_NUMERIC_RTOL = 1e-5
_D4_NUMERIC_ATOL = 1e-6
_D4_MODEL_PROOF_SEED_INDEX = 0
_D4_GENOME_PROOF_SEED_INDEX = 1
_VIEWER_PROCESS_TIMEOUT_SECONDS = 30.0
_VIEWER_PROCESS_GROUP_WAIT_SECONDS = 5.0
_VIEWER_PROCESS_POLL_SECONDS = 0.05
_VIEWER_IO_CHUNK_BYTES = 64 * 1024


def _readiness() -> Any:
    # Imported lazily so readiness can register these functions without a
    # module-initialization cycle.
    from evolution_sim.mind import open_ecology_phase_a_readiness

    return open_ecology_phase_a_readiness


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _signal_config() -> SignalConfig:
    return OpenEcologySignalTreatment().as_signal_config()


def _model_config(
    *, encoder_size: int = 8, hidden_size: int = 8
) -> RecurrentActorCriticConfig:
    return RecurrentActorCriticConfig.for_signal_config(
        _signal_config(),
        encoder_size=encoder_size,
        hidden_size=hidden_size,
        recurrent_layers=1,
        genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
    )


def _world_config(*, initial_agents: int = 0) -> WorldConfig:
    return WorldConfig(
        seed=20260727,
        max_ticks=2,
        width=5,
        height=5,
        initial_agents=initial_agents,
        water_tile_ratio=0.0,
        forest_tile_ratio=0.0,
        wetland_tile_ratio=0.0,
        rocky_tile_ratio=0.0,
        base_energy_drain=0.0,
        base_hydration_drain=0.0,
        signals=_signal_config(),
        reproduction=ReproductionConfig(
            min_age=100,
            cooldown_ticks=100,
            min_hydration_fraction=0.0,
            energy_cost=0.0,
        ),
        diet_matching=DietMatchingConfig(
            specialist_threshold=0.0,
            omnivore_threshold=0.0,
        ),
        combat=CombatConfig(
            min_attack_health_ratio=1.0,
            min_attack_energy_ratio=1.0,
            min_attack_hydration_ratio=1.0,
        ),
    )


def _place_probe_agent(world: SimulationWorld, *, x: int, y: int) -> Agent:
    genome = replace(
        Genome.sample_initial(world.rng),
        reproductive=ReproductiveGenome(signal_emission_bias=1.0),
    )
    agent = Agent(
        agent_id=world.next_agent_id,
        parent_id=None,
        lineage_id=1,
        birth_tick=0,
        death_tick=None,
        x=x,
        y=y,
        energy=genome.max_energy * 1.25,
        hydration=genome.max_hydration,
        health=genome.max_health,
        max_health=genome.max_health,
        injury_load=0.0,
        age=10,
        alive=True,
        last_reproduction_tick=-10_000,
        last_damage_source="none",
        recent_plant_energy=0.0,
        recent_fresh_kill_energy=0.0,
        recent_carcass_energy=0.0,
        genome_vector=genome_vector(genome),
        genome=genome,
        reproductive_group_id=1,
        reproductive_stage="stage0_asexual",
        reproductive_expression="asexual",
    )
    world._place_agent(agent)
    world.next_agent_id += 1
    return agent


def _ordered_action_keys(contract: Mapping[str, object]) -> list[str]:
    specs = contract["actions"]
    if not isinstance(specs, Sequence):
        raise ValueError("action contract has no ordered action specs")
    return [str(spec["key"]) for spec in specs if isinstance(spec, Mapping)]


def _communication_shape_from_action_order(
    action_order: Sequence[str],
) -> tuple[int, int]:
    profile_ids_by_token: dict[int, set[int]] = {}
    for action_name in action_order:
        if not action_name.startswith("signal_"):
            continue
        parts = action_name.split("_")
        if len(parts) != 4 or parts[2] != "profile":
            raise ValueError("communication action name is malformed")
        token_id = int(parts[1])
        profile_id = int(parts[3])
        profile_ids_by_token.setdefault(token_id, set()).add(profile_id)
    token_ids = set(profile_ids_by_token)
    if token_ids != set(range(len(token_ids))):
        raise ValueError("communication action token ids are not contiguous")
    profile_counts = {len(values) for values in profile_ids_by_token.values()}
    if len(profile_counts) != 1:
        raise ValueError("communication action profile counts differ by token")
    profiles_per_token = next(iter(profile_counts), 0)
    expected_profile_ids = set(range(profiles_per_token))
    if any(values != expected_profile_ids for values in profile_ids_by_token.values()):
        raise ValueError("communication action profile ids are not contiguous")
    expected_signal_actions = len(token_ids) * profiles_per_token
    observed_signal_actions = sum(
        int(action_name.startswith("signal_")) for action_name in action_order
    )
    if observed_signal_actions != expected_signal_actions:
        raise ValueError("communication action token/profile matrix is incomplete")
    return len(token_ids), profiles_per_token


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_stable_open_file(
    descriptor: int,
    path: Path,
    before: os.stat_result,
    expected_sha256: str,
    *,
    field: str,
) -> None:
    after = os.fstat(descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"{field} disappeared after execution: {error}") from error
    if (
        not stat.S_ISREG(after.st_mode)
        or _stable_file_identity(after) != _stable_file_identity(before)
        or _stable_file_identity(path_after) != _stable_file_identity(before)
        or _fd_sha256(descriptor) != expected_sha256
    ):
        raise RuntimeError(f"{field} identity or bytes changed during execution")


def _viewer_surface_probe(replay: Mapping[str, object]) -> dict[str, object]:
    node_candidate = shutil.which("node")
    if node_candidate is None:
        raise RuntimeError("D1 viewer proof requires Node.js")
    node_path = Path(node_candidate).resolve(strict=True)
    if not node_path.is_file():
        raise RuntimeError("D1 viewer Node.js executable is not a regular file")
    repository = Path(__file__).resolve().parents[3]
    validator_path = repository / "viewer" / "replay_validator.mjs"
    if not validator_path.is_file() or validator_path.is_symlink():
        raise RuntimeError("D1 viewer validator source is missing or unsafe")
    open_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    node_fd = os.open(node_path, open_flags)
    validator_fd = os.open(validator_path, open_flags)
    node_before = os.fstat(node_fd)
    validator_before = os.fstat(validator_fd)
    if not stat.S_ISREG(node_before.st_mode) or not stat.S_ISREG(
        validator_before.st_mode
    ):
        os.close(node_fd)
        os.close(validator_fd)
        raise RuntimeError("D1 viewer runtime inputs must be regular files")
    node_sha256 = _fd_sha256(node_fd)
    validator_sha256 = _fd_sha256(validator_fd)
    module_uri = validator_path.resolve(strict=True).as_uri()
    script = (
        "import fs from 'node:fs';"
        "import crypto from 'node:crypto';"
        f"const m=await import({json.dumps(module_uri)});"
        "const x=JSON.parse(fs.readFileSync(0,'utf8'));"
        "const r=m.validateTrajectoryContractForEvidence(x);"
        "const nodeExecutableSha256=crypto.createHash('sha256')"
        ".update(fs.readFileSync(process.execPath)).digest('hex');"
        "process.stdout.write(JSON.stringify({result:r,"
        "nodeVersion:process.version,nodeExecPath:process.execPath,"
        "nodeExecutableSha256}));"
    )
    maximum_output_bytes = 256 * 1024
    replay_bytes = _canonical_json_bytes(replay)
    if len(replay_bytes) > 1024 * 1024:
        raise RuntimeError("D1 viewer validator input exceeded its bound")
    try:
        process = subprocess.Popen(
            [str(node_path), "--input-type=module", "--eval", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repository,
            env={"LANG": "C", "LC_ALL": "C"},
            start_new_session=True,
        )
        returncode, stdout, stderr = _run_viewer_validator_process(
            process,
            input_bytes=replay_bytes,
            maximum_output_bytes=maximum_output_bytes,
            timeout_seconds=_VIEWER_PROCESS_TIMEOUT_SECONDS,
        )
        _require_stable_open_file(
            node_fd,
            node_path,
            node_before,
            node_sha256,
            field="D1 Node.js executable",
        )
        _require_stable_open_file(
            validator_fd,
            validator_path,
            validator_before,
            validator_sha256,
            field="D1 viewer validator source",
        )
    finally:
        os.close(node_fd)
        os.close(validator_fd)
    if returncode != 0:
        raise RuntimeError(
            "D1 viewer validator rejected the production trajectory contract: "
            + stderr.decode("utf-8", errors="replace")[:2_000]
        )
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("D1 viewer validator returned invalid JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise RuntimeError("D1 viewer validator returned malformed evidence")
    node_version = payload.get("nodeVersion")
    if not isinstance(node_version, str) or not node_version:
        raise RuntimeError("D1 viewer validator omitted its Node.js version")
    node_exec_path = payload.get("nodeExecPath")
    executed_node_sha256 = payload.get("nodeExecutableSha256")
    if (
        not isinstance(node_exec_path, str)
        or Path(node_exec_path).resolve(strict=True) != node_path
        or executed_node_sha256 != node_sha256
    ):
        raise RuntimeError("D1 executed Node.js identity or bytes drifted")
    return {
        "result": payload["result"],
        "node_realpath": str(node_path),
        "node_version": node_version,
        "node_executable_sha256": str(executed_node_sha256),
        "validator_realpath": str(validator_path.resolve(strict=True)),
        "validator_source_sha256": validator_sha256,
    }


def _run_viewer_validator_process(
    process: subprocess.Popen[bytes],
    *,
    input_bytes: bytes,
    maximum_output_bytes: int,
    timeout_seconds: float,
) -> tuple[int, bytes, bytes]:
    """Exchange bounded bytes and close the original process group before reap."""

    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("D1 viewer validator timeout must be finite and positive")
    if maximum_output_bytes <= 0:
        raise ValueError("D1 viewer validator output bound must be positive")
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_viewer_process_group(process)
        raise RuntimeError("D1 viewer validator pipes were not created")
    selector: selectors.BaseSelector | None = None
    streams = {
        "stdin": process.stdin,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    outputs = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    pending_input = memoryview(input_bytes)
    deadline = time.monotonic() + timeout_seconds
    group_cleanup_attempted = False
    leader_exit_observed = False
    try:
        selector = selectors.DefaultSelector()
        for name, stream in streams.items():
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            events = selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ
            selector.register(stream, events, data=name)
        if not pending_input:
            selector.unregister(process.stdin)
            process.stdin.close()

        while not leader_exit_observed:
            leader_exit_observed = leader_exit_observed_without_reaping(process)
            if leader_exit_observed:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("D1 viewer validator timed out")
            events = selector.select(
                timeout=min(remaining, _VIEWER_PROCESS_POLL_SECONDS)
            )
            for key, _mask in events:
                name = str(key.data)
                stream = key.fileobj
                if name == "stdin":
                    try:
                        written = os.write(
                            key.fd,
                            pending_input[:_VIEWER_IO_CHUNK_BYTES],
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError as error:
                        raise RuntimeError(
                            "D1 viewer validator closed stdin before consuming "
                            "the replay"
                        ) from error
                    pending_input = pending_input[written:]
                    if not pending_input:
                        selector.unregister(stream)
                        streams["stdin"].close()
                    continue
                try:
                    chunk = os.read(key.fd, _VIEWER_IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    streams[name].close()
                    continue
                output = outputs[name]
                output.extend(chunk)
                if len(output) > maximum_output_bytes:
                    raise RuntimeError("D1 viewer validator output exceeded its bound")

        if pending_input:
            raise RuntimeError(
                "D1 viewer validator exited before consuming the complete replay"
            )
        group_cleanup_attempted = True
        returncode = _terminate_viewer_process_group(
            process,
            leader_exit_observed=True,
        )
        if not streams["stdin"].closed:
            try:
                selector.unregister(streams["stdin"])
            except KeyError:
                pass
            streams["stdin"].close()
        while any(not streams[name].closed for name in ("stdout", "stderr")):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("D1 viewer validator timed out")
            events = selector.select(
                timeout=min(remaining, _VIEWER_PROCESS_POLL_SECONDS)
            )
            for key, _mask in events:
                name = str(key.data)
                if name == "stdin":
                    continue
                stream = key.fileobj
                try:
                    chunk = os.read(key.fd, _VIEWER_IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    streams[name].close()
                    continue
                output = outputs[name]
                output.extend(chunk)
                if len(output) > maximum_output_bytes:
                    raise RuntimeError("D1 viewer validator output exceeded its bound")
        return returncode, bytes(outputs["stdout"]), bytes(outputs["stderr"])
    finally:
        if not group_cleanup_attempted:
            _terminate_viewer_process_group(
                process,
                leader_exit_observed=(True if leader_exit_observed else None),
            )
        if selector is not None:
            selector.close()
        for stream in streams.values():
            if not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


def _terminate_viewer_process_group(
    process: subprocess.Popen[bytes],
    *,
    leader_exit_observed: bool | None = None,
) -> int:
    """Bound cleanup of a validator group before its leader can be reaped."""

    try:
        observed = (
            leader_exit_observed_without_reaping(process)
            if leader_exit_observed is None
            else leader_exit_observed
        )
        return terminate_process_group_before_reap(
            process,
            wait_timeout_seconds=_VIEWER_PROCESS_GROUP_WAIT_SECONDS,
            leader_exit_observed=observed,
        )
    except (OSError, OpenEcologyProcessGroupError) as error:
        raise RuntimeError(
            "D1 viewer validator process group survived termination"
        ) from error


def _surface_action_order(
    *,
    observation: Mapping[str, object],
    action: Mapping[str, object],
    model_artifact: Mapping[str, object],
    replay: Mapping[str, object],
    checkpoint: Mapping[str, object],
    viewer_probe: Mapping[str, object],
) -> dict[str, list[str]]:
    replay_action = replay["action_contract"]
    if not isinstance(replay_action, Mapping):
        raise ValueError("replay action contract is malformed")
    model_payload = model_artifact["model"]
    checkpoint_model = checkpoint["model"]
    if not isinstance(model_payload, Mapping) or not isinstance(
        checkpoint_model, Mapping
    ):
        raise ValueError("model or checkpoint contract is malformed")
    observation_names = observation["action_names"]
    if not isinstance(observation_names, Sequence):
        raise ValueError("observation contract has no action names")

    viewer_result = viewer_probe["result"]
    if not isinstance(viewer_result, Mapping) or not isinstance(
        viewer_result.get("actionOrdering"), Sequence
    ):
        raise ValueError("viewer action-order evidence is malformed")
    return {
        "observation": [str(value) for value in observation_names],
        "action": _ordered_action_keys(action),
        "model_artifact": [str(value) for value in model_payload["action_ordering"]],
        "replay": _ordered_action_keys(replay_action),
        "checkpoint": [str(value) for value in checkpoint_model["action_ordering"]],
        "viewer": [str(value) for value in viewer_result["actionOrdering"]],
    }


def _run_cross_surface_probe() -> dict[str, object]:
    signal_config = _signal_config()
    observation = observation_contract(signal_config)
    action = action_contract(signal_config)
    replay = trajectory_contract(signal_config)
    viewer = viewer_contract_payload()
    viewer_probe = _viewer_surface_probe({**replay, "records": []})
    config = _model_config()
    model = PublicRecurrentActorCritic(config, initialization_seed=7001)
    artifact = build_recurrent_artifact(
        model,
        training_config={"probe": D1_KIND},
        seed_registry_digest="0" * 64,
        source_commit="0" * 40,
        data_metadata={"probe": True},
        run_metadata={"scientific_outcomes_allowed": False},
        learner_seed=7001,
        learner_device="cpu",
    )
    checkpoint = build_recurrent_training_crash_checkpoint(
        model,
        optimizer_state={},
        rng_state={},
        optimizer_type="probe_no_optimizer",
        training_config={"probe": D1_KIND},
        seed_registry_digest="0" * 64,
        source_commit="0" * 40,
        source_manifest_sha256="0" * 64,
        learner_seed=7001,
        completed_updates=0,
        run_id="d1-contract-probe",
    )
    action_order = _surface_action_order(
        observation=observation,
        action=action,
        model_artifact=artifact,
        replay=replay,
        checkpoint=checkpoint,
        viewer_probe=viewer_probe,
    )
    observation_channels = observation["communication_token_channels"]
    replay_observation = replay["observation_contract"]
    if not isinstance(observation_channels, Mapping) or not isinstance(
        replay_observation, Mapping
    ):
        raise ValueError("D1 tokenized observation contracts are malformed")
    replay_channels = replay_observation["communication_token_channels"]
    observation_signal = observation["signal_contract"]
    replay_signal = replay_observation["signal_contract"]
    action_communication = action["communication"]
    replay_communication = replay["action_contract"]["communication"]
    viewer_result = viewer_probe["result"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            replay_channels,
            observation_signal,
            replay_signal,
            action_communication,
            replay_communication,
            viewer_result,
        )
    ):
        raise ValueError("D1 communication contracts are malformed")
    model_shape = _communication_shape_from_action_order(action_order["model_artifact"])
    checkpoint_shape = _communication_shape_from_action_order(
        action_order["checkpoint"]
    )
    surface_token_count = {
        "observation": int(observation_signal["communication_token_count"]),
        "action": int(action_communication["token_count"]),
        "model_artifact": model_shape[0],
        "replay": int(replay_signal["communication_token_count"]),
        "checkpoint": checkpoint_shape[0],
        "viewer": int(viewer_result["communicationTokenCount"]),
    }
    surface_profiles_per_token = {
        "observation": int(observation_signal["communication_profiles_per_token"]),
        "action": int(action_communication["profiles_per_token"]),
        "model_artifact": model_shape[1],
        "replay": int(replay_signal["communication_profiles_per_token"]),
        "checkpoint": checkpoint_shape[1],
        "viewer": int(viewer_result["communicationProfilesPerToken"]),
    }
    if (
        len(list(observation_channels["field_order"]))
        != surface_token_count["observation"]
        or len(list(replay_channels["field_order"])) != surface_token_count["replay"]
        or int(replay_communication["token_count"]) != surface_token_count["replay"]
        or int(replay_communication["profiles_per_token"])
        != surface_profiles_per_token["replay"]
    ):
        raise ValueError("D1 observation/replay communication shapes disagree")

    self_echo_cases: list[dict[str, object]] = []
    for token_id in range(int(action_communication["token_count"])):
        for profile_index in range(int(action_communication["profiles_per_token"])):
            world = SimulationWorld(_world_config())
            emitter = _place_probe_agent(world, x=2, y=2)
            receiver = _place_probe_agent(world, x=2, y=3)
            action_name = f"signal_{token_id}_profile_{profile_index}"
            mask = build_action_mask(world._action_mask_context(emitter))
            _moved, outcome = world._resolve_action_with_outcome(
                emitter,
                action_name,
                observation_action_mask=mask,
                resolution_action_mask=mask,
            )
            emitter_observation = world._observe_agent(emitter)
            receiver_observation = world._observe_agent(receiver)
            token_field = f"communication_signal_token_{token_id}"
            emitter_self = emitter_observation["self"]
            emitter_patch = emitter_observation["local_patch"]
            receiver_self = receiver_observation["self"]
            receiver_patch = receiver_observation["local_patch"]
            signal_outcome = outcome["signal"]
            if (
                not isinstance(emitter_self, Mapping)
                or not isinstance(emitter_patch, Sequence)
                or not isinstance(receiver_self, Mapping)
                or not isinstance(receiver_patch, Sequence)
                or not isinstance(signal_outcome, Mapping)
            ):
                raise ValueError("D1 self-echo probe observation is malformed")
            events = list(world.tick_signal_emission_events)
            event = events[0] if len(events) == 1 else {}
            self_echo_cases.append(
                {
                    "action": action_name,
                    "token_id": token_id,
                    "profile_index": profile_index,
                    "emitted": signal_outcome.get("emitted") is True,
                    "emitter_self_projection_nonzero": (
                        float(emitter_self[token_field]) != 0.0
                    ),
                    "emitter_local_patch_projection_nonzero": any(
                        isinstance(cell, Mapping)
                        and float(cell.get(token_field, 0.0)) != 0.0
                        for cell in emitter_patch
                    ),
                    "adjacent_receiver_self_projection_positive": (
                        float(receiver_self[token_field]) > 0.0
                    ),
                    "adjacent_receiver_local_patch_projection_positive": any(
                        isinstance(cell, Mapping)
                        and float(cell.get(token_field, 0.0)) > 0.0
                        for cell in receiver_patch
                    ),
                    "global_token_field_positive": (
                        world._current_signal_state().field(token_field)[emitter.y][
                            emitter.x
                        ]
                        > 0.0
                    ),
                    "emission_event_count": len(events),
                    "identity_matches": (
                        signal_outcome.get("token_id") == token_id
                        and signal_outcome.get("profile_index") == profile_index
                        and isinstance(event, Mapping)
                        and event.get("field_name") == token_field
                        and event.get("token_id") == token_id
                        and event.get("profile_index") == profile_index
                        and event.get("source_agent_id") == emitter.agent_id
                    ),
                }
            )
    return {
        "surface_action_order": action_order,
        "surface_contract_sha256": {
            "observation": _sha256_json(observation),
            "action": _sha256_json(action),
            "model_artifact": _sha256_json(artifact["model"]),
            "replay": _sha256_json(replay),
            "checkpoint": _sha256_json(checkpoint["model"]),
            "viewer": _sha256_json(
                {
                    "contract": viewer,
                    "runtime_probe": viewer_probe,
                }
            ),
        },
        "surface_token_count": surface_token_count,
        "surface_profiles_per_token": surface_profiles_per_token,
        "communication": {
            "token_count": int(action_communication["token_count"]),
            "profiles_per_token": int(action_communication["profiles_per_token"]),
        },
        "self_echo_cases": self_echo_cases,
    }


def _cross_surface_facts(observed: Mapping[str, object]) -> dict[str, object]:
    surfaces = (
        "observation",
        "action",
        "model_artifact",
        "replay",
        "checkpoint",
        "viewer",
    )
    orders = observed["surface_action_order"]
    token_counts = observed["surface_token_count"]
    profile_counts = observed["surface_profiles_per_token"]
    communication = observed["communication"]
    self_echo_cases = observed["self_echo_cases"]
    if (
        not isinstance(orders, Mapping)
        or not isinstance(token_counts, Mapping)
        or not isinstance(profile_counts, Mapping)
        or not isinstance(communication, Mapping)
        or not isinstance(self_echo_cases, Sequence)
        or not all(isinstance(case, Mapping) for case in self_echo_cases)
    ):
        raise ValueError("D1 raw observation is malformed")
    order_digests = {
        surface: _sha256_json(list(orders[surface])) for surface in surfaces
    }
    return {
        "field_order_sha256_by_surface": order_digests,
        "token_profile_count_by_surface": {
            surface: int(token_counts[surface]) for surface in surfaces
        },
        "profiles_per_token_by_surface": {
            surface: int(profile_counts[surface]) for surface in surfaces
        },
        "action_count_by_surface": {
            surface: len(list(orders[surface])) for surface in surfaces
        },
        "self_echo_case_count": len(self_echo_cases),
        "emitter_own_signal_self_projection_nonzero_count": sum(
            int(case["emitter_self_projection_nonzero"] is True)
            for case in self_echo_cases
        ),
        "emitter_own_signal_local_patch_projection_nonzero_count": sum(
            int(case["emitter_local_patch_projection_nonzero"] is True)
            for case in self_echo_cases
        ),
        "adjacent_receiver_self_projection_positive_count": sum(
            int(case["adjacent_receiver_self_projection_positive"] is True)
            for case in self_echo_cases
        ),
        "adjacent_receiver_local_patch_projection_positive_count": sum(
            int(case["adjacent_receiver_local_patch_projection_positive"] is True)
            for case in self_echo_cases
        ),
        "signal_emission_success_count": sum(
            int(case["emitted"] is True) for case in self_echo_cases
        ),
        "global_signal_field_positive_count": sum(
            int(case["global_token_field_positive"] is True) for case in self_echo_cases
        ),
        "global_signal_emission_event_count": sum(
            int(case["emission_event_count"]) for case in self_echo_cases
        ),
        "signal_action_identity_mismatch_count": sum(
            int(case["identity_matches"] is not True) for case in self_echo_cases
        ),
    }


def _decision_payload(decision: object) -> dict[str, object]:
    return {
        "requested_action": str(getattr(decision, "requested_action")),
        "source": str(getattr(decision, "source")),
        "policy_id": str(getattr(decision, "policy_id")),
        "policy_version": str(getattr(decision, "policy_version")),
        "diagnostics": copy.deepcopy(getattr(decision, "diagnostics")),
    }


def _manager_agent_sha256s(
    manager: RecurrentGenomePopulationManager,
) -> dict[str, str]:
    return {
        str(agent_id): manager.genome_sha256_for_agent(agent_id)
        for agent_id in manager.registered_agent_ids
    }


def _snapshot_replay_decisions(
    *,
    snapshot: Mapping[str, object],
    serialized_snapshot: bytes,
    observations: Sequence[Mapping[str, object]],
    model_seed: int,
    environment_seed: int,
    policy_sampling_seed: int,
) -> dict[str, object]:
    expected_sha256 = str(snapshot["snapshot_sha256"])
    artifact_manager = RecurrentGenomePopulationManager.from_snapshot_artifact(
        snapshot,
        expected_snapshot_sha256=expected_sha256,
    )
    serialized_manager = RecurrentGenomePopulationManager.from_serialized_snapshot(
        serialized_snapshot,
        expected_snapshot_sha256=expected_sha256,
    )

    def decisions_for(
        manager: RecurrentGenomePopulationManager,
    ) -> list[dict[str, object]]:
        model = PublicRecurrentActorCritic(
            _model_config(),
            initialization_seed=model_seed,
        )
        collector = RecurrentOnPolicyCollector(TorchRecurrentPolicyCore(model))
        collector.start_world(
            world_id=manager.world_identity,
            rollout_ticks=1,
            environment_seed=environment_seed,
            policy_sampling_seed=policy_sampling_seed,
            genome_stream_seed=manager.genome_stream_seed,
            genome_population_mode=manager.mode,
        )
        # This is the same fail-closed restoration seam used by the runtime
        # checkpoint loader.  The fresh manager created by start_world has no
        # live bindings; the validated restored manager supplies them.
        collector._genome_population_manager = manager
        return [
            _decision_payload(
                collector.decide(
                    copy.deepcopy(dict(observation)),
                    dict(observation["action_mask"]),
                )
            )
            for observation in observations
        ]

    artifact_decisions = decisions_for(artifact_manager)
    serialized_decisions = decisions_for(serialized_manager)
    return {
        "expected_snapshot_sha256": expected_sha256,
        "serialized_snapshot_sha256": hashlib.sha256(serialized_snapshot).hexdigest(),
        "artifact_restored_snapshot_sha256": artifact_manager.snapshot_artifact()[
            "snapshot_sha256"
        ],
        "serialized_restored_snapshot_sha256": (
            serialized_manager.snapshot_artifact()["snapshot_sha256"]
        ),
        "source_agent_genome_sha256s": {
            str(entry["agent_id"]): str(entry["genome"]["genome_sha256"])
            for entry in snapshot["agents"]
        },
        "artifact_agent_genome_sha256s": _manager_agent_sha256s(artifact_manager),
        "serialized_agent_genome_sha256s": _manager_agent_sha256s(serialized_manager),
        "artifact_decisions": artifact_decisions,
        "serialized_decisions": serialized_decisions,
    }


def _run_runtime_genome_mode_probe(
    *,
    mode: str,
    world_identity: str,
    genome_stream_seed: int,
    policy_sampling_seed: int,
    model_seed: int,
) -> dict[str, object]:
    environment_seed = _world_config(initial_agents=1).seed
    model = PublicRecurrentActorCritic(
        _model_config(),
        initialization_seed=model_seed,
    )
    collector = RecurrentOnPolicyCollector(TorchRecurrentPolicyCore(model))
    collector.start_world(
        world_id=world_identity,
        rollout_ticks=2,
        environment_seed=environment_seed,
        policy_sampling_seed=policy_sampling_seed,
        genome_stream_seed=genome_stream_seed,
        genome_population_mode=mode,
    )
    world = SimulationWorld(_world_config(initial_agents=1), policy=collector)
    parent = world.alive_agents()[0]
    if parent.agent_id != 1:
        raise ValueError("D2 runtime probe expected canonical first agent id")
    manager = collector._genome_population_manager
    if manager is None:
        raise ValueError("D2 runtime collector has no genome population")

    # Exercise one complete, finalized parent transition before the birth.  A
    # birth-reset proof with a never-acted parent can falsely pass even when a
    # child inherits its parent's live recurrent state, because both states
    # happen to be zero.
    parent.age = 0
    parent.energy = parent.genome.max_energy * 1.25
    parent.hydration = parent.genome.max_hydration
    parent.health = parent.max_health
    parent.last_reproduction_tick = 0
    world.tick = 0
    pre_birth_births, _pre_birth_deaths = world._run_tick()
    if pre_birth_births != 0:
        raise ValueError("D2 runtime probe reproduced before the parent state probe")
    parent_hidden = collector._hidden_by_agent.get(parent.agent_id)
    parent_feedback = collector._feedback_by_agent.get(parent.agent_id)
    parent_pre_birth_state = {
        "hidden_present": parent_hidden is not None,
        "hidden_nonzero": (
            parent_hidden is not None
            and any(float(value) != 0.0 for value in parent_hidden)
        ),
        "feedback_present": parent_feedback is not None,
        "feedback_nonzero": (
            parent_feedback is not None
            and parent_feedback != PreviousPublicFeedback.zero()
        ),
    }
    if not all(parent_pre_birth_state.values()):
        raise ValueError(
            "D2 parent recurrent state did not become nonzero before birth"
        )

    parent.age = 100
    parent.energy = parent.genome.max_energy * 1.25
    parent.hydration = parent.genome.max_hydration
    parent.health = parent.max_health
    parent.last_reproduction_tick = -10_000
    world.tick = 1

    expected_child_id = world.next_agent_id
    pre_birth_snapshot = manager.snapshot_artifact()
    reference_manager = RecurrentGenomePopulationManager.from_snapshot_artifact(
        pre_birth_snapshot,
        expected_snapshot_sha256=str(pre_birth_snapshot["snapshot_sha256"]),
    )
    expected_child_metadata = reference_manager.child_metadata(
        child_agent_id=expected_child_id,
        primary_parent_id=parent.agent_id,
        secondary_parent_id=None,
    )
    expected_parent_genome = reference_manager.genome_for_agent(parent.agent_id)
    expected_child_genome = reference_manager.genome_for_agent(expected_child_id)

    births = runtime_reproduction.run_reproduction_phase(
        world,
        context=world._reproduction_context(),
    )
    children = sorted(
        (
            agent
            for agent in world.agents.values()
            if agent.parent_id == parent.agent_id
        ),
        key=lambda agent: agent.agent_id,
    )
    if births != 1 or len(children) != 1:
        raise ValueError("D2 runtime probe expected exactly one actual child")
    child = children[0]
    if child.agent_id != expected_child_id:
        raise ValueError("D2 runtime child identity drifted from the pre-birth state")
    actual_parent_genome = manager.genome_for_agent(parent.agent_id)
    actual_child_genome = manager.genome_for_agent(child.agent_id)
    exact_child_derivation = {
        "pre_birth_snapshot_sha256": pre_birth_snapshot["snapshot_sha256"],
        "mutation_max_step": pre_birth_snapshot["mutation"]["max_step"],
        "expected_parent_genome_sha256": expected_parent_genome.sha256,
        "actual_parent_genome_sha256": actual_parent_genome.sha256,
        "expected_parent_genome_values": list(expected_parent_genome.values),
        "actual_parent_genome_values": list(actual_parent_genome.values),
        "expected_child_genome_sha256": expected_child_genome.sha256,
        "actual_child_genome_sha256": actual_child_genome.sha256,
        "expected_child_genome_values": list(expected_child_genome.values),
        "actual_child_genome_values": list(actual_child_genome.values),
        "expected_child_metadata": copy.deepcopy(expected_child_metadata),
        "actual_child_metadata": copy.deepcopy(child.mind_inheritance_metadata),
    }
    pre_run_observations = [
        copy.deepcopy(world._observe_agent(agent))
        for agent in sorted(
            (parent, child),
            key=lambda agent: agent.agent_id,
        )
    ]

    missing = copy.deepcopy(pre_run_observations[0])
    missing_metadata = missing["metadata"]
    if not isinstance(missing_metadata, dict):
        raise ValueError("D2 observation metadata is malformed")
    missing_metadata["agent_id"] = 99
    missing_rejection = ""
    try:
        collector.decide(missing, dict(missing["action_mask"]))
    except RecurrentRolloutError as error:
        missing_rejection = str(error)
    if not missing_rejection:
        raise ValueError("D2 missing-genome decision did not fail closed")

    snapshot = manager.snapshot_artifact()
    serialized_snapshot = manager.serialize_snapshot()
    replay = _snapshot_replay_decisions(
        snapshot=snapshot,
        serialized_snapshot=serialized_snapshot,
        observations=pre_run_observations,
        model_seed=model_seed,
        environment_seed=environment_seed,
        policy_sampling_seed=policy_sampling_seed,
    )

    world._run_tick()
    world.tick = 2
    world._run_tick()
    collector.finish_world()
    first_step_by_agent: dict[int, object] = {}
    for step in collector.buffer.steps:
        first_step_by_agent.setdefault(step.agent_id, step)
    reset_evidence: list[dict[str, object]] = []
    for agent in (parent, child):
        step = first_step_by_agent.get(agent.agent_id)
        if step is None:
            raise ValueError(
                "D2 runtime probe did not collect an agent decision for "
                f"agent {agent.agent_id}; observed={sorted(first_step_by_agent)}"
            )
        reset_evidence.append(
            {
                "agent_id": agent.agent_id,
                "is_child": agent.agent_id == child.agent_id,
                "initial_hidden_all_zero": all(
                    float(value) == 0.0 for value in step.hidden
                ),
                "initial_feedback_zero": (
                    step.previous_feedback == PreviousPublicFeedback.zero()
                ),
                "step_genome_sha256": step.genome_sha256,
                "metadata_genome_sha256": agent.mind_inheritance_metadata.get(
                    "genome_sha256"
                ),
            }
        )
    return {
        "mode": mode,
        "birth_count": births,
        "founder": {
            "agent_id": parent.agent_id,
            "mind_inheritance_metadata": copy.deepcopy(
                parent.mind_inheritance_metadata
            ),
        },
        "children": [
            {
                "agent_id": child.agent_id,
                "parent_id": child.parent_id,
                "mind_inheritance_metadata": copy.deepcopy(
                    child.mind_inheritance_metadata
                ),
            }
        ],
        "missing_genome_rejection": missing_rejection,
        "parent_pre_birth_state": parent_pre_birth_state,
        "exact_child_derivation": exact_child_derivation,
        "snapshot_replay": replay,
        "newborn_state": reset_evidence,
        "trajectory_action_sources": [
            str(record["action_source"]) for record in world.trajectory_records
        ],
    }


def _run_runtime_genome_probe() -> dict[str, object]:
    return {
        "heritable": _run_runtime_genome_mode_probe(
            mode="heritable",
            world_identity="d2-heritable-population",
            genome_stream_seed=8101,
            policy_sampling_seed=8111,
            model_seed=8103,
        ),
        "zero_all": _run_runtime_genome_mode_probe(
            mode="zero_all",
            world_identity="d2-zero-population",
            genome_stream_seed=8102,
            policy_sampling_seed=8112,
            model_seed=8104,
        ),
        "expected_zero_sha256": zero_recurrent_genome().sha256,
    }


def _runtime_genome_facts(observed: Mapping[str, object]) -> dict[str, object]:
    heritable = observed["heritable"]
    zero_all = observed["zero_all"]
    if not all(isinstance(value, Mapping) for value in (heritable, zero_all)):
        raise ValueError("D2 raw observation is malformed")
    heritable_founder = heritable["founder"]
    heritable_children = list(heritable["children"])
    zero_founder = zero_all["founder"]
    zero_children = list(zero_all["children"])
    if (
        not isinstance(heritable_founder, Mapping)
        or not isinstance(zero_founder, Mapping)
        or not all(isinstance(child, Mapping) for child in heritable_children)
        or not all(isinstance(child, Mapping) for child in zero_children)
    ):
        raise ValueError("D2 runtime founder/child evidence is malformed")
    heritable_founder_metadata = heritable_founder["mind_inheritance_metadata"]
    if not isinstance(heritable_founder_metadata, Mapping):
        raise ValueError("D2 heritable founder metadata is malformed")
    parent_sha256 = str(heritable_founder_metadata["genome_sha256"])
    parent_match = sum(
        int(
            list(child["mind_inheritance_metadata"]["parent_genome_sha256s"])
            == [parent_sha256]
        )
        for child in heritable_children
    )
    expected_zero = str(observed["expected_zero_sha256"])
    zero_metadata = [
        zero_founder["mind_inheritance_metadata"],
        *[child["mind_inheritance_metadata"] for child in zero_children],
    ]
    replay_mismatches = 0
    exact_child_derivation_mismatches = 0
    inheritance_bound_violations = 0
    for mode_evidence in (heritable, zero_all):
        replay = mode_evidence["snapshot_replay"]
        if not isinstance(replay, Mapping):
            raise ValueError("D2 snapshot replay evidence is malformed")
        expected_snapshot = str(replay["expected_snapshot_sha256"])
        replay_mismatches += int(
            replay["artifact_restored_snapshot_sha256"] != expected_snapshot
        )
        replay_mismatches += int(
            replay["serialized_restored_snapshot_sha256"] != expected_snapshot
        )
        replay_mismatches += int(
            replay["artifact_agent_genome_sha256s"]
            != replay["source_agent_genome_sha256s"]
        )
        replay_mismatches += int(
            replay["serialized_agent_genome_sha256s"]
            != replay["source_agent_genome_sha256s"]
        )
        replay_mismatches += int(
            replay["artifact_decisions"] != replay["serialized_decisions"]
        )
        derivation = mode_evidence["exact_child_derivation"]
        if not isinstance(derivation, Mapping):
            raise ValueError("D2 exact child derivation evidence is malformed")
        exact_child_derivation_mismatches += int(
            derivation["expected_parent_genome_sha256"]
            != derivation["actual_parent_genome_sha256"]
        )
        exact_child_derivation_mismatches += int(
            derivation["expected_parent_genome_values"]
            != derivation["actual_parent_genome_values"]
        )
        exact_child_derivation_mismatches += int(
            derivation["expected_child_genome_sha256"]
            != derivation["actual_child_genome_sha256"]
        )
        exact_child_derivation_mismatches += int(
            derivation["expected_child_genome_values"]
            != derivation["actual_child_genome_values"]
        )
        exact_child_derivation_mismatches += int(
            derivation["expected_child_metadata"] != derivation["actual_child_metadata"]
        )
        mutation_max_step = float(derivation["mutation_max_step"])
        inheritance_bound_violations += sum(
            int(abs(float(child) - float(parent)) > mutation_max_step + 1e-12)
            for parent, child in zip(
                derivation["actual_parent_genome_values"],
                derivation["actual_child_genome_values"],
                strict=True,
            )
        )
    child_reset_count = sum(
        int(
            row["is_child"] is True
            and row["initial_hidden_all_zero"] is True
            and row["initial_feedback_zero"] is True
            and row["step_genome_sha256"] == row["metadata_genome_sha256"]
        )
        for mode_evidence in (heritable, zero_all)
        for row in mode_evidence["newborn_state"]
    )
    action_sources = [
        str(source)
        for mode_evidence in (heritable, zero_all)
        for source in mode_evidence["trajectory_action_sources"]
    ]
    return {
        "heritable_founder_count": 1,
        "heritable_child_count": len(heritable_children),
        "parent_lineage_match_count": parent_match,
        "zero_all_founder_count": 1,
        "zero_all_child_count": len(zero_children),
        "zero_all_nonzero_genome_count": sum(
            int(metadata["genome_sha256"] != expected_zero)
            for metadata in zero_metadata
        ),
        "birth_state_reset_count": child_reset_count,
        "parent_pre_birth_nonzero_state_count": sum(
            int(
                isinstance(mode_evidence["parent_pre_birth_state"], Mapping)
                and all(mode_evidence["parent_pre_birth_state"].values())
            )
            for mode_evidence in (heritable, zero_all)
        ),
        "exact_child_derivation_mismatch_count": (exact_child_derivation_mismatches),
        "inheritance_bound_violation_count": inheritance_bound_violations,
        "genome_digest_replay_mismatch_count": replay_mismatches,
        "missing_genome_rejection_count": sum(
            int(
                "no registered recurrent genome"
                in str(mode_evidence["missing_genome_rejection"])
            )
            for mode_evidence in (heritable, zero_all)
        ),
        "decision_count": len(action_sources),
        "heuristic_action_source_count": sum(
            int(source != RECURRENT_ROLLOUT_ACTION_SOURCE) for source in action_sources
        ),
    }


@contextmanager
def _isolated_torch_process_state(
    *,
    num_threads: int,
    preserve_cuda_rng: bool = False,
) -> Any:
    """Temporarily bind Torch process globals and restore them exactly."""

    import torch

    cpu_rng_state = torch.get_rng_state().clone()
    previous_threads = int(torch.get_num_threads())
    cuda_rng_state = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if preserve_cuda_rng and torch.cuda.is_available()
        else None
    )
    try:
        torch.set_num_threads(num_threads)
        yield torch
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        torch.set_num_threads(previous_threads)


def _gradient_l2(parameters: Sequence[Any]) -> float:
    import torch

    squares = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            squares += gradient.detach().to(dtype=torch.float64).square().sum()
    return float(torch.sqrt(squares).item())


def _run_critic_gradient_probe(
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    architecture = preregistration["architecture"]
    if not isinstance(architecture, Mapping):
        raise ValueError("Phase A architecture contract is malformed")
    from evolution_sim.mind import open_ecology_phase_a as phase_a
    from evolution_sim.mind import recurrent_experiment

    canonical_cells = [
        {
            "cell_id": cell_id,
            "critic_genome_conditioning": phase_a.OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][
                0
            ],
            "value_shared_trunk_gradient": (
                phase_a.OPEN_ECOLOGY_PHASE_A_CELLS[cell_id][1]
            ),
        }
        for cell_id in phase_a.OPEN_ECOLOGY_PHASE_A_CELL_ORDER
    ]
    encoder_size = int(architecture["encoder_size"])
    hidden_size = int(architecture["hidden_size"])
    recurrent_layers = int(architecture["recurrent_layers"])
    if (
        encoder_size <= 0
        or hidden_size <= 0
        or recurrent_layers <= 0
        or list(architecture["critic_gradient_cells"]) != canonical_cells
    ):
        raise ValueError("Phase A architecture does not match canonical cells")

    with _isolated_torch_process_state(num_threads=1) as torch:
        base_config = replace(
            _model_config(
                encoder_size=encoder_size,
                hidden_size=hidden_size,
            ),
            recurrent_layers=recurrent_layers,
        )
        observations = torch.linspace(
            -0.75,
            0.75,
            steps=3 * 2 * base_config.public_input_size,
            dtype=torch.float32,
        ).reshape(3, 2, base_config.public_input_size)
        masks = torch.ones((3, 2, len(ACTION_NAMES)), dtype=torch.bool)
        feedback = torch.zeros((3, 2, PREVIOUS_PUBLIC_FEEDBACK_SIZE))
        genome = torch.tensor(
            founder_recurrent_genome(seed=8202).values,
            dtype=torch.float32,
        )
        genomes = genome.reshape(
            1,
            1,
            RECURRENT_CONTROLLER_GENOME_SIZE,
        ).repeat(3, 2, 1)
        alternate_genomes = torch.zeros_like(genomes)
        rows: list[dict[str, object]] = []
        common_states: list[dict[str, Any]] = []
        for cell in canonical_cells:
            config = replace(
                base_config,
                critic_genome_conditioning=str(cell["critic_genome_conditioning"]),
                value_shared_trunk_gradient=str(cell["value_shared_trunk_gradient"]),
            )
            model = PublicRecurrentActorCritic(
                config,
                initialization_seed=8203,
            )
            common_states.append(
                {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                    if not key.startswith("critic_genome_")
                }
            )
            value_output = model.forward_sequence(
                observations,
                masks,
                feedback,
                genome_values=genomes,
            )
            alternate_value_output = model.forward_sequence(
                observations,
                masks,
                feedback,
                genome_values=alternate_genomes,
            )
            critic_genome_sensitivity = float(
                (value_output.values.detach() - alternate_value_output.values.detach())
                .abs()
                .max()
                .item()
            )
            value_output.values.sum().backward()
            value_trunk_l2 = _gradient_l2(
                tuple(model.encoder.parameters()) + tuple(model.recurrent.parameters())
            )
            value_head_l2 = _gradient_l2(tuple(model.value.parameters()))
            critic_genome_parameter_l2 = _gradient_l2(
                tuple(
                    parameter
                    for name, parameter in model.named_parameters()
                    if name.startswith("critic_genome_")
                )
            )
            model.zero_grad(set_to_none=True)
            actor_output = model.forward_sequence(
                observations,
                masks,
                feedback,
                genome_values=genomes,
            )
            actor_output.raw_logits[
                ...,
                ACTION_NAMES.index("eat"),
            ].sum().backward()
            actor_trunk_l2 = _gradient_l2(
                tuple(model.encoder.parameters()) + tuple(model.recurrent.parameters())
            )
            rows.append(
                {
                    **cell,
                    "observed_cell_contract": {
                        "cell_id": str(cell["cell_id"]),
                        "critic_genome_conditioning": (
                            model.config.critic_genome_conditioning
                        ),
                        "value_shared_trunk_gradient": (
                            model.config.value_shared_trunk_gradient
                        ),
                    },
                    "model_config": {
                        "encoder_size": config.encoder_size,
                        "hidden_size": config.hidden_size,
                        "recurrent_layers": config.recurrent_layers,
                        "genome_conditioning_mode": (config.genome_conditioning_mode),
                    },
                    "value_shared_trunk_gradient_l2": round(
                        value_trunk_l2,
                        12,
                    ),
                    "value_head_gradient_l2": round(value_head_l2, 12),
                    "critic_genome_sensitivity_max_abs_delta": round(
                        critic_genome_sensitivity,
                        12,
                    ),
                    "critic_genome_parameter_gradient_l2": round(
                        critic_genome_parameter_l2,
                        12,
                    ),
                    "actor_shared_trunk_gradient_l2": round(
                        actor_trunk_l2,
                        12,
                    ),
                }
            )
        reference_state = common_states[0]
        common_parameter_mismatch_count = 0
        for state in common_states[1:]:
            common_parameter_mismatch_count += len(set(reference_state) ^ set(state))
            common_parameter_mismatch_count += sum(
                int(not torch.equal(reference_state[key], state[key]))
                for key in set(reference_state) & set(state)
            )
        if common_parameter_mismatch_count:
            raise ValueError(
                "D3 common model initialization drifted across canonical cells"
            )
        return {
            "cell_contracts": canonical_cells,
            "cell_gradient_observations": rows,
            "phase_a_density_schedule": list(
                recurrent_experiment._open_ecology_density_cycle("phase_a")
            ),
            "phase_b_density_cycle": list(
                recurrent_experiment._open_ecology_density_cycle("phase_b")
            ),
            "common_parameter_mismatch_count": (common_parameter_mismatch_count),
            "torch_runtime": {
                "version": str(torch.__version__),
                "device": "cpu",
                "dtype": "float32",
                "num_threads": int(torch.get_num_threads()),
            },
        }


def _critic_gradient_facts(observed: Mapping[str, object]) -> dict[str, object]:
    rows = list(observed["cell_gradient_observations"])
    contracts = list(observed["cell_contracts"])
    if (
        int(observed["common_parameter_mismatch_count"]) != 0
        or len(rows) != 4
        or len(contracts) != 4
    ):
        raise ValueError("D3 canonical four-cell evidence is incomplete")
    rows_by_cell = {
        str(row["cell_id"]): row for row in rows if isinstance(row, Mapping)
    }
    contract_ids = [str(contract["cell_id"]) for contract in contracts]
    if list(rows_by_cell) != contract_ids:
        raise ValueError("D3 cell gradient order or identity drifted")
    for contract in contracts:
        cell_id = str(contract["cell_id"])
        row = rows_by_cell[cell_id]
        if any(
            row[field] != contract[field]
            for field in (
                "cell_id",
                "critic_genome_conditioning",
                "value_shared_trunk_gradient",
            )
        ):
            raise ValueError("D3 observed cell does not match its contract")
    observed_contracts = [copy.deepcopy(row["observed_cell_contract"]) for row in rows]
    if not all(isinstance(contract, Mapping) for contract in observed_contracts):
        raise ValueError("D3 observed model contracts are malformed")
    shared_rows = [
        row
        for row in rows
        if row["value_shared_trunk_gradient"] == VALUE_SHARED_TRUNK_GRADIENT_SHARED
    ]
    stopped_rows = [
        row
        for row in rows
        if row["value_shared_trunk_gradient"] == VALUE_SHARED_TRUNK_GRADIENT_STOP_V1
    ]
    if len(shared_rows) != 2 or len(stopped_rows) != 2:
        raise ValueError("D3 shared/stopped cell coverage drifted")
    conditioned_rows = [
        row
        for row in rows
        if row["critic_genome_conditioning"] == CRITIC_GENOME_CONDITIONING_FILM_V1
    ]
    unconditioned_rows = [
        row
        for row in rows
        if row["critic_genome_conditioning"] == CRITIC_GENOME_CONDITIONING_NONE
    ]
    if len(conditioned_rows) != 2 or len(unconditioned_rows) != 2:
        raise ValueError("D3 conditioned/unconditioned cell coverage drifted")
    return {
        "cell_contracts": copy.deepcopy(contracts),
        "observed_cell_contracts": observed_contracts,
        "phase_a_density_schedule": copy.deepcopy(observed["phase_a_density_schedule"]),
        "phase_b_density_cycle": copy.deepcopy(observed["phase_b_density_cycle"]),
        "critic_conditioned_genome_sensitivity_min_abs_delta": min(
            float(row["critic_genome_sensitivity_max_abs_delta"])
            for row in conditioned_rows
        ),
        "critic_unconditioned_genome_sensitivity_max_abs_delta": max(
            abs(float(row["critic_genome_sensitivity_max_abs_delta"]))
            for row in unconditioned_rows
        ),
        "critic_conditioned_parameter_gradient_l2": min(
            float(row["critic_genome_parameter_gradient_l2"])
            for row in conditioned_rows
        ),
        "shared_value_trunk_gradient_l2": min(
            float(row["value_shared_trunk_gradient_l2"]) for row in shared_rows
        ),
        "stop_gradient_value_trunk_gradient_l2": max(
            abs(float(row["value_shared_trunk_gradient_l2"])) for row in stopped_rows
        ),
        "stop_gradient_value_head_gradient_l2": min(
            float(row["value_head_gradient_l2"]) for row in stopped_rows
        ),
        "actor_loss_shared_trunk_gradient_l2": min(
            float(row["actor_shared_trunk_gradient_l2"]) for row in rows
        ),
    }


def _semantic_row(output: Any) -> dict[str, object]:
    logits = tuple(float(value) for value in output.logits)
    best_action = max(range(len(logits)), key=logits.__getitem__)
    if not all(
        value == value and abs(value) != float("inf")
        for value in (*logits, float(output.value), *output.next_hidden)
    ):
        raise ValueError("D4 recurrent core emitted non-finite values")
    return {
        "action_index": best_action,
    }


def _sealed_d4_deterministic_cuda_contract() -> dict[str, object]:
    return {
        "device": "cuda",
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms_enabled": True,
        "deterministic_algorithms_warn_only_enabled": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "default_dtype": "torch.float32",
    }


def _observe_d4_deterministic_cuda_contract() -> dict[str, object]:
    import torch

    return {
        "device": "cuda",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
    }


def _configure_and_verify_d4_deterministic_cuda_contract() -> dict[str, object]:
    from evolution_sim.mind.open_ecology_phase_a import (
        configure_open_ecology_phase_a_determinism,
    )

    configure_open_ecology_phase_a_determinism()
    observed = _observe_d4_deterministic_cuda_contract()
    if observed != _sealed_d4_deterministic_cuda_contract():
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError(
                "D4 deterministic Phase-A CUDA contract could not be established"
            )
        )
    return observed


def _verify_current_d4_deterministic_cuda_contract() -> dict[str, object]:
    observed = _observe_d4_deterministic_cuda_contract()
    if observed != _sealed_d4_deterministic_cuda_contract():
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError("D4 deterministic Phase-A CUDA contract drifted")
        )
    return observed


def _d4_proof_seed_contract(*, shape: Mapping[str, int]) -> dict[str, object]:
    proof_seeds = tuple(OPEN_ECOLOGY_SEED_REGISTRY[OPEN_ECOLOGY_PROOF_SEED_ROLE])
    worlds = int(shape["worlds"])
    initial_agents = int(shape["initial_agents"])
    if (
        worlds <= 0
        or worlds > len(proof_seeds)
        or initial_agents <= 0
        or len(proof_seeds) != 16
    ):
        raise ValueError("D4 proof seed request exceeds the sealed engineering role")
    core_indices = [index % len(proof_seeds) for index in range(initial_agents)]
    policy_sampling_identities = [
        _open_ecology_proof_policy_sampling_identity(
            environment_seed_index=index,
            initial_agents=initial_agents,
        )
        for index in range(worlds)
    ]
    return {
        "seed_registry_version": OPEN_ECOLOGY_SEED_REGISTRY_VERSION,
        "seed_registry_sha256": OPEN_ECOLOGY_CANONICAL_SHA256,
        "engineering_seed_role": OPEN_ECOLOGY_PROOF_SEED_ROLE,
        "environment_seed_indices": list(range(worlds)),
        "environment_seeds": list(proof_seeds[:worlds]),
        "model_initialization_seed_index": _D4_MODEL_PROOF_SEED_INDEX,
        "model_initialization_seed": proof_seeds[_D4_MODEL_PROOF_SEED_INDEX],
        "genome_stream_seed_index": _D4_GENOME_PROOF_SEED_INDEX,
        "genome_stream_seed": proof_seeds[_D4_GENOME_PROOF_SEED_INDEX],
        "core_genome_seed_indices": core_indices,
        "core_genome_seeds": [proof_seeds[index] for index in core_indices],
        "policy_sampling_identity_contract": (
            "mind_v3_open_ecology_engineering_proof_task_v1"
        ),
        "task_ids": [
            _open_ecology_proof_task_id(
                environment_seed_index=index,
                environment_seed=proof_seeds[index],
                initial_agents=initial_agents,
            )
            for index in range(worlds)
        ],
        "policy_sampling_identities": policy_sampling_identities,
        "policy_sampling_seeds": [
            derive_recurrent_policy_sampling_seed(task_identity=identity)
            for identity in policy_sampling_identities
        ],
        "scientific_training_seed_access_count": 0,
        "scientific_selection_seed_access_count": 0,
    }


def _d4_input_rows(
    *,
    initial_agents: int,
    public_input_size: int,
    proof_seed_contract: Mapping[str, object],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[bool, ...], ...],
    tuple[PreviousPublicFeedback, ...],
    tuple[tuple[float, ...], ...],
]:
    observations = tuple(
        tuple(
            round(
                (((agent_index + 1) * (field_index + 3)) % 97) / 96.0,
                8,
            )
            for field_index in range(public_input_size)
        )
        for agent_index in range(initial_agents)
    )
    masks = tuple(tuple(True for _ in ACTION_NAMES) for _ in range(initial_agents))
    feedback = tuple(PreviousPublicFeedback.zero() for _ in range(initial_agents))
    genome_seeds = tuple(int(seed) for seed in proof_seed_contract["core_genome_seeds"])
    if len(genome_seeds) != initial_agents:
        raise ValueError("D4 core genome proof seed coverage drifted")
    genomes = tuple(founder_recurrent_genome(seed=seed).values for seed in genome_seeds)
    return observations, masks, feedback, genomes


def _run_direct_model_batch(
    model: PublicRecurrentActorCritic,
    *,
    observations: Sequence[Sequence[float]],
    masks: Sequence[Sequence[bool]],
    feedback: Sequence[PreviousPublicFeedback],
    hidden: Sequence[Sequence[float]],
    genomes: Sequence[Sequence[float]],
) -> tuple[RecurrentCoreOutput, ...]:
    """Exercise the model API independently of both rollout-core adapters."""

    import torch

    active_rows = len(observations)
    reference = next(model.parameters())
    observation_tensor = torch.tensor(
        tuple(tuple(row) for row in observations),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, active_rows, model.config.public_input_size)
    mask_tensor = torch.tensor(
        tuple(tuple(row) for row in masks),
        device=reference.device,
        dtype=torch.bool,
    ).reshape(1, active_rows, len(ACTION_NAMES))
    feedback_tensor = torch.tensor(
        tuple(row.vector() for row in feedback),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, active_rows, PREVIOUS_PUBLIC_FEEDBACK_SIZE)
    hidden_tensor = torch.tensor(
        tuple(tuple(row) for row in hidden),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(
        active_rows,
        model.config.recurrent_layers,
        model.config.hidden_size,
    )
    genome_tensor = torch.tensor(
        tuple(tuple(row) for row in genomes),
        device=reference.device,
        dtype=reference.dtype,
    ).reshape(1, active_rows, RECURRENT_CONTROLLER_GENOME_SIZE)
    with torch.no_grad():
        output = model.forward_sequence(
            observation_tensor,
            mask_tensor,
            feedback_tensor,
            genome_values=genome_tensor,
            initial_state=hidden_tensor.permute(1, 0, 2),
        )
    final_state = output.final_state.detach().cpu()
    raw_logits = output.raw_logits[0].detach().cpu()
    values = output.values[0].detach().cpu()
    return tuple(
        RecurrentCoreOutput(
            logits=tuple(float(value) for value in raw_logits[row].tolist()),
            value=float(values[row].item()),
            next_hidden=tuple(
                float(value) for value in final_state[:, row, :].reshape(-1).tolist()
            ),
        )
        for row in range(active_rows)
    )


def _synchronize_device(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


def _d4_phase_a_collection_inputs(
    *,
    shape: Mapping[str, int],
) -> tuple[
    PublicRecurrentActorCritic,
    tuple[OpenEcologyProofRolloutTask, ...],
    dict[str, object],
]:
    """Build the production Phase-A path using engineering proof seeds only."""

    import torch
    from evolution_sim.mind.open_ecology_phase_a import (
        OPEN_ECOLOGY_PHASE_A_CELLS,
        OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS,
        OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE,
    )

    worlds = int(shape["worlds"])
    rollout_ticks = int(shape["rollout_ticks"])
    if int(shape["initial_agents"]) != OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS:
        raise ValueError(
            "D4 production collector requires the Phase-A initial population"
        )
    if not 1 <= worlds <= OPEN_ECOLOGY_PHASE_A_WORLDS_PER_UPDATE:
        raise ValueError("D4 production collector world count is out of range")
    if rollout_ticks <= 0:
        raise ValueError("D4 production collector rollout_ticks must be positive")
    proof_seed_contract = _d4_proof_seed_contract(shape=shape)
    treatment = OpenEcologyBroadWorldTreatment(
        initial_agents=OPEN_ECOLOGY_PHASE_A_INITIAL_AGENTS
    )
    critic, gradient = OPEN_ECOLOGY_PHASE_A_CELLS["A0"]
    model_config = RecurrentActorCriticConfig.for_signal_config(
        treatment.signals.as_signal_config(),
        encoder_size=256,
        hidden_size=256,
        recurrent_layers=1,
        genome_conditioning_mode=GENOME_CONDITIONING_ACTOR_FILM_V1,
        critic_genome_conditioning=critic,
        value_shared_trunk_gradient=gradient,
    )
    tasks = tuple(
        OpenEcologyProofRolloutTask(
            task_id=str(proof_seed_contract["task_ids"][environment_index]),
            scenario="broad",
            environment_seed=int(
                proof_seed_contract["environment_seeds"][environment_index]
            ),
            rollout_ticks=rollout_ticks,
            policy_sampling_identity=str(
                proof_seed_contract["policy_sampling_identities"][environment_index]
            ),
            policy_sampling_seed=int(
                proof_seed_contract["policy_sampling_seeds"][environment_index]
            ),
            seed_role=OPEN_ECOLOGY_PROOF_SEED_ROLE,
            genome_stream_seed=int(proof_seed_contract["genome_stream_seed"]),
            genome_population_mode=RecurrentGenomePopulationMode.HERITABLE.value,
            open_ecology_treatment=treatment,
            open_ecology_learner_seed=int(
                proof_seed_contract["model_initialization_seed"]
            ),
            open_ecology_environment_seed_index=environment_index,
            open_ecology_genome_stream_seed_index=(_D4_GENOME_PROOF_SEED_INDEX),
            open_ecology_training_phase=OPEN_ECOLOGY_PHASE_A,
            open_ecology_update_index=0,
            open_ecology_world_index=environment_index,
        )
        for environment_index in range(worlds)
    )
    model = PublicRecurrentActorCritic(
        model_config,
        initialization_seed=int(proof_seed_contract["model_initialization_seed"]),
    ).to(device="cpu", dtype=torch.float32)
    model.eval()
    if (
        [task.task_id for task in tasks] != proof_seed_contract["task_ids"]
        or [task.seed_role for task in tasks] != [OPEN_ECOLOGY_PROOF_SEED_ROLE] * worlds
        or [task.environment_seed for task in tasks]
        != proof_seed_contract["environment_seeds"]
        or [task.policy_sampling_identity for task in tasks]
        != proof_seed_contract["policy_sampling_identities"]
        or [task.policy_sampling_seed for task in tasks]
        != proof_seed_contract["policy_sampling_seeds"]
    ):
        raise ValueError("D4 production collector proof task identity drifted")
    return model, tasks, proof_seed_contract


def _d4_collector_semantic_row(step: RecurrentRolloutStep) -> dict[str, object]:
    """Serialize behavior semantics while excluding tolerated model numerics."""

    row = asdict(step)
    for field in (
        "logprob",
        "entropy",
        "value",
        "hidden",
        "bootstrap_value",
        "fixed_batch_runtime_sha256",
        "fixed_batch_turn_rank",
        "fixed_batch_active_rows",
        "fixed_batch_capacity",
    ):
        row.pop(field)
    return row


def _d4_collector_semantic_sha256(buffer: RecurrentRolloutBuffer) -> str:
    digest = hashlib.sha256()
    for step in buffer.steps:
        digest.update(_canonical_json_bytes(_d4_collector_semantic_row(step)))
    return digest.hexdigest()


def _d4_collector_path_facts(
    buffer: RecurrentRolloutBuffer,
) -> dict[str, object]:
    steps = buffer.steps
    fixed_steps = tuple(
        step for step in steps if step.fixed_batch_runtime_sha256 is not None
    )
    return {
        "semantic_sha256": _d4_collector_semantic_sha256(buffer),
        "transition_count": len(steps),
        "fixed_batch_step_count": len(fixed_steps),
        "fixed_batch_runtime_sha256": sorted(
            {
                str(step.fixed_batch_runtime_sha256)
                for step in fixed_steps
                if step.fixed_batch_runtime_sha256 is not None
            }
        ),
    }


def _run_d4_collector_mode(
    *,
    mode: str,
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentRolloutTask],
    rollout_workers: int,
) -> tuple[RecurrentRolloutBuffer, dict[str, object], int]:
    if mode == "scalar":
        fixed_batch_capacity = None
    elif mode == "batched":
        fixed_batch_capacity = OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
    else:
        raise ValueError(f"unknown D4 execution mode {mode!r}")
    started = time.perf_counter_ns()
    buffer, _diagnostics = collect_recurrent_rollout_batch(
        model,
        tasks,
        rollout_workers=rollout_workers,
        fixed_batch_capacity=fixed_batch_capacity,
    )
    elapsed_ns = time.perf_counter_ns() - started
    return buffer, _d4_collector_path_facts(buffer), elapsed_ns


def _numeric_error(
    reference: float,
    candidate: float,
) -> tuple[float, float]:
    if not math.isfinite(reference) or not math.isfinite(candidate):
        raise ValueError("D4 numerical comparison received a non-finite value")
    error = abs(candidate - reference)
    tolerance = _D4_NUMERIC_ATOL + _D4_NUMERIC_RTOL * abs(reference)
    return error, error / tolerance


def _d4_output_error(
    reference: RecurrentCoreOutput,
    candidate: RecurrentCoreOutput,
) -> dict[str, float | int]:
    reference_semantics = _semantic_row(reference)
    candidate_semantics = _semantic_row(candidate)
    if len(reference.logits) != len(candidate.logits):
        raise ValueError("D4 compared logit widths differ")
    if len(reference.next_hidden) != len(candidate.next_hidden):
        raise ValueError("D4 compared hidden widths differ")
    logit_errors = [
        _numeric_error(float(left), float(right))
        for left, right in zip(reference.logits, candidate.logits, strict=True)
    ]
    value_error, value_ratio = _numeric_error(
        float(reference.value),
        float(candidate.value),
    )
    hidden_errors = [
        _numeric_error(float(left), float(right))
        for left, right in zip(
            reference.next_hidden,
            candidate.next_hidden,
            strict=True,
        )
    ]
    return {
        "action_mismatch_count": int(
            reference_semantics["action_index"] != candidate_semantics["action_index"]
        ),
        "max_abs_logit_error": max(
            (error for error, _ratio in logit_errors), default=0.0
        ),
        "max_abs_value_error": value_error,
        "max_abs_hidden_error": max(
            (error for error, _ratio in hidden_errors),
            default=0.0,
        ),
        "max_logit_tolerance_ratio": max(
            (ratio for _error, ratio in logit_errors),
            default=0.0,
        ),
        "max_value_tolerance_ratio": value_ratio,
        "max_hidden_tolerance_ratio": max(
            (ratio for _error, ratio in hidden_errors),
            default=0.0,
        ),
    }


def _run_d4_equivalence(
    *,
    device: str,
    shape: Mapping[str, int],
) -> dict[str, object]:
    import torch

    proof_seed_contract = _d4_proof_seed_contract(shape=shape)
    model = PublicRecurrentActorCritic(
        _model_config(encoder_size=256, hidden_size=256),
        initialization_seed=int(proof_seed_contract["model_initialization_seed"]),
    ).to(device=device, dtype=torch.float32)
    model.eval()
    core = TorchRecurrentPolicyCore(model)
    observations, masks, feedback, genomes = _d4_input_rows(
        initial_agents=int(shape["initial_agents"]),
        public_input_size=core.public_input_size,
        proof_seed_contract=proof_seed_contract,
    )
    scalar_world_digests: list[str] = []
    batched_world_digests: list[str] = []
    reference_world_digests: list[str] = []
    action_mismatch_count = 0
    scalar_reference_action_mismatch_count = 0
    batched_reference_action_mismatch_count = 0
    comparison_count = 0
    reference_comparison_count = 0
    max_abs_logit_error = 0.0
    max_abs_value_error = 0.0
    max_abs_hidden_error = 0.0
    max_logit_tolerance_ratio = 0.0
    max_value_tolerance_ratio = 0.0
    max_hidden_tolerance_ratio = 0.0
    max_scalar_reference_logit_tolerance_ratio = 0.0
    max_scalar_reference_value_tolerance_ratio = 0.0
    max_scalar_reference_hidden_tolerance_ratio = 0.0
    max_batched_reference_logit_tolerance_ratio = 0.0
    max_batched_reference_value_tolerance_ratio = 0.0
    max_batched_reference_hidden_tolerance_ratio = 0.0
    initial_hidden = core.initial_hidden()
    for world_index in range(int(shape["worlds"])):
        hidden = [initial_hidden for _ in observations]
        scalar_hash = hashlib.sha256()
        batched_hash = hashlib.sha256()
        reference_hash = hashlib.sha256()
        for tick in range(int(shape["rollout_ticks"])):
            scalar_outputs = tuple(
                core.forward_step(
                    observation,
                    mask,
                    previous,
                    hidden_row,
                    genome_values=genome,
                )
                for observation, mask, previous, hidden_row, genome in zip(
                    observations,
                    masks,
                    feedback,
                    hidden,
                    genomes,
                    strict=True,
                )
            )
            batched_outputs = core.forward_fixed_batch(
                observations,
                masks,
                feedback,
                tuple(hidden),
                batch_capacity=OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
                genome_values=genomes,
            )
            reference_outputs = _run_direct_model_batch(
                model,
                observations=observations,
                masks=masks,
                feedback=feedback,
                hidden=hidden,
                genomes=genomes,
            )
            for agent_index, (
                scalar_output,
                batched_output,
                reference_output,
            ) in enumerate(
                zip(
                    scalar_outputs,
                    batched_outputs,
                    reference_outputs,
                    strict=True,
                )
            ):
                scalar_semantics = _semantic_row(scalar_output)
                batched_semantics = _semantic_row(batched_output)
                reference_semantics = _semantic_row(reference_output)
                scalar_hash.update(
                    _canonical_json_bytes(
                        {
                            "world": world_index,
                            "tick": tick,
                            "agent": agent_index,
                            **scalar_semantics,
                        }
                    )
                )
                batched_hash.update(
                    _canonical_json_bytes(
                        {
                            "world": world_index,
                            "tick": tick,
                            "agent": agent_index,
                            **batched_semantics,
                        }
                    )
                )
                reference_hash.update(
                    _canonical_json_bytes(
                        {
                            "world": world_index,
                            "tick": tick,
                            "agent": agent_index,
                            **reference_semantics,
                        }
                    )
                )
                scalar_batched = _d4_output_error(scalar_output, batched_output)
                scalar_reference = _d4_output_error(
                    reference_output,
                    scalar_output,
                )
                batched_reference = _d4_output_error(
                    reference_output,
                    batched_output,
                )
                action_mismatch_count += int(scalar_batched["action_mismatch_count"])
                scalar_reference_action_mismatch_count += int(
                    scalar_reference["action_mismatch_count"]
                )
                batched_reference_action_mismatch_count += int(
                    batched_reference["action_mismatch_count"]
                )
                max_abs_logit_error = max(
                    max_abs_logit_error,
                    float(scalar_batched["max_abs_logit_error"]),
                )
                max_abs_value_error = max(
                    max_abs_value_error,
                    float(scalar_batched["max_abs_value_error"]),
                )
                max_abs_hidden_error = max(
                    max_abs_hidden_error,
                    float(scalar_batched["max_abs_hidden_error"]),
                )
                max_logit_tolerance_ratio = max(
                    max_logit_tolerance_ratio,
                    float(scalar_batched["max_logit_tolerance_ratio"]),
                )
                max_value_tolerance_ratio = max(
                    max_value_tolerance_ratio,
                    float(scalar_batched["max_value_tolerance_ratio"]),
                )
                max_hidden_tolerance_ratio = max(
                    max_hidden_tolerance_ratio,
                    float(scalar_batched["max_hidden_tolerance_ratio"]),
                )
                max_scalar_reference_logit_tolerance_ratio = max(
                    max_scalar_reference_logit_tolerance_ratio,
                    float(scalar_reference["max_logit_tolerance_ratio"]),
                )
                max_scalar_reference_value_tolerance_ratio = max(
                    max_scalar_reference_value_tolerance_ratio,
                    float(scalar_reference["max_value_tolerance_ratio"]),
                )
                max_scalar_reference_hidden_tolerance_ratio = max(
                    max_scalar_reference_hidden_tolerance_ratio,
                    float(scalar_reference["max_hidden_tolerance_ratio"]),
                )
                max_batched_reference_logit_tolerance_ratio = max(
                    max_batched_reference_logit_tolerance_ratio,
                    float(batched_reference["max_logit_tolerance_ratio"]),
                )
                max_batched_reference_value_tolerance_ratio = max(
                    max_batched_reference_value_tolerance_ratio,
                    float(batched_reference["max_value_tolerance_ratio"]),
                )
                max_batched_reference_hidden_tolerance_ratio = max(
                    max_batched_reference_hidden_tolerance_ratio,
                    float(batched_reference["max_hidden_tolerance_ratio"]),
                )
                comparison_count += 1
                reference_comparison_count += 1
            # Advance from the independently exercised model API.  Neither
            # rollout adapter can silently become the reference for the other.
            hidden = [tuple(output.next_hidden) for output in reference_outputs]
        scalar_world_digests.append(scalar_hash.hexdigest())
        batched_world_digests.append(batched_hash.hexdigest())
        reference_world_digests.append(reference_hash.hexdigest())
    return {
        "proof_seed_contract": proof_seed_contract,
        "numeric_contract": {
            "relative_tolerance": _D4_NUMERIC_RTOL,
            "absolute_tolerance": _D4_NUMERIC_ATOL,
        },
        "comparison_count": comparison_count,
        "reference_comparison_count": reference_comparison_count,
        "action_mismatch_count": action_mismatch_count,
        "scalar_reference_action_mismatch_count": (
            scalar_reference_action_mismatch_count
        ),
        "batched_reference_action_mismatch_count": (
            batched_reference_action_mismatch_count
        ),
        "max_abs_logit_error": max_abs_logit_error,
        "max_abs_value_error": max_abs_value_error,
        "max_abs_hidden_error": max_abs_hidden_error,
        "max_logit_tolerance_ratio": max_logit_tolerance_ratio,
        "max_value_tolerance_ratio": max_value_tolerance_ratio,
        "max_hidden_tolerance_ratio": max_hidden_tolerance_ratio,
        "scalar_semantic_sha256": _sha256_json(scalar_world_digests),
        "batched_semantic_sha256": _sha256_json(batched_world_digests),
        "reference_semantic_sha256": _sha256_json(reference_world_digests),
        "max_scalar_reference_logit_tolerance_ratio": (
            max_scalar_reference_logit_tolerance_ratio
        ),
        "max_scalar_reference_value_tolerance_ratio": (
            max_scalar_reference_value_tolerance_ratio
        ),
        "max_scalar_reference_hidden_tolerance_ratio": (
            max_scalar_reference_hidden_tolerance_ratio
        ),
        "max_batched_reference_logit_tolerance_ratio": (
            max_batched_reference_logit_tolerance_ratio
        ),
        "max_batched_reference_value_tolerance_ratio": (
            max_batched_reference_value_tolerance_ratio
        ),
        "max_batched_reference_hidden_tolerance_ratio": (
            max_batched_reference_hidden_tolerance_ratio
        ),
    }


def _d4_collector_equivalence(
    scalar: RecurrentRolloutBuffer,
    batched: RecurrentRolloutBuffer,
) -> dict[str, object]:
    scalar_steps = scalar.steps
    batched_steps = batched.steps
    paired_count = min(len(scalar_steps), len(batched_steps))
    semantic_mismatch_count = abs(len(scalar_steps) - len(batched_steps))
    semantic_mismatch_count += sum(
        _d4_collector_semantic_row(left) != _d4_collector_semantic_row(right)
        for left, right in zip(
            scalar_steps[:paired_count],
            batched_steps[:paired_count],
            strict=True,
        )
    )
    return {
        "transition_count": len(scalar_steps),
        "batched_transition_count": len(batched_steps),
        "semantic_mismatch_count": semantic_mismatch_count,
        # This is recomputed from the actual ordered merged batched buffer. It
        # is not copied from either scalar or core-probe evidence.
        "ordered_merge_semantic_sha256": _d4_collector_semantic_sha256(batched),
    }


def _run_fixed_batch_probe(
    *,
    device: str,
    shape: Mapping[str, int],
    repeat_count: int,
    rollout_workers: int = 1,
) -> dict[str, object]:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("D4 production proof requires an available CUDA device")
    if device not in {"cpu", "cuda"}:
        raise ValueError("D4 device must be cpu or cuda")
    if isinstance(repeat_count, bool) or repeat_count < 2:
        raise ValueError("D4 requires at least two timing repeats per mode")
    if isinstance(rollout_workers, bool) or rollout_workers <= 0:
        raise ValueError("D4 rollout_workers must be a positive integer")
    if int(shape["initial_agents"]) > OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY:
        raise ValueError("D4 active rows exceed the canonical batch capacity")
    with _isolated_torch_process_state(
        num_threads=1,
        preserve_cuda_rng=device == "cuda",
    ):
        equivalence = _run_d4_equivalence(device=device, shape=shape)
        (
            collector_model,
            collector_tasks,
            proof_seed_contract,
        ) = _d4_phase_a_collection_inputs(shape=shape)
        scalar_shas: list[str] = []
        scalar_elapsed: list[int] = []
        batched_shas: list[str] = []
        batched_elapsed: list[int] = []
        first_buffers: dict[str, RecurrentRolloutBuffer] = {}
        first_path_facts: dict[str, dict[str, object]] = {}
        timing_order: list[list[str]] = []
        for repeat in range(repeat_count):
            order = ("scalar", "batched") if repeat % 2 == 0 else ("batched", "scalar")
            timing_order.append(list(order))
            for mode in order:
                buffer, path_facts, elapsed_ns = _run_d4_collector_mode(
                    mode=mode,
                    model=collector_model,
                    tasks=collector_tasks,
                    rollout_workers=rollout_workers,
                )
                first_buffers.setdefault(mode, buffer)
                first_path_facts.setdefault(mode, path_facts)
                semantic_sha = str(path_facts["semantic_sha256"])
                if mode == "scalar":
                    scalar_shas.append(semantic_sha)
                    scalar_elapsed.append(elapsed_ns)
                else:
                    batched_shas.append(semantic_sha)
                    batched_elapsed.append(elapsed_ns)
        collector_equivalence = _d4_collector_equivalence(
            first_buffers["scalar"],
            first_buffers["batched"],
        )
        return {
            "device": device,
            "deterministic_cuda_contract": (
                _verify_current_d4_deterministic_cuda_contract()
                if device == "cuda"
                else None
            ),
            "collector_device": "cpu",
            "rollout_workers": rollout_workers,
            "torch_version": str(torch.__version__),
            "shape": dict(shape),
            "batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            "repeat_count": repeat_count,
            "timing_order": timing_order,
            "proof_seed_contract": proof_seed_contract,
            "equivalence": equivalence,
            "scalar": {
                "semantic_sha256": scalar_shas,
                "elapsed_ns": scalar_elapsed,
            },
            "batched": {
                "semantic_sha256": batched_shas,
                "elapsed_ns": batched_elapsed,
            },
            "collector_equivalence": collector_equivalence,
            "scalar_collector_path": first_path_facts["scalar"],
            "batched_collector_path": first_path_facts["batched"],
        }


def _fixed_batch_facts(observed: Mapping[str, object]) -> dict[str, object]:
    scalar = observed["scalar"]
    batched = observed["batched"]
    if not isinstance(scalar, Mapping) or not isinstance(batched, Mapping):
        raise ValueError("D4 raw observations are malformed")
    scalar_elapsed = [int(value) for value in scalar["elapsed_ns"]]
    batched_elapsed = [int(value) for value in batched["elapsed_ns"]]
    scalar_shas = [str(value) for value in scalar["semantic_sha256"]]
    batched_shas = [str(value) for value in batched["semantic_sha256"]]
    repeat_count = int(observed["repeat_count"])
    expected_timing_order = [
        ["scalar", "batched"] if repeat % 2 == 0 else ["batched", "scalar"]
        for repeat in range(repeat_count)
    ]
    timing_order = copy.deepcopy(observed["timing_order"])
    if (
        timing_order != expected_timing_order
        or len(scalar_elapsed) != repeat_count
        or len(batched_elapsed) != repeat_count
        or len(scalar_shas) != repeat_count
        or len(batched_shas) != repeat_count
        or any(value <= 0 for value in (*scalar_elapsed, *batched_elapsed))
    ):
        raise ValueError("D4 timing sample count, order, or duration drifted")
    equivalence = observed["equivalence"]
    if not isinstance(equivalence, Mapping):
        raise ValueError("D4 equivalence observation is malformed")
    collector_equivalence = observed["collector_equivalence"]
    scalar_collector_path = observed["scalar_collector_path"]
    batched_collector_path = observed["batched_collector_path"]
    if (
        not isinstance(collector_equivalence, Mapping)
        or not isinstance(scalar_collector_path, Mapping)
        or not isinstance(batched_collector_path, Mapping)
    ):
        raise ValueError("D4 production collector observations are malformed")
    if (
        scalar_collector_path.get("semantic_sha256") != scalar_shas[0]
        or batched_collector_path.get("semantic_sha256") != batched_shas[0]
    ):
        raise ValueError("D4 production collector digest provenance drifted")
    collector_transition_count = int(collector_equivalence["transition_count"])
    batched_transition_count = int(collector_equivalence["batched_transition_count"])
    if (
        len(set((*scalar_shas, *batched_shas))) != 1
        or collector_equivalence.get("ordered_merge_semantic_sha256") != scalar_shas[0]
        or int(collector_equivalence["semantic_mismatch_count"]) != 0
        or collector_transition_count <= 0
        or batched_transition_count != collector_transition_count
        or int(scalar_collector_path["fixed_batch_step_count"]) != 0
        or int(batched_collector_path["fixed_batch_step_count"])
        != collector_transition_count
    ):
        raise ValueError("D4 production collector equivalence failed")
    if equivalence.get("numeric_contract") != {
        "relative_tolerance": _D4_NUMERIC_RTOL,
        "absolute_tolerance": _D4_NUMERIC_ATOL,
    }:
        raise ValueError("D4 raw numeric contract drifted")
    declared_shape = dict(observed["shape"])
    expected_proof_seed_contract = _d4_proof_seed_contract(shape=declared_shape)
    proof_seed_contract = observed.get("proof_seed_contract")
    if (
        not isinstance(proof_seed_contract, Mapping)
        or dict(proof_seed_contract) != expected_proof_seed_contract
        or equivalence.get("proof_seed_contract") != expected_proof_seed_contract
    ):
        raise ValueError("D4 proof-only seed provenance drifted")
    deterministic_cuda_contract = observed.get("deterministic_cuda_contract")
    if (
        str(observed["device"]) == "cuda"
        and deterministic_cuda_contract != _sealed_d4_deterministic_cuda_contract()
    ) or (
        str(observed["device"]) != "cuda" and deterministic_cuda_contract is not None
    ):
        raise ValueError("D4 deterministic CUDA provenance drifted")
    return {
        "device": str(observed["device"]),
        "deterministic_cuda_contract": copy.deepcopy(deterministic_cuda_contract),
        "proof_seed_contract": copy.deepcopy(expected_proof_seed_contract),
        "collector_device": str(observed["collector_device"]),
        "rollout_workers": int(observed["rollout_workers"]),
        "torch_version": str(observed["torch_version"]),
        "declared_shape": declared_shape,
        "batch_capacity": int(observed["batch_capacity"]),
        "repeat_count": repeat_count,
        "timing_order": timing_order,
        "scalar_timing_sample_count": len(scalar_elapsed),
        "batched_timing_sample_count": len(batched_elapsed),
        "scalar_semantic_sha256": scalar_shas[0],
        "scalar_repeat_semantic_sha256": scalar_shas,
        "batched_semantic_sha256": batched_shas[0],
        "batched_repeat_semantic_sha256": batched_shas,
        "core_scalar_semantic_sha256": str(equivalence["scalar_semantic_sha256"]),
        "core_batched_semantic_sha256": str(equivalence["batched_semantic_sha256"]),
        "reference_semantic_sha256": str(equivalence["reference_semantic_sha256"]),
        "ordered_merge_semantic_sha256": str(
            collector_equivalence["ordered_merge_semantic_sha256"]
        ),
        "collector_transition_count": collector_transition_count,
        "collector_batched_transition_count": batched_transition_count,
        "collector_semantic_mismatch_count": int(
            collector_equivalence["semantic_mismatch_count"]
        ),
        "scalar_fixed_batch_step_count": int(
            scalar_collector_path["fixed_batch_step_count"]
        ),
        "batched_fixed_batch_step_count": int(
            batched_collector_path["fixed_batch_step_count"]
        ),
        "batched_fixed_batch_runtime_sha256": list(
            batched_collector_path["fixed_batch_runtime_sha256"]
        ),
        "numeric_comparison_count": int(equivalence["comparison_count"]),
        "reference_numeric_comparison_count": int(
            equivalence["reference_comparison_count"]
        ),
        "action_mismatch_count": int(equivalence["action_mismatch_count"]),
        "scalar_reference_action_mismatch_count": int(
            equivalence["scalar_reference_action_mismatch_count"]
        ),
        "batched_reference_action_mismatch_count": int(
            equivalence["batched_reference_action_mismatch_count"]
        ),
        "max_abs_logit_error": float(equivalence["max_abs_logit_error"]),
        "max_abs_value_error": float(equivalence["max_abs_value_error"]),
        "max_abs_hidden_error": float(equivalence["max_abs_hidden_error"]),
        "max_logit_tolerance_ratio": float(equivalence["max_logit_tolerance_ratio"]),
        "max_value_tolerance_ratio": float(equivalence["max_value_tolerance_ratio"]),
        "max_hidden_tolerance_ratio": float(equivalence["max_hidden_tolerance_ratio"]),
        "max_scalar_reference_logit_tolerance_ratio": float(
            equivalence["max_scalar_reference_logit_tolerance_ratio"]
        ),
        "max_scalar_reference_value_tolerance_ratio": float(
            equivalence["max_scalar_reference_value_tolerance_ratio"]
        ),
        "max_scalar_reference_hidden_tolerance_ratio": float(
            equivalence["max_scalar_reference_hidden_tolerance_ratio"]
        ),
        "max_batched_reference_logit_tolerance_ratio": float(
            equivalence["max_batched_reference_logit_tolerance_ratio"]
        ),
        "max_batched_reference_value_tolerance_ratio": float(
            equivalence["max_batched_reference_value_tolerance_ratio"]
        ),
        "max_batched_reference_hidden_tolerance_ratio": float(
            equivalence["max_batched_reference_hidden_tolerance_ratio"]
        ),
        "scalar_median_elapsed_ns": int(statistics.median(scalar_elapsed)),
        "batched_median_elapsed_ns": int(statistics.median(batched_elapsed)),
    }


def _produce_bundle(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    evidence_kind: str,
    probe_spec: Mapping[str, object],
    run_probe: Callable[[], Mapping[str, object]],
    reconstruct: Callable[[Mapping[str, object]], dict[str, object]],
    pre_publish_check: Callable[[], object] | None = None,
) -> dict[str, object]:
    readiness = _readiness()
    contract = readiness._contract()
    contract.validate_open_ecology_phase_a_preregistration(preregistration)
    readiness._require_live_source_twice(preregistration, before=True)
    destination_input = Path(output_directory)
    if destination_input.exists() or destination_input.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{evidence_kind} output directory must be new and absent"
        )
    parent_input = destination_input.parent
    if parent_input.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{evidence_kind} output parent must be one real directory"
        )
    parent = parent_input.resolve()
    if not parent.is_dir():
        raise contract.OpenEcologyPhaseAError(
            f"{evidence_kind} output parent must exist"
        )
    destination = parent / destination_input.name
    observed = run_probe()
    if not isinstance(observed, Mapping):
        raise contract.OpenEcologyPhaseAError(
            f"{evidence_kind} observations must be a mapping"
        )
    raw_payloads: Mapping[str, object] = {
        "probe_spec": dict(probe_spec),
        "observations": observed,
    }
    staging = parent / (
        f".{destination.name}.pending-{os.getpid()}-{os.urandom(8).hex()}"
    )
    staging.mkdir()
    raw_root = staging / "raw"
    raw_root.mkdir()
    produced_at_utc = readiness._format_utc(readiness._utc_now())
    raw_files: dict[str, Path] = {}
    for role in sorted(raw_payloads):
        path = raw_root / f"{role.replace('_', '-')}.json"
        readiness._write_new_json(path, raw_payloads[role])
        raw_files[role] = path
    manifest = readiness.build_open_ecology_raw_evidence_manifest(
        preregistration,
        evidence_kind=evidence_kind,
        bundle_root=staging,
        produced_at_utc=produced_at_utc,
        producer_name=_PRODUCER_NAMES[evidence_kind],
        producer_contract=_PRODUCER_CONTRACTS[evidence_kind],
        raw_files=raw_files,
    )
    manifest_path = staging / "raw-evidence-manifest.json"
    readiness._write_new_json(manifest_path, manifest)
    facts = reconstruct(observed)
    readiness._SEMANTIC_VALIDATORS[evidence_kind](
        facts,
        preregistration,
        None,
    )
    report: dict[str, object] = {
        "schema_version": (
            readiness.OPEN_ECOLOGY_PHASE_A_READINESS_REPORT_SCHEMA_VERSION
        ),
        "evidence_kind": evidence_kind,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": readiness._source_binding(preregistration),
        "produced_at_utc": produced_at_utc,
        "raw_evidence_manifest": contract._file_reference(
            manifest_path,
            base=staging,
        ),
        "facts": facts,
    }
    report["exact_digest"] = stable_payload_digest(report)
    readiness._write_new_json(staging / "report.json", report)
    if destination.exists() or destination.is_symlink():
        raise contract.OpenEcologyPhaseAError(
            f"{evidence_kind} destination appeared during production"
        )
    # A failed final source check intentionally leaves the unique pending
    # bundle intact for diagnosis; it is never renamed into authority.
    readiness._require_live_source_twice(preregistration, before=False)
    if pre_publish_check is not None:
        pre_publish_check()
    readiness._rename_path_no_replace(staging, destination)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return report


def _load_verified_raw_observations(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: object,
    *,
    evidence_kind: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    readiness = _readiness()
    root = report_path.parent.absolute()
    root_fd = readiness._open_real_directory(
        root,
        field=f"{evidence_kind} report bundle",
    )
    try:
        report_capture = readiness._capture_regular_file_at(
            root_fd,
            root,
            report_path.name,
            field=f"{evidence_kind} report",
            maximum_bytes=readiness._MAX_AUTHORITY_JSON_BYTES,
        )
    finally:
        os.close(root_fd)
    report = readiness._strict_json_from_captured_bytes(
        report_capture,
        field=f"{evidence_kind} report",
    )
    _manifest_path, manifest, captures = readiness._load_report_raw_evidence_manifest(
        report_capture.path,
        report,
        expected_kind=evidence_kind,
        preregistration=preregistration,
        authorization_time=authorization_time,
    )
    if manifest.get("producer") != {
        "name": _PRODUCER_NAMES[evidence_kind],
        "contract": _PRODUCER_CONTRACTS[evidence_kind],
    }:
        raise readiness._contract().OpenEcologyPhaseAError(
            f"{evidence_kind} raw producer identity drifted"
        )
    if set(captures) != {"observations", "probe_spec"}:
        raise readiness._contract().OpenEcologyPhaseAError(
            f"{evidence_kind} raw evidence roles are incomplete"
        )
    probe_spec = readiness._strict_json_from_captured_bytes(
        captures["probe_spec"],
        field=f"{evidence_kind} raw probe spec",
    )
    observations = readiness._strict_json_from_captured_bytes(
        captures["observations"],
        field=f"{evidence_kind} raw observations",
    )
    return probe_spec, observations


def _authority_result(
    preregistration: Mapping[str, object],
    *,
    evidence_kind: str,
    facts: Mapping[str, object],
) -> dict[str, object]:
    readiness = _readiness()
    return {
        "evidence_kind": evidence_kind,
        "campaign_digest": preregistration["exact_digest"],
        "configuration_sha256": preregistration["configuration_sha256"],
        "source": readiness._source_binding(preregistration),
        "facts": dict(facts),
    }


def produce_cross_surface_and_self_echo_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    return _produce_bundle(
        preregistration,
        output_directory=output_directory,
        evidence_kind=D1_KIND,
        probe_spec={
            "signal_treatment": asdict(OpenEcologySignalTreatment()),
            "scientific_seed_access_count": 0,
        },
        run_probe=_run_cross_surface_probe,
        reconstruct=_cross_surface_facts,
    )


def verify_cross_surface_and_self_echo_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: object,
) -> Mapping[str, object]:
    spec, observed = _load_verified_raw_observations(
        report_path,
        preregistration,
        authorization_time,
        evidence_kind=D1_KIND,
    )
    if spec != {
        "signal_treatment": asdict(OpenEcologySignalTreatment()),
        "scientific_seed_access_count": 0,
    }:
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError("D1 probe specification drifted")
        )
    repeated = _run_cross_surface_probe()
    if observed != repeated:
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError(
                "D1 raw contracts or self-echo behavior did not reproduce"
            )
        )
    return _authority_result(
        preregistration,
        evidence_kind=D1_KIND,
        facts=_cross_surface_facts(observed),
    )


def produce_runtime_genome_and_action_source_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    return _produce_bundle(
        preregistration,
        output_directory=output_directory,
        evidence_kind=D2_KIND,
        probe_spec={
            "heritable_world_identity": "d2-heritable-population",
            "zero_world_identity": "d2-zero-population",
            "scientific_seed_access_count": 0,
        },
        run_probe=_run_runtime_genome_probe,
        reconstruct=_runtime_genome_facts,
    )


def verify_runtime_genome_and_action_source_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: object,
) -> Mapping[str, object]:
    spec, observed = _load_verified_raw_observations(
        report_path,
        preregistration,
        authorization_time,
        evidence_kind=D2_KIND,
    )
    if spec != {
        "heritable_world_identity": "d2-heritable-population",
        "zero_world_identity": "d2-zero-population",
        "scientific_seed_access_count": 0,
    }:
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError("D2 probe specification drifted")
        )
    repeated = _run_runtime_genome_probe()
    if observed != repeated:
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError(
                "D2 runtime genome/action-source behavior did not reproduce"
            )
        )
    return _authority_result(
        preregistration,
        evidence_kind=D2_KIND,
        facts=_runtime_genome_facts(observed),
    )


def produce_critic_gradient_and_density_schedule_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    return _produce_bundle(
        preregistration,
        output_directory=output_directory,
        evidence_kind=D3_KIND,
        probe_spec={
            "device": "cpu",
            "dtype": "float32",
            "initialization_seed": 8203,
            "scientific_seed_access_count": 0,
        },
        run_probe=lambda: _run_critic_gradient_probe(preregistration),
        reconstruct=_critic_gradient_facts,
    )


def verify_critic_gradient_and_density_schedule_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: object,
) -> Mapping[str, object]:
    spec, observed = _load_verified_raw_observations(
        report_path,
        preregistration,
        authorization_time,
        evidence_kind=D3_KIND,
    )
    if spec != {
        "device": "cpu",
        "dtype": "float32",
        "initialization_seed": 8203,
        "scientific_seed_access_count": 0,
    }:
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError("D3 probe specification drifted")
        )
    repeated = _run_critic_gradient_probe(preregistration)
    if observed != repeated:
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError("D3 critic-gradient behavior did not reproduce")
        )
    return _authority_result(
        preregistration,
        evidence_kind=D3_KIND,
        facts=_critic_gradient_facts(observed),
    )


def produce_fixed_batch_equivalence_and_speed_report(
    preregistration: Mapping[str, object],
    *,
    output_directory: str | Path,
    device: str = "cuda",
) -> dict[str, object]:
    if device != "cuda":
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError(
                "authoritative D4 production requires the CUDA runtime"
            )
        )
    deterministic_cuda_contract = _configure_and_verify_d4_deterministic_cuda_contract()
    proof_seed_contract = _d4_proof_seed_contract(shape=_D4_DECLARED_SHAPE)
    rollout_workers = int(
        preregistration["runtime_contract"]["rollout_workers"]  # type: ignore[index]
    )

    def run_deterministic_probe() -> Mapping[str, object]:
        _verify_current_d4_deterministic_cuda_contract()
        result = _run_fixed_batch_probe(
            device=device,
            shape=_D4_DECLARED_SHAPE,
            repeat_count=_D4_REPEAT_COUNT,
            rollout_workers=rollout_workers,
        )
        _verify_current_d4_deterministic_cuda_contract()
        return result

    return _produce_bundle(
        preregistration,
        output_directory=output_directory,
        evidence_kind=D4_KIND,
        probe_spec={
            "device": device,
            "shape": _D4_DECLARED_SHAPE,
            "batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
            "rollout_workers": rollout_workers,
            "repeat_count": _D4_REPEAT_COUNT,
            "semantic_contract": (
                "production_phase_a_collector_ordered_merge_equivalence_v3"
            ),
            "numeric_contract": {
                "relative_tolerance": _D4_NUMERIC_RTOL,
                "absolute_tolerance": _D4_NUMERIC_ATOL,
            },
            "deterministic_cuda_contract": deterministic_cuda_contract,
            "proof_seed_contract": proof_seed_contract,
        },
        run_probe=run_deterministic_probe,
        reconstruct=_fixed_batch_facts,
        pre_publish_check=_verify_current_d4_deterministic_cuda_contract,
    )


def verify_fixed_batch_equivalence_and_speed_report(
    report_path: Path,
    preregistration: Mapping[str, object],
    authorization_time: object,
) -> Mapping[str, object]:
    deterministic_cuda_contract = _configure_and_verify_d4_deterministic_cuda_contract()
    spec, observed = _load_verified_raw_observations(
        report_path,
        preregistration,
        authorization_time,
        evidence_kind=D4_KIND,
    )
    _verify_current_d4_deterministic_cuda_contract()
    expected_spec = {
        "device": "cuda",
        "shape": _D4_DECLARED_SHAPE,
        "batch_capacity": OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY,
        "rollout_workers": int(
            preregistration["runtime_contract"]["rollout_workers"]  # type: ignore[index]
        ),
        "repeat_count": _D4_REPEAT_COUNT,
        "semantic_contract": (
            "production_phase_a_collector_ordered_merge_equivalence_v3"
        ),
        "numeric_contract": {
            "relative_tolerance": _D4_NUMERIC_RTOL,
            "absolute_tolerance": _D4_NUMERIC_ATOL,
        },
        "deterministic_cuda_contract": deterministic_cuda_contract,
        "proof_seed_contract": _d4_proof_seed_contract(shape=_D4_DECLARED_SHAPE),
    }
    if spec != expected_spec or observed.get("device") != "cuda":
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError("D4 probe specification drifted")
        )
    facts = _fixed_batch_facts(observed)
    collector_semantic_values = {
        facts["scalar_semantic_sha256"],
        facts["batched_semantic_sha256"],
        facts["ordered_merge_semantic_sha256"],
        *facts["scalar_repeat_semantic_sha256"],
        *facts["batched_repeat_semantic_sha256"],
    }
    core_semantic_values = {
        facts["core_scalar_semantic_sha256"],
        facts["core_batched_semantic_sha256"],
        facts["reference_semantic_sha256"],
    }
    expected_comparisons = math.prod(_D4_DECLARED_SHAPE.values())
    if (
        len(collector_semantic_values) != 1
        or len(core_semantic_values) != 1
        or facts["device"] != "cuda"
        or facts["collector_device"] != "cpu"
        or facts["rollout_workers"] != expected_spec["rollout_workers"]
        or facts["declared_shape"] != _D4_DECLARED_SHAPE
        or facts["batch_capacity"] != OPEN_ECOLOGY_RECURRENT_FIXED_BATCH_CAPACITY
        or facts["repeat_count"] != _D4_REPEAT_COUNT
        or facts["timing_order"] != [["scalar", "batched"], ["batched", "scalar"]]
        or facts["scalar_timing_sample_count"] != _D4_REPEAT_COUNT
        or facts["batched_timing_sample_count"] != _D4_REPEAT_COUNT
        or len(facts["scalar_repeat_semantic_sha256"]) != _D4_REPEAT_COUNT
        or len(facts["batched_repeat_semantic_sha256"]) != _D4_REPEAT_COUNT
        or facts["numeric_comparison_count"] != expected_comparisons
        or facts["reference_numeric_comparison_count"] != expected_comparisons
        or facts["collector_transition_count"] <= 0
        or facts["collector_batched_transition_count"]
        != facts["collector_transition_count"]
        or facts["collector_semantic_mismatch_count"] != 0
        or facts["scalar_fixed_batch_step_count"] != 0
        or facts["batched_fixed_batch_step_count"]
        != facts["collector_transition_count"]
        or len(facts["batched_fixed_batch_runtime_sha256"]) != 1
        or facts["action_mismatch_count"] != 0
        or facts["scalar_reference_action_mismatch_count"] != 0
        or facts["batched_reference_action_mismatch_count"] != 0
        or any(
            not math.isfinite(float(facts[field])) or float(facts[field]) < 0.0
            for field in (
                "max_abs_logit_error",
                "max_abs_value_error",
                "max_abs_hidden_error",
            )
        )
        or any(
            not math.isfinite(float(facts[field]))
            or not 0.0 <= float(facts[field]) <= 1.0
            for field in (
                "max_logit_tolerance_ratio",
                "max_value_tolerance_ratio",
                "max_hidden_tolerance_ratio",
                "max_scalar_reference_logit_tolerance_ratio",
                "max_scalar_reference_value_tolerance_ratio",
                "max_scalar_reference_hidden_tolerance_ratio",
                "max_batched_reference_logit_tolerance_ratio",
                "max_batched_reference_value_tolerance_ratio",
                "max_batched_reference_hidden_tolerance_ratio",
            )
        )
    ):
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError(
                "D4 raw scalar/batched numerical equivalence differs"
            )
        )
    # Timings remain the captured OS-clock observations.  Reexecute the full
    # declared scalar/batched comparison to independently reproduce semantics
    # and the numerical tolerance boundary from the current exact source.
    with _isolated_torch_process_state(
        num_threads=1,
        preserve_cuda_rng=spec["device"] == "cuda",
    ):
        _verify_current_d4_deterministic_cuda_contract()
        repeated = _run_d4_equivalence(
            device=str(spec["device"]),
            shape=_D4_DECLARED_SHAPE,
        )
        (
            collector_model,
            collector_tasks,
            repeated_proof_seed_contract,
        ) = _d4_phase_a_collection_inputs(shape=_D4_DECLARED_SHAPE)
        repeated_scalar_buffer, repeated_scalar_path, _scalar_elapsed = (
            _run_d4_collector_mode(
                mode="scalar",
                model=collector_model,
                tasks=collector_tasks,
                rollout_workers=int(spec["rollout_workers"]),
            )
        )
        repeated_batched_buffer, repeated_batched_path, _batched_elapsed = (
            _run_d4_collector_mode(
                mode="batched",
                model=collector_model,
                tasks=collector_tasks,
                rollout_workers=int(spec["rollout_workers"]),
            )
        )
        repeated_collector = _d4_collector_equivalence(
            repeated_scalar_buffer,
            repeated_batched_buffer,
        )
        _verify_current_d4_deterministic_cuda_contract()
    if (
        repeated["proof_seed_contract"] != facts["proof_seed_contract"]
        or repeated_proof_seed_contract != facts["proof_seed_contract"]
        or repeated["scalar_semantic_sha256"] != facts["core_scalar_semantic_sha256"]
        or repeated["batched_semantic_sha256"] != facts["core_scalar_semantic_sha256"]
        or repeated["reference_semantic_sha256"] != facts["core_scalar_semantic_sha256"]
        or repeated_scalar_path["semantic_sha256"] != facts["scalar_semantic_sha256"]
        or repeated_batched_path["semantic_sha256"] != facts["batched_semantic_sha256"]
        or repeated_collector["ordered_merge_semantic_sha256"]
        != facts["ordered_merge_semantic_sha256"]
        or repeated_collector["transition_count"] != facts["collector_transition_count"]
        or repeated_collector["batched_transition_count"]
        != facts["collector_batched_transition_count"]
        or repeated_collector["semantic_mismatch_count"] != 0
        or repeated_scalar_path["fixed_batch_step_count"] != 0
        or repeated_batched_path["fixed_batch_step_count"]
        != facts["collector_transition_count"]
        or repeated_batched_path["fixed_batch_runtime_sha256"]
        != facts["batched_fixed_batch_runtime_sha256"]
        or repeated["comparison_count"] != expected_comparisons
        or repeated["reference_comparison_count"] != expected_comparisons
        or repeated["action_mismatch_count"] != 0
        or repeated["scalar_reference_action_mismatch_count"] != 0
        or repeated["batched_reference_action_mismatch_count"] != 0
        or any(
            not math.isfinite(float(repeated[field]))
            or not 0.0 <= float(repeated[field]) <= 1.0
            for field in (
                "max_logit_tolerance_ratio",
                "max_value_tolerance_ratio",
                "max_hidden_tolerance_ratio",
                "max_scalar_reference_logit_tolerance_ratio",
                "max_scalar_reference_value_tolerance_ratio",
                "max_scalar_reference_hidden_tolerance_ratio",
                "max_batched_reference_logit_tolerance_ratio",
                "max_batched_reference_value_tolerance_ratio",
                "max_batched_reference_hidden_tolerance_ratio",
            )
        )
    ):
        raise (
            _readiness()
            ._contract()
            .OpenEcologyPhaseAError(
                "D4 production-shape numerical equivalence did not reproduce"
            )
        )
    _verify_current_d4_deterministic_cuda_contract()
    return _authority_result(
        preregistration,
        evidence_kind=D4_KIND,
        facts=facts,
    )


__all__ = [
    "produce_cross_surface_and_self_echo_report",
    "produce_runtime_genome_and_action_source_report",
    "produce_critic_gradient_and_density_schedule_report",
    "produce_fixed_batch_equivalence_and_speed_report",
    "verify_cross_surface_and_self_echo_report",
    "verify_runtime_genome_and_action_source_report",
    "verify_critic_gradient_and_density_schedule_report",
    "verify_fixed_batch_equivalence_and_speed_report",
]
