from __future__ import annotations

from collections.abc import Iterable, Mapping
import ctypes
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys

import torch

from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_counterfactual_auxiliary import (
    RECURRENT_COUNTERFACTUAL_AGGREGATE_AUXILIARY_SCHEMA_VERSION,
)
from evolution_sim.mind.recurrent_counterfactual_branch import (
    RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION,
    RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION,
)
from evolution_sim.mind.recurrent_scale_campaign import (
    RECURRENT_SCALE_CAMPAIGN_POLICY,
    RECURRENT_SCALE_CAMPAIGN_PREREGISTRATION_SCHEMA_VERSION,
    RECURRENT_SCALE_CONCURRENT_CUDA_ARM_PROCESSES,
)
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
)


RECURRENT_SCALE_RUNTIME_PROVENANCE_SCHEMA_VERSION = (
    "mind_v3_public_recurrent_ippo_scale_runtime_provenance_v2"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_DISTRIBUTION_NAME_NORMALIZER = re.compile(r"[-_.]+")
_WORKER_FIELDS = (
    "rollout_workers",
    "counterfactual_workers",
    "evaluation_workers",
)
_DETERMINISM_ENVIRONMENT_FIELDS = (
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


class RecurrentRuntimeProvenanceError(ValueError):
    """Raised when scale runtime provenance is incomplete or has drifted."""


def installed_dependency_freeze(
    distributions: Iterable[metadata.Distribution] | None = None,
) -> dict[str, object]:
    """Return a deterministic, path-free inventory of installed distributions.

    ``pip freeze`` can contain local paths, VCS URLs, credentials, and machine
    topology.  The scale contract needs versions, not those secrets, so this
    inventory deliberately records only canonical distribution names and
    versions.
    """

    installed: dict[str, str] = {}
    source = metadata.distributions() if distributions is None else distributions
    for distribution in source:
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise RecurrentRuntimeProvenanceError(
                "installed distribution is missing its canonical name"
            )
        if not isinstance(raw_version, str) or not raw_version.strip():
            raise RecurrentRuntimeProvenanceError(
                f"installed distribution {raw_name!r} is missing its version"
            )
        name = _canonical_distribution_name(raw_name)
        version = raw_version.strip()
        previous = installed.get(name)
        if previous is not None and previous != version:
            raise RecurrentRuntimeProvenanceError(
                f"installed distribution {name!r} has conflicting versions"
            )
        installed[name] = version

    packages = [
        {"name": name, "version": installed[name]} for name in sorted(installed)
    ]
    if not packages:
        raise RecurrentRuntimeProvenanceError(
            "installed dependency freeze cannot be empty"
        )
    return {
        "contract": "canonical_distribution_name_and_version_only_no_paths_v1",
        "package_count": len(packages),
        "packages": packages,
        "packages_sha256": stable_payload_digest(packages),
    }


def build_recurrent_scale_runtime_provenance(
    *,
    source_commit: str,
    source_manifest_sha256: str,
    preregistration_digest: str,
    repository_clean: bool,
    device: torch.device | str,
    rollout_workers: int,
    counterfactual_workers: int,
    evaluation_workers: int,
    dependency_freeze: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind one campaign runtime to source, dependencies, hardware, and workers.

    The payload intentionally excludes time, host name, user name, paths, PIDs,
    network identity, device UUIDs, and environment variables unrelated to
    deterministic execution.  Repeated capture in an unchanged environment is
    therefore byte-stable and suitable for resume checks.
    """

    parsed_commit = _commit(source_commit, field="source_commit")
    parsed_manifest = _sha256(
        source_manifest_sha256,
        field="source_manifest_sha256",
    )
    parsed_preregistration = _sha256(
        preregistration_digest,
        field="preregistration_digest",
    )
    if repository_clean is not True:
        raise RecurrentRuntimeProvenanceError(
            "runtime provenance requires an exact clean repository"
        )
    worker_contract = {
        "rollout_workers": _positive_int(
            rollout_workers,
            field="rollout_workers",
        ),
        "counterfactual_workers": _positive_int(
            counterfactual_workers,
            field="counterfactual_workers",
        ),
        "evaluation_workers": _positive_int(
            evaluation_workers,
            field="evaluation_workers",
        ),
        "torch_intraop_threads": _positive_int(
            torch.get_num_threads(),
            field="torch_intraop_threads",
        ),
        "torch_interop_threads": _positive_int(
            torch.get_num_interop_threads(),
            field="torch_interop_threads",
        ),
        "logical_cpu_count": _optional_positive_int(os.cpu_count()),
        "worker_process_start_method": "spawn",
        "concurrent_cuda_arm_processes": (
            RECURRENT_SCALE_CONCURRENT_CUDA_ARM_PROCESSES
        ),
    }
    freeze = (
        installed_dependency_freeze()
        if dependency_freeze is None
        else _validated_dependency_freeze(dependency_freeze)
    )
    resolved_device = torch.device(device)
    payload: dict[str, object] = {
        "schema_version": RECURRENT_SCALE_RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "contract_binding": {
            "preregistration_digest": parsed_preregistration,
            "preregistration_schema_version": (
                RECURRENT_SCALE_CAMPAIGN_PREREGISTRATION_SCHEMA_VERSION
            ),
            "campaign_policy": RECURRENT_SCALE_CAMPAIGN_POLICY,
            "seed_registry_version": SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
            "seed_registry_sha256": SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
            "counterfactual_collection_contract_version": (
                RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION
            ),
            "counterfactual_source_branch_schema_version": (
                RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION
            ),
            "counterfactual_aggregate_schema_version": (
                RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION
            ),
            "counterfactual_aggregate_auxiliary_schema_version": (
                RECURRENT_COUNTERFACTUAL_AGGREGATE_AUXILIARY_SCHEMA_VERSION
            ),
            "counterfactual_rng_retape_boundary": (
                RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY
            ),
            "source_commit": parsed_commit,
            "source_manifest_sha256": parsed_manifest,
            "repository_clean": True,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "byte_order": sys.byteorder,
            "pointer_bits": struct.calcsize("P") * 8,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "torch": _torch_runtime_payload(),
        "device": _device_payload(resolved_device),
        "nvidia": _nvidia_runtime_payload(resolved_device),
        "dependency_freeze": freeze,
        "workers": worker_contract,
        "determinism_environment": {
            field: os.environ.get(field) for field in _DETERMINISM_ENVIRONMENT_FIELDS
        },
        "privacy_contract": {
            "hostname_recorded": False,
            "username_recorded": False,
            "filesystem_paths_recorded": False,
            "network_identity_recorded": False,
            "device_uuid_recorded": False,
            "environment_allowlist_only": list(_DETERMINISM_ENVIRONMENT_FIELDS),
        },
    }
    payload["exact_digest"] = stable_payload_digest(payload)
    validate_recurrent_scale_runtime_provenance(payload)
    return payload


def validate_recurrent_scale_runtime_provenance(
    provenance: Mapping[str, object],
) -> None:
    if not isinstance(provenance, Mapping):
        raise RecurrentRuntimeProvenanceError("runtime provenance must be a mapping")
    _exact_keys(
        provenance,
        {
            "schema_version",
            "contract_binding",
            "python",
            "platform",
            "torch",
            "device",
            "nvidia",
            "dependency_freeze",
            "workers",
            "determinism_environment",
            "privacy_contract",
            "exact_digest",
        },
        field="runtime provenance",
    )
    if provenance.get("schema_version") != (
        RECURRENT_SCALE_RUNTIME_PROVENANCE_SCHEMA_VERSION
    ):
        raise RecurrentRuntimeProvenanceError(
            "runtime provenance schema version drifted"
        )
    payload = dict(provenance)
    supplied_digest = _sha256(
        payload.pop("exact_digest", None),
        field="exact_digest",
    )
    if stable_payload_digest(payload) != supplied_digest:
        raise RecurrentRuntimeProvenanceError(
            "runtime provenance exact digest mismatched"
        )

    binding = _mapping(provenance.get("contract_binding"), field="contract_binding")
    _exact_keys(
        binding,
        {
            "preregistration_digest",
            "preregistration_schema_version",
            "campaign_policy",
            "seed_registry_version",
            "seed_registry_sha256",
            "counterfactual_collection_contract_version",
            "counterfactual_source_branch_schema_version",
            "counterfactual_aggregate_schema_version",
            "counterfactual_aggregate_auxiliary_schema_version",
            "counterfactual_rng_retape_boundary",
            "source_commit",
            "source_manifest_sha256",
            "repository_clean",
        },
        field="contract_binding",
    )
    _sha256(
        binding.get("preregistration_digest"),
        field="contract_binding.preregistration_digest",
    )
    expected_binding = {
        "preregistration_schema_version": (
            RECURRENT_SCALE_CAMPAIGN_PREREGISTRATION_SCHEMA_VERSION
        ),
        "campaign_policy": RECURRENT_SCALE_CAMPAIGN_POLICY,
        "seed_registry_version": SCALE_DEVELOPMENT_V2_SEED_REGISTRY_VERSION,
        "seed_registry_sha256": SCALE_DEVELOPMENT_V2_CANONICAL_SHA256,
        "counterfactual_collection_contract_version": (
            RECURRENT_COUNTERFACTUAL_MULTI_TAPE_COLLECTION_CONTRACT_VERSION
        ),
        "counterfactual_source_branch_schema_version": (
            RECURRENT_COUNTERFACTUAL_BOUNDARY_SOURCE_BRANCH_SCHEMA_VERSION
        ),
        "counterfactual_aggregate_schema_version": (
            RECURRENT_COUNTERFACTUAL_AGGREGATE_SCHEMA_VERSION
        ),
        "counterfactual_aggregate_auxiliary_schema_version": (
            RECURRENT_COUNTERFACTUAL_AGGREGATE_AUXILIARY_SCHEMA_VERSION
        ),
        "counterfactual_rng_retape_boundary": (
            RECURRENT_COUNTERFACTUAL_CONTINUATION_RNG_RETAPE_BOUNDARY
        ),
    }
    if any(binding.get(field) != value for field, value in expected_binding.items()):
        raise RecurrentRuntimeProvenanceError(
            "runtime provenance v2 campaign contract binding drifted"
        )
    _commit(
        binding.get("source_commit"),
        field="contract_binding.source_commit",
    )
    _sha256(
        binding.get("source_manifest_sha256"),
        field="contract_binding.source_manifest_sha256",
    )
    if binding.get("repository_clean") is not True:
        raise RecurrentRuntimeProvenanceError(
            "runtime provenance repository must be clean"
        )

    python_runtime = _mapping(provenance.get("python"), field="python")
    _exact_keys(
        python_runtime,
        {"implementation", "version", "cache_tag", "byte_order", "pointer_bits"},
        field="python",
    )
    for field in ("implementation", "version", "cache_tag"):
        _nonempty_string(python_runtime.get(field), field=f"python.{field}")
    if python_runtime.get("byte_order") not in {"little", "big"}:
        raise RecurrentRuntimeProvenanceError("python.byte_order is invalid")
    if python_runtime.get("pointer_bits") not in {32, 64}:
        raise RecurrentRuntimeProvenanceError("python.pointer_bits is invalid")

    platform_runtime = _mapping(provenance.get("platform"), field="platform")
    _exact_keys(
        platform_runtime,
        {"system", "release", "machine", "libc"},
        field="platform",
    )
    for field in ("system", "release", "machine"):
        _nonempty_string(platform_runtime.get(field), field=f"platform.{field}")
    libc = platform_runtime.get("libc")
    if (
        not isinstance(libc, list)
        or len(libc) != 2
        or not all(isinstance(value, str) for value in libc)
    ):
        raise RecurrentRuntimeProvenanceError(
            "platform.libc must contain name and version strings"
        )

    _validate_torch_runtime(provenance.get("torch"))
    device_payload = _mapping(provenance.get("device"), field="device")
    _validate_device(device_payload)
    _validate_nvidia_runtime(
        provenance.get("nvidia"),
        device_type=_nonempty_string(device_payload.get("type"), field="device.type"),
    )

    freeze = _mapping(
        provenance.get("dependency_freeze"),
        field="dependency_freeze",
    )
    _validated_dependency_freeze(freeze)
    workers = _mapping(provenance.get("workers"), field="workers")
    _exact_keys(
        workers,
        {
            *_WORKER_FIELDS,
            "torch_intraop_threads",
            "torch_interop_threads",
            "logical_cpu_count",
            "worker_process_start_method",
            "concurrent_cuda_arm_processes",
        },
        field="workers",
    )
    for field in _WORKER_FIELDS:
        _positive_int(workers.get(field), field=f"workers.{field}")
    for field in ("torch_intraop_threads", "torch_interop_threads"):
        _positive_int(workers.get(field), field=f"workers.{field}")
    _optional_positive_int(workers.get("logical_cpu_count"))
    if workers.get("worker_process_start_method") != "spawn":
        raise RecurrentRuntimeProvenanceError(
            "runtime worker process start method drifted"
        )
    if workers.get("concurrent_cuda_arm_processes") != (
        RECURRENT_SCALE_CONCURRENT_CUDA_ARM_PROCESSES
    ):
        raise RecurrentRuntimeProvenanceError(
            "runtime concurrent CUDA arm-process topology drifted"
        )

    determinism_environment = _mapping(
        provenance.get("determinism_environment"),
        field="determinism_environment",
    )
    _exact_keys(
        determinism_environment,
        set(_DETERMINISM_ENVIRONMENT_FIELDS),
        field="determinism_environment",
    )
    if any(
        value is not None and not isinstance(value, str)
        for value in determinism_environment.values()
    ):
        raise RecurrentRuntimeProvenanceError(
            "determinism environment values must be strings or null"
        )

    privacy = _mapping(provenance.get("privacy_contract"), field="privacy_contract")
    _exact_keys(
        privacy,
        {
            "hostname_recorded",
            "username_recorded",
            "filesystem_paths_recorded",
            "network_identity_recorded",
            "device_uuid_recorded",
            "environment_allowlist_only",
        },
        field="privacy_contract",
    )
    for field in (
        "hostname_recorded",
        "username_recorded",
        "filesystem_paths_recorded",
        "network_identity_recorded",
        "device_uuid_recorded",
    ):
        if privacy.get(field) is not False:
            raise RecurrentRuntimeProvenanceError(
                f"runtime privacy field {field!r} must remain false"
            )
    if privacy.get("environment_allowlist_only") != list(
        _DETERMINISM_ENVIRONMENT_FIELDS
    ):
        raise RecurrentRuntimeProvenanceError(
            "runtime privacy environment allowlist drifted"
        )


def assert_recurrent_scale_runtime_provenance_match(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> None:
    """Fail closed when a resumed cell moves to a different runtime contract."""

    validate_recurrent_scale_runtime_provenance(expected)
    validate_recurrent_scale_runtime_provenance(observed)
    if dict(expected) != dict(observed):
        raise RecurrentRuntimeProvenanceError(
            "runtime provenance drifted from the pinned campaign environment"
        )


def _torch_runtime_payload() -> dict[str, object]:
    cudnn = getattr(torch.backends, "cudnn", None)
    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul = getattr(cuda_backend, "matmul", None)
    return {
        "version": str(torch.__version__),
        "build_git_version": getattr(torch.version, "git_version", None),
        "debug_build": getattr(torch.version, "debug", None),
        "cuda_build_version": getattr(torch.version, "cuda", None),
        "hip_build_version": getattr(torch.version, "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cudnn_available": bool(cudnn is not None and cudnn.is_available()),
        "cudnn_version": None if cudnn is None else cudnn.version(),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": None if cudnn is None else bool(cudnn.deterministic),
        "cudnn_benchmark": None if cudnn is None else bool(cudnn.benchmark),
        "cudnn_allow_tf32": None if cudnn is None else bool(cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": None if matmul is None else bool(matmul.allow_tf32),
    }


def _device_payload(device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RecurrentRuntimeProvenanceError(
                "CUDA runtime provenance requested but CUDA is unavailable"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RecurrentRuntimeProvenanceError(
                f"CUDA device index {index} is unavailable"
            )
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        return {
            "type": "cuda",
            "index": index,
            "name": properties.name,
            "compute_capability": list(capability),
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RecurrentRuntimeProvenanceError(
                "MPS runtime provenance requested but MPS is unavailable"
            )
        return {"type": "mps", "index": device.index, "name": "mps"}
    if device.type == "cpu":
        return {"type": "cpu", "index": device.index, "name": "cpu"}
    raise RecurrentRuntimeProvenanceError(
        f"unsupported runtime device type: {device.type!r}"
    )


def _nvidia_runtime_payload(device: torch.device) -> dict[str, object]:
    empty = {
        "required": False,
        "driver_version": None,
        "cuda_driver_api_version": None,
        "cuda_driver_api_version_raw": None,
        "cuda_runtime_api_version": None,
        "cuda_runtime_api_version_raw": None,
    }
    if device.type != "cuda":
        return empty
    if not torch.cuda.is_available():
        raise RecurrentRuntimeProvenanceError(
            "NVIDIA runtime provenance requested but CUDA is unavailable"
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecurrentRuntimeProvenanceError(
            "NVIDIA driver version query failed"
        ) from error
    driver_versions = {
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    }
    if len(driver_versions) != 1:
        raise RecurrentRuntimeProvenanceError(
            "NVIDIA driver query did not return exactly one version"
        )
    driver_version = next(iter(driver_versions))
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver_version) is None:
        raise RecurrentRuntimeProvenanceError(
            "NVIDIA driver version is malformed"
        )

    try:
        driver_library = ctypes.CDLL("libcuda.so.1")
        driver_raw = ctypes.c_int()
        driver_status = driver_library.cuDriverGetVersion(
            ctypes.byref(driver_raw)
        )
    except (AttributeError, OSError) as error:
        raise RecurrentRuntimeProvenanceError(
            "CUDA driver API version query failed"
        ) from error
    if driver_status != 0 or driver_raw.value <= 0:
        raise RecurrentRuntimeProvenanceError(
            "CUDA driver API returned an invalid version"
        )

    torch.cuda.synchronize(index)
    maps_path = Path("/proc/self/maps")
    try:
        mapped_paths = {
            line.split()[-1]
            for line in maps_path.read_text(encoding="utf-8").splitlines()
            if "libcudart.so" in line and "/" in line
        }
    except OSError as error:
        raise RecurrentRuntimeProvenanceError(
            "CUDA runtime library inventory is unavailable"
        ) from error
    runtime_versions: set[int] = set()
    for mapped_path in mapped_paths:
        try:
            runtime_library = ctypes.CDLL(mapped_path)
            runtime_raw = ctypes.c_int()
            runtime_status = runtime_library.cudaRuntimeGetVersion(
                ctypes.byref(runtime_raw)
            )
        except (AttributeError, OSError) as error:
            raise RecurrentRuntimeProvenanceError(
                "CUDA runtime API version query failed"
            ) from error
        if runtime_status != 0 or runtime_raw.value <= 0:
            raise RecurrentRuntimeProvenanceError(
                "CUDA runtime API returned an invalid version"
            )
        runtime_versions.add(runtime_raw.value)
    if len(runtime_versions) != 1:
        raise RecurrentRuntimeProvenanceError(
            "CUDA runtime API did not resolve to one loaded version"
        )
    runtime_raw_value = next(iter(runtime_versions))
    return {
        "required": True,
        "driver_version": driver_version,
        "cuda_driver_api_version": _cuda_api_version_string(driver_raw.value),
        "cuda_driver_api_version_raw": driver_raw.value,
        "cuda_runtime_api_version": _cuda_api_version_string(runtime_raw_value),
        "cuda_runtime_api_version_raw": runtime_raw_value,
    }


def _cuda_api_version_string(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentRuntimeProvenanceError(
            "CUDA API version must be a positive integer"
        )
    return f"{value // 1000}.{(value % 1000) // 10}"


def _validated_dependency_freeze(
    value: Mapping[str, object],
) -> dict[str, object]:
    freeze = dict(value)
    _exact_keys(
        freeze,
        {"contract", "package_count", "packages", "packages_sha256"},
        field="dependency_freeze",
    )
    if freeze.get("contract") != (
        "canonical_distribution_name_and_version_only_no_paths_v1"
    ):
        raise RecurrentRuntimeProvenanceError("dependency-freeze contract drifted")
    packages = freeze.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RecurrentRuntimeProvenanceError(
            "dependency-freeze packages must be a non-empty list"
        )
    normalized: list[dict[str, str]] = []
    previous_name: str | None = None
    for index, raw_package in enumerate(packages):
        package = _mapping(raw_package, field=f"dependency_freeze.packages[{index}]")
        _exact_keys(
            package,
            {"name", "version"},
            field=f"dependency_freeze.packages[{index}]",
        )
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or name != _canonical_distribution_name(name):
            raise RecurrentRuntimeProvenanceError(
                "dependency-freeze package name is not canonical"
            )
        if (
            not isinstance(version, str)
            or not version
            or any(token in version for token in ("/", "\\", "@", "://"))
        ):
            raise RecurrentRuntimeProvenanceError(
                "dependency-freeze package version must be non-secret and path-free"
            )
        if previous_name is not None and name <= previous_name:
            raise RecurrentRuntimeProvenanceError(
                "dependency-freeze packages must be uniquely name-sorted"
            )
        previous_name = name
        normalized.append({"name": name, "version": version})
    if freeze.get("package_count") != len(normalized):
        raise RecurrentRuntimeProvenanceError(
            "dependency-freeze package count mismatched"
        )
    digest = _sha256(
        freeze.get("packages_sha256"),
        field="dependency_freeze.packages_sha256",
    )
    if stable_payload_digest(normalized) != digest:
        raise RecurrentRuntimeProvenanceError(
            "dependency-freeze package digest mismatched"
        )
    return freeze


def _validate_torch_runtime(value: object) -> None:
    runtime = _mapping(value, field="torch")
    boolean_fields = {
        "cuda_available",
        "cudnn_available",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
    }
    optional_boolean_fields = {
        "debug_build",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
    }
    optional_string_fields = {
        "build_git_version",
        "cuda_build_version",
        "hip_build_version",
    }
    _exact_keys(
        runtime,
        {
            "version",
            "cuda_device_count",
            "cudnn_version",
            *boolean_fields,
            *optional_boolean_fields,
            *optional_string_fields,
        },
        field="torch",
    )
    _nonempty_string(runtime.get("version"), field="torch.version")
    for field in boolean_fields:
        if not isinstance(runtime.get(field), bool):
            raise RecurrentRuntimeProvenanceError(f"torch.{field} must be boolean")
    for field in optional_boolean_fields:
        if runtime.get(field) is not None and not isinstance(runtime.get(field), bool):
            raise RecurrentRuntimeProvenanceError(
                f"torch.{field} must be boolean or null"
            )
    for field in optional_string_fields:
        if runtime.get(field) is not None:
            _nonempty_string(runtime.get(field), field=f"torch.{field}")
    device_count = runtime.get("cuda_device_count")
    if (
        isinstance(device_count, bool)
        or not isinstance(device_count, int)
        or device_count < 0
    ):
        raise RecurrentRuntimeProvenanceError(
            "torch.cuda_device_count must be a non-negative integer"
        )
    cudnn_version = runtime.get("cudnn_version")
    if cudnn_version is not None and (
        isinstance(cudnn_version, bool)
        or not isinstance(cudnn_version, int)
        or cudnn_version <= 0
    ):
        raise RecurrentRuntimeProvenanceError(
            "torch.cudnn_version must be a positive integer or null"
        )


def _validate_device(value: object) -> None:
    device = _mapping(value, field="device")
    device_type = device.get("type")
    if device_type == "cuda":
        _exact_keys(
            device,
            {
                "type",
                "index",
                "name",
                "compute_capability",
                "total_memory_bytes",
                "multiprocessor_count",
            },
            field="device",
        )
        index = device.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RecurrentRuntimeProvenanceError(
                "CUDA device index must be a non-negative integer"
            )
        _nonempty_string(device.get("name"), field="device.name")
        capability = device.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(part, bool) or not isinstance(part, int) or part < 0
                for part in capability
            )
        ):
            raise RecurrentRuntimeProvenanceError(
                "CUDA compute capability must contain two non-negative integers"
            )
        _positive_int(
            device.get("total_memory_bytes"),
            field="device.total_memory_bytes",
        )
        _positive_int(
            device.get("multiprocessor_count"),
            field="device.multiprocessor_count",
        )
        return
    if device_type in {"cpu", "mps"}:
        _exact_keys(device, {"type", "index", "name"}, field="device")
        index = device.get("index")
        if index is not None and (
            isinstance(index, bool) or not isinstance(index, int) or index < 0
        ):
            raise RecurrentRuntimeProvenanceError(
                "runtime device index must be non-negative or null"
            )
        if device.get("name") != device_type:
            raise RecurrentRuntimeProvenanceError(
                "runtime device name/type contract drifted"
            )
        return
    raise RecurrentRuntimeProvenanceError(
        "runtime device type must be cpu, mps, or cuda"
    )


def _validate_nvidia_runtime(value: object, *, device_type: str) -> None:
    runtime = _mapping(value, field="nvidia")
    _exact_keys(
        runtime,
        {
            "required",
            "driver_version",
            "cuda_driver_api_version",
            "cuda_driver_api_version_raw",
            "cuda_runtime_api_version",
            "cuda_runtime_api_version_raw",
        },
        field="nvidia",
    )
    if device_type != "cuda":
        if runtime != {
            "required": False,
            "driver_version": None,
            "cuda_driver_api_version": None,
            "cuda_driver_api_version_raw": None,
            "cuda_runtime_api_version": None,
            "cuda_runtime_api_version_raw": None,
        }:
            raise RecurrentRuntimeProvenanceError(
                "non-CUDA runtime cannot carry NVIDIA version evidence"
            )
        return
    if runtime.get("required") is not True:
        raise RecurrentRuntimeProvenanceError(
            "CUDA runtime requires NVIDIA version evidence"
        )
    driver_version = _nonempty_string(
        runtime.get("driver_version"),
        field="nvidia.driver_version",
    )
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", driver_version) is None:
        raise RecurrentRuntimeProvenanceError(
            "NVIDIA driver version is malformed"
        )
    for prefix in ("cuda_driver_api", "cuda_runtime_api"):
        raw = _positive_int(
            runtime.get(f"{prefix}_version_raw"),
            field=f"nvidia.{prefix}_version_raw",
        )
        if runtime.get(f"{prefix}_version") != _cuda_api_version_string(raw):
            raise RecurrentRuntimeProvenanceError(
                f"NVIDIA {prefix} version string/raw value drifted"
            )


def _canonical_distribution_name(value: str) -> str:
    canonical = _DISTRIBUTION_NAME_NORMALIZER.sub("-", value.strip()).lower()
    if not canonical:
        raise RecurrentRuntimeProvenanceError(
            "distribution name cannot normalize to empty"
        )
    return canonical


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise RecurrentRuntimeProvenanceError(
            f"{field} field set drifted: expected {sorted(expected)!r}, "
            f"got {sorted(observed)!r}"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecurrentRuntimeProvenanceError(f"{field} must be a mapping")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrentRuntimeProvenanceError(f"{field} must be a non-empty string")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecurrentRuntimeProvenanceError(f"{field} must be lowercase SHA-256")
    return value


def _commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise RecurrentRuntimeProvenanceError(
            f"{field} must be a full lowercase Git commit"
        )
    return value


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecurrentRuntimeProvenanceError(f"{field} must be a positive integer")
    return value


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field="logical_cpu_count")


__all__ = [
    "RECURRENT_SCALE_RUNTIME_PROVENANCE_SCHEMA_VERSION",
    "RecurrentRuntimeProvenanceError",
    "assert_recurrent_scale_runtime_provenance_match",
    "build_recurrent_scale_runtime_provenance",
    "installed_dependency_freeze",
    "validate_recurrent_scale_runtime_provenance",
]
