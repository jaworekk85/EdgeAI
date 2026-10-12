"""Strict YAML configuration with explicit, confined fragment references."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from edge_ai.datasets.workload import Workload
from edge_ai.models.variant import ModelVariant
from edge_ai.profiling.protocol import ProfilingProtocol
from edge_ai.utils.artifacts import ArtifactIdentity
from edge_ai.utils.serialization import from_dict, positive, safe_name, to_dict, validate_types


@dataclass(frozen=True)
class DeviceSettings:
    """Requested settings; actual device observations are captured separately."""

    adapter: Literal["cpu", "mock"] = "cpu"
    requested_power_mode: str | None = None

    def __post_init__(self) -> None:
        validate_types(self)


@dataclass(frozen=True)
class ExecutionSettings:
    intra_op_threads: int = 1
    inter_op_threads: int = 1
    provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"
    ort_execution: Literal["sequential"] = "sequential"
    memory_interval_s: float = 0.01

    def __post_init__(self) -> None:
        validate_types(self)
        for name in ("intra_op_threads", "inter_op_threads", "memory_interval_s"):
            positive(getattr(self, name), name)


@dataclass(frozen=True)
class PowerSettings:
    source: Literal["constant_mock"] = "constant_mock"
    assumed_watts: float = 5.0
    assumed_idle_watts: float = 1.0
    sampling_hz: float = 100.0
    idle_duration_s: float = 0.02

    def __post_init__(self) -> None:
        validate_types(self)
        for name in ("assumed_watts", "assumed_idle_watts"):
            positive(getattr(self, name), name, zero=True)
        positive(self.sampling_hz, "sampling_hz")
        positive(self.idle_duration_s, "idle_duration_s")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    seed: int
    model: ModelVariant
    workload: Workload
    protocol: ProfilingProtocol = field(default_factory=ProfilingProtocol)
    device: DeviceSettings = field(default_factory=DeviceSettings)
    baseline_id: str | None = None
    changed_components: tuple[
        Literal["model", "compression", "runtime", "device_settings"], ...
    ] = ()
    artifact: ArtifactIdentity | None = None
    execution: ExecutionSettings | None = None
    power: PowerSettings | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        safe_name(self.experiment, "experiment")
        positive(self.seed, "seed", zero=True)
        if self.model.task != self.workload.task:
            raise ValueError("model and workload tasks differ")
        if self.model.resolution != self.workload.resolution:
            raise ValueError("model and workload resolutions differ")
        if len(set(self.changed_components)) != len(self.changed_components):
            raise ValueError("changed_components must not contain duplicates")
        if self.baseline_id is None and self.changed_components:
            raise ValueError("changed_components require a baseline_id")
        if self.baseline_id is not None:
            safe_name(self.baseline_id, "baseline_id")
        if self.artifact is not None:
            if self.artifact.checkpoint_sha256 != self.model.weights_digest:
                raise ValueError("deployment artifact and canonical weights identity differ")
            expected_role = {"pytorch": "checkpoint", "onnxruntime": "onnx"}.get(self.model.runtime)
            if expected_role is not None and self.artifact.role != expected_role:
                raise ValueError("artifact role differs from selected runtime")


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a safe mapping with unique string keys."""
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any) -> dict[str, Any]:
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError("YAML mappings require unique string keys")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueKeyLoader.add_constructor("tag:yaml.org,2002:map", mapping)
    with path.open(encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ValueError("config fragment must be a mapping")
    return data


def load_config(
    path: Path,
    *,
    workspace: Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """References are workspace-relative YAML paths in known typed positions.

    Read public fragments from configs/ and private fragments from private/.
    Preserve the complete resolved config through ``to_dict(config)``.
    """
    workspace = workspace.resolve(strict=True)
    public_root = (workspace / "configs").resolve()
    private_root = (workspace / "private").resolve()
    if not public_root.is_relative_to(workspace) or not private_root.is_relative_to(workspace):
        raise ValueError("config roots must remain within the workspace")
    used_private = False

    def read(reference: str | Path) -> dict[str, Any]:
        nonlocal used_private
        candidate = Path(reference)
        candidate = (workspace / candidate).resolve(strict=True)
        is_private = candidate.is_relative_to(private_root)
        if not candidate.is_relative_to(public_root) and not is_private:
            raise ValueError("config path escapes configs/ and private/")
        used_private |= is_private
        if candidate.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("config references must be YAML files")
        return read_yaml(candidate)

    data = read(path)
    for key in ("model", "workload", "device", "protocol"):
        if isinstance(data.get(key), str):
            data[key] = read(data[key])
    model = data.get("model", {})
    if isinstance(model, dict) and isinstance(model.get("compression"), str):
        model["compression"] = read(model["compression"])
    workload = data.get("workload", {})
    if isinstance(workload, dict) and isinstance(workload.get("conditions"), list):
        workload["conditions"] = [
            read(item) if isinstance(item, str) else item for item in workload["conditions"]
        ]
    if overrides is not None:
        data = _merge(data, to_dict(overrides))
    config = from_dict(ExperimentConfig, data)
    if used_private and config.workload.privacy != "private":
        raise ValueError("private fragments require a private workload")
    source = Path(config.workload.source_id)
    if (workspace / source).resolve().is_relative_to(private_root):
        if config.workload.privacy != "private":
            raise ValueError("a source in private/ requires a private workload")
    return config
