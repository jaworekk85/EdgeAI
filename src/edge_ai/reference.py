"""Public reference configuration, preparation and isolated CPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

from edge_ai.compression.accounting import count_nominal
from edge_ai.config import DeviceSettings, ExecutionSettings, ExperimentConfig, PowerSettings
from edge_ai.datasets.manifest import DatasetManifest, select_manifest
from edge_ai.datasets.preprocessing import PreprocessingSpec
from edge_ai.datasets.workload import Workload
from edge_ai.devices.profile import capture_device_profile
from edge_ai.export.onnx import export_onnx
from edge_ai.models.loading import load_pytorch_model
from edge_ai.models.variant import ModelVariant
from edge_ai.profiling.protocol import ProfilingProtocol
from edge_ai.profiling.runner import profile_cpu
from edge_ai.records.schema import RunRecord
from edge_ai.records.store import ResultStore, new_run_id
from edge_ai.reporting.comparison import baseline_changes
from edge_ai.reporting.render import render_reports
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file, workspace_file
from edge_ai.utils.downloads import download_public, extract_images
from edge_ai.utils.provenance import capture_provenance
from edge_ai.utils.serialization import from_dict, positive, safe_name, to_dict, validate_types


@dataclass(frozen=True)
class ReferenceConfig:
    experiment: str = "reference"
    seed: int = 7
    architectures: tuple[Literal["mobilenet_v3_small", "mobilenet_v3_large"], ...] = (
        "mobilenet_v3_small",
        "mobilenet_v3_large",
    )
    runtimes: tuple[Literal["pytorch", "onnxruntime"], ...] = ("pytorch", "onnxruntime")
    per_class: int = 2
    archive_url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    archive_md5: str = "e793b78cc4c9e9a4ccc0c1155377a412"
    preprocessing: PreprocessingSpec = field(default_factory=PreprocessingSpec)
    protocol: ProfilingProtocol = field(default_factory=lambda: ProfilingProtocol(5, 20, 1))
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    power: PowerSettings = field(default_factory=PowerSettings)
    device: DeviceSettings = field(default_factory=DeviceSettings)

    def __post_init__(self):
        validate_types(self)
        safe_name(self.experiment, "reference experiment")
        positive(self.seed, "seed", zero=True)
        positive(self.per_class, "per_class")
        if not self.architectures or not self.runtimes:
            raise ValueError("reference requires models and runtimes")
        if len(set(self.architectures)) != len(self.architectures) or len(
            set(self.runtimes)
        ) != len(self.runtimes):
            raise ValueError("reference matrix contains duplicates")
        if self.protocol.iterations != self.per_class * 10:
            raise ValueError("iterations must cover the selected validation subset once")
        if self.device.adapter != "cpu" or self.device.requested_power_mode is not None:
            raise ValueError("Phase 2 reference requires the CPU adapter without hardware settings")

    def digest(self):
        return hashlib.sha256(json.dumps(to_dict(self), sort_keys=True).encode()).hexdigest()


def load_reference_config(path: Path, workspace: Path) -> ReferenceConfig:
    # Reuse the config loader's strict duplicate-key handling through a standalone reader.
    from edge_ai.config import read_yaml

    relative = path.relative_to(workspace) if path.is_absolute() else path
    file = workspace_file(workspace, str(relative))
    if not file.is_relative_to((workspace / "configs").resolve()):
        raise ValueError("reference configuration must be under configs/")
    return from_dict(ReferenceConfig, read_yaml(file))


def prepare_reference(config: ReferenceConfig, workspace: Path) -> Path:
    import torch
    from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

    from edge_ai.devices.profile import package_versions

    versions = package_versions(
        ("torch", "torchvision", "onnx", "onnxscript", "onnx_ir", "onnxruntime")
    )
    identity = hashlib.sha256(
        json.dumps(
            {"config": config.digest(), "versions": versions, "preparation_revision": 1},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    parent = workspace / "artifacts" / "reference"
    root = parent / identity[:16]
    for candidate in (root, *sorted(parent.glob(identity[:16] + "_*"))):
        prepared = candidate / "prepared.json"
        if prepared.exists():
            load_prepared(prepared, config, workspace)
            return prepared
    if root.exists():
        # Retain interrupted preparations; reserve a new directory instead of overwriting.
        root = parent / (identity[:16] + "_" + new_run_id(config))
    prepared = root / "prepared.json"
    root.mkdir(parents=True, exist_ok=False)
    archive = download_public(
        config.archive_url, workspace / "data" / "imagenette2-160.tgz", md5=config.archive_md5
    )
    extract_images(archive, workspace / "data")
    manifest = select_manifest(
        workspace,
        workspace / "data" / "imagenette2-160" / "val",
        archive,
        archive_url=config.archive_url,
        seed=config.seed,
        per_class=config.per_class,
        preprocessing=config.preprocessing,
    )
    manifest_path = root / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(to_dict(manifest), stream, indent=2)
    torch.set_num_threads(config.execution.intra_op_threads)
    batches = tuple(manifest.batch(workspace, index)[0] for index in range(len(manifest.samples)))
    variants = []
    for architecture in config.architectures:
        weights = (
            MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if architecture.endswith("small")
            else MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
        filename = weights.url.rsplit("/", 1)[1]
        checkpoint = download_public(
            weights.url,
            workspace / "artifacts" / "checkpoints" / filename,
            sha256_prefix=filename.rsplit("-", 1)[1].split(".")[0],
        )
        checkpoint_hash = sha256_file(checkpoint)
        checkpoint_identity = ArtifactIdentity(
            checkpoint.relative_to(workspace).as_posix(),
            checkpoint_hash,
            "checkpoint",
            checkpoint_hash,
        )
        from edge_ai.compression.spec import CompressionSpec

        variant = ModelVariant(
            architecture + "-fp32",
            "mobilenet_v3",
            "classification",
            weights.url,
            checkpoint_hash,
            "fp32",
            CompressionSpec(),
            config.preprocessing.crop,
            "pytorch",
            architecture,
        )
        model = load_pytorch_model(variant, checkpoint)
        nominal = count_nominal(model, tuple(batches[0].shape))
        variant = replace(variant, compression=nominal)
        onnx_identity = export_onnx(
            model,
            torch.from_numpy(batches[0]),
            root / architecture,
            workspace=workspace,
            checkpoint_sha256=checkpoint_hash,
            verification_batches=batches,
        )
        variants.append(
            {
                "model": to_dict(variant),
                "pytorch": to_dict(checkpoint_identity),
                "onnxruntime": to_dict(onnx_identity),
            }
        )
        del model
    with prepared.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "config": to_dict(config),
                "config_sha256": config.digest(),
                "versions": versions,
                "manifest": to_dict(manifest),
                "manifest_path": manifest_path.relative_to(workspace).as_posix(),
                "variants": variants,
            },
            stream,
            indent=2,
        )
    return prepared


def load_prepared(path: Path, config: ReferenceConfig, workspace: Path) -> dict:
    file = workspace_file(workspace, path.relative_to(workspace).as_posix())
    data = json.loads(file.read_text(encoding="utf-8"))
    stored_digest = hashlib.sha256(json.dumps(data["config"], sort_keys=True).encode()).hexdigest()
    if (
        data["config_sha256"] != stored_digest
        or from_dict(ReferenceConfig, data["config"]) != config
    ):
        raise ValueError("prepared assets differ from reference configuration")
    manifest = from_dict(DatasetManifest, data["manifest"])
    from collections import Counter

    from edge_ai.datasets.manifest import IMAGENETTE_LABELS

    if (
        manifest.seed != config.seed
        or manifest.preprocessing != config.preprocessing
        or manifest.archive_url != config.archive_url
        or Counter(sample.wnid for sample in manifest.samples)
        != {wnid: config.per_class for wnid in IMAGENETTE_LABELS}
    ):
        raise ValueError("prepared manifest differs from the configured stratified subset")
    stored = json.loads(
        workspace_file(workspace, data["manifest_path"]).read_text(encoding="utf-8")
    )
    if stored != data["manifest"]:
        raise ValueError("manifest file differs from prepared identity")
    manifest.verify(workspace)
    if [item["model"]["architecture"] for item in data["variants"]] != list(config.architectures):
        raise ValueError("prepared model matrix differs from reference config")
    for item in data["variants"]:
        from_dict(ModelVariant, item["model"])
        for runtime in config.runtimes:
            from_dict(ArtifactIdentity, item[runtime]).verify(workspace)
    return data


def worker(config: ExperimentConfig, manifest: DatasetManifest, workspace: Path) -> RunRecord:
    manifest.verify(workspace)
    if config.model.runtime == "pytorch":
        from edge_ai.runtimes.pytorch import PyTorchRuntime

        runtime = PyTorchRuntime(workspace, config.artifact, config.execution)
        packages = ("torch", "torchvision", "numpy", "Pillow", "psutil")
    elif config.model.runtime == "onnxruntime":
        from edge_ai.runtimes.onnxruntime import ONNXRuntime

        runtime = ONNXRuntime(workspace, config.artifact, config.execution)
        packages = ("onnxruntime", "numpy", "Pillow", "psutil")
    else:
        raise ValueError("unsupported reference runtime")
    provenance = capture_provenance(workspace, runtime_packages=packages)
    if config.model.runtime == "pytorch":
        provenance.runtime_versions["pytorch"] = provenance.runtime_versions["torch"]
    device = capture_device_profile()
    result = profile_cpu(
        runtime, config, lambda index: manifest.batch(workspace, index % len(manifest.samples))
    )
    return RunRecord(
        run_id=new_run_id(config),
        config=config,
        provenance=provenance,
        device=device,
        timings=result.timings,
        end_to_end_timings=result.end_to_end_timings,
        memory=result.memory,
        run_duration_s=result.run_duration_s,
        power_trace=result.power_trace,
        idle_power_trace=result.idle_power_trace,
        energy=result.energy,
        quality=result.quality,
        quality_metric="top1_accuracy",
        schema_version=3 if result.sustained is not None else 2,
        dataset_manifest=to_dict(manifest),
        sustained=result.sustained,
    )


def run_reference(
    config: ReferenceConfig, prepared: Path, workspace: Path,
    *, sustained_duration_s: float | None = None, sustained_max_iterations: int | None = None,
) -> list[Path]:
    import tempfile

    started = perf_counter()
    assets = load_prepared(prepared, config, workspace)
    if sustained_duration_s is not None or sustained_max_iterations is not None:
        protocol = replace(
            config.protocol,
            sustained_duration_s=(
                sustained_duration_s if sustained_duration_s is not None
                else config.protocol.sustained_duration_s
            ),
            sustained_max_iterations=(
                sustained_max_iterations if sustained_max_iterations is not None
                else config.protocol.sustained_max_iterations
            ),
        )
        if protocol.sustained_duration_s is None:
            raise ValueError("an inference cap override requires sustained mode")
        config = replace(config, protocol=protocol)
    manifest = from_dict(DatasetManifest, assets["manifest"])
    groups = []
    baselines = {}
    with tempfile.TemporaryDirectory(prefix="edge-ai-reference-") as temporary:
        temporary = Path(temporary)
        for backend in config.runtimes:
            records = []
            for item in assets["variants"]:
                model = replace(from_dict(ModelVariant, item["model"]), runtime=backend)
                experiment = ExperimentConfig(
                    experiment=config.experiment,
                    seed=config.seed,
                    model=model,
                    workload=Workload(
                        "classification",
                        "dataset",
                        assets["manifest_path"],
                        config.preprocessing.crop,
                        1,
                        split="val",
                        subset=f"{config.per_class}/class",
                        preprocessing=config.preprocessing,
                        manifest_sha256=manifest.digest(),
                    ),
                    protocol=config.protocol,
                    device=config.device,
                    artifact=from_dict(ArtifactIdentity, item[backend]),
                    execution=config.execution,
                    power=config.power,
                )
                request, response = temporary / "request.json", temporary / "response.json"
                request.write_text(
                    json.dumps({"config": to_dict(experiment), "manifest": to_dict(manifest)}),
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "edge_ai.reference",
                        "worker",
                        "--workspace",
                        str(workspace),
                        "--request",
                        str(request),
                        "--response",
                        str(response),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=110 + (config.protocol.sustained_duration_s or 0),
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"reference worker failed: {completed.stderr}")
                record = from_dict(RunRecord, json.loads(response.read_text(encoding="utf-8")))
                baseline = baselines.get(model.architecture)
                if baseline is None and records:
                    baseline = records[0]
                if baseline is not None:
                    resolved = replace(
                        record.config,
                        baseline_id=baseline.run_id,
                        changed_components=tuple(sorted(baseline_changes(baseline, record))),
                    )
                    record = replace(record, config=resolved, run_id=new_run_id(resolved))
                baselines.setdefault(model.architecture, record)
                records.append(record)
                print(f"Profile complete: {model.architecture} / {backend}", flush=True)
            reports = render_reports(records)
            groups.append((records, reports))
    paths = []
    for records, reports in groups:
        for record in records:
            paths.append(ResultStore(workspace).write(record, reports=reports))
    elapsed = perf_counter() - started
    print(f"Reference complete: {len(paths)} immutable runs; wall time {elapsed:.3f} s", flush=True)
    for path in paths:
        print(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="CPU reference: preparation is separate from runs")
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/reference_cpu.yaml")
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--sustained-seconds", type=float)
    parser.add_argument("--max-inferences", type=int)
    arguments = parser.parse_args()
    if arguments.command != "run" and (
        arguments.sustained_seconds is not None or arguments.max_inferences is not None
    ):
        parser.error("sustained overrides are supported only by run")
    workspace = arguments.workspace.resolve(strict=True)
    if arguments.command == "worker":
        data = json.loads(arguments.request.read_text(encoding="utf-8"))
        record = worker(
            from_dict(ExperimentConfig, data["config"]),
            from_dict(DatasetManifest, data["manifest"]),
            workspace,
        )
        arguments.response.write_text(json.dumps(to_dict(record)), encoding="utf-8")
    else:
        config = load_reference_config(arguments.config, workspace)
        if arguments.command == "prepare":
            started = perf_counter()
            print(prepare_reference(config, workspace))
            print(f"Preparation wall time: {perf_counter() - started:.3f} s")
        else:
            if arguments.prepared is None:
                parser.error("run requires --prepared from the separate prepare command")
            prepared = arguments.prepared
            if not prepared.is_absolute():
                prepared = workspace / prepared
            run_reference(
                config, prepared, workspace, sustained_duration_s=arguments.sustained_seconds,
                sustained_max_iterations=arguments.max_inferences,
            )


if __name__ == "__main__":
    main()
