"""No-download preparation and isolated CPU validation of compression recipes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal

from edge_ai.compression.pruning import prune_onnx
from edge_ai.compression.quantization import checkpoint_fp16, quantize_onnx
from edge_ai.compression.recipe import CompressionDetails, CompressionRecipe
from edge_ai.compression.verification import (
    reserve_directory,
    verify_prepared_identity,
    write_json,
)
from edge_ai.config import read_yaml
from edge_ai.datasets.manifest import DatasetManifest, select_manifest
from edge_ai.models.variant import ModelVariant
from edge_ai.records.store import ResultStore, new_run_id
from edge_ai.reference import (
    ReferenceConfig,
    load_prepared,
    load_reference_config,
    reference_experiment,
    run_worker,
    with_baseline,
)
from edge_ai.reporting.render import render_reports
from edge_ai.utils.artifacts import ArtifactIdentity, sha256_file, workspace_file
from edge_ai.utils.serialization import from_dict, positive, safe_name, to_dict, validate_types


@dataclass(frozen=True)
class CompressionReferenceConfig:
    experiment: str
    architecture: Literal["mobilenet_v3_small", "mobilenet_v3_large"]
    recipes: tuple[CompressionRecipe, ...]
    calibration_per_class: int = 1

    def __post_init__(self) -> None:
        validate_types(self)
        safe_name(self.experiment, "compression reference experiment")
        positive(self.calibration_per_class, "calibration images per class")
        if not self.recipes or len({item.id for item in self.recipes}) != len(self.recipes):
            raise ValueError("compression reference needs unique recipes")
        if {item.id for item in self.recipes} & {"baseline-pytorch", "baseline-onnxruntime"}:
            raise ValueError("compression recipe id conflicts with a reference baseline")


def load_matrix(path: Path, workspace: Path) -> CompressionReferenceConfig:
    file = workspace_file(
        workspace,
        path.as_posix() if not path.is_absolute() else path.relative_to(workspace).as_posix(),
    )
    if not file.is_relative_to((workspace / "configs").resolve()):
        raise ValueError("public compression matrix must be under configs/")
    return from_dict(CompressionReferenceConfig, read_yaml(file))


def prepare_compression(
    matrix: CompressionReferenceConfig,
    source_prepared: Path,
    workspace: Path,
    reference_config: Path = Path("configs/experiments/reference_cpu.yaml"),
) -> Path:
    reference = load_reference_config(reference_config, workspace)
    assets = load_prepared(source_prepared, reference, workspace)
    matches = [
        item for item in assets["variants"] if item["model"]["architecture"] == matrix.architecture
    ]
    if len(matches) != 1:
        raise ValueError("requested architecture is absent from prepared reference")
    item = matches[0]
    model = from_dict(ModelVariant, item["model"])
    manifest = from_dict(DatasetManifest, assets["manifest"])
    batches = tuple(manifest.batch(workspace, index)[0] for index in range(len(manifest.samples)))
    root = reserve_directory(
        workspace / "artifacts" / "compression" / new_run_id(matrix), workspace
    )
    calibration = None
    if any(recipe.method == "int8" for recipe in matrix.recipes):
        calibration = select_manifest(
            workspace,
            workspace / "data/imagenette2-160/train",
            workspace / "data/imagenette2-160.tgz",
            archive_url=reference.archive_url,
            seed=reference.seed,
            per_class=matrix.calibration_per_class,
            preprocessing=reference.preprocessing,
            split="train",
        )
        if calibration.archive_sha256 != manifest.archive_sha256:
            raise ValueError("calibration archive differs from the verified evaluation archive")
        write_json(root / "calibration.json", calibration)
    descriptors = []
    needed_runtimes = {
        "pytorch" if recipe.method == "fp16" else "onnxruntime" for recipe in matrix.recipes
    }
    for runtime in ("pytorch", "onnxruntime"):
        if runtime not in needed_runtimes:
            continue
        descriptors.append(
            {"model": to_dict(replace(model, runtime=runtime)), "artifact": item[runtime]}
        )
    for recipe in matrix.recipes:
        runtime = "pytorch" if recipe.method == "fp16" else "onnxruntime"
        source = from_dict(ArtifactIdentity, item[runtime])
        destination = root / recipe.id
        if recipe.method == "fp16":
            identity, compression = checkpoint_fp16(
                source,
                model.compression,
                recipe,
                destination,
                workspace=workspace,
            )
            # Check this actual architecture on every evaluation input before publishing the matrix.
            import numpy as np
            import torch

            from edge_ai.models.loading import load_pytorch_model

            variant = replace(
                model,
                id=model.id + "-" + recipe.id,
                precision=recipe.precision,
                compression=compression,
                runtime=runtime,
            )
            torch.set_num_threads(reference.execution.intra_op_threads)
            converted = load_pytorch_model(variant, identity.verify(workspace))
            for batch in batches:
                with torch.inference_mode():
                    if not np.isfinite(converted(torch.from_numpy(batch).half()).numpy()).all():
                        raise ValueError("FP16 classifier produced nonfinite output")
            del converted
        elif recipe.method in ("int8", "int4"):
            identity, compression = quantize_onnx(
                source,
                model.compression,
                recipe,
                destination,
                workspace=workspace,
                verification_batches=batches,
                calibration=calibration if recipe.method == "int8" else None,
                evaluation=manifest,
            )
        else:
            identity, compression = prune_onnx(
                source,
                model.compression,
                recipe,
                destination,
                workspace=workspace,
                verification_batches=batches,
            )
        variant = replace(
            model,
            id=model.id + "-" + recipe.id,
            precision=recipe.precision,
            compression=compression,
            runtime=runtime,
        )
        descriptors.append({"model": to_dict(variant), "artifact": to_dict(identity)})
        print(f"Prepared compression: {recipe.id}", flush=True)
    prepared = root / "prepared.json"
    write_json(
        prepared,
        {
            "matrix": to_dict(matrix),
            "source_prepared": source_prepared.relative_to(workspace).as_posix(),
            "source_prepared_sha256": sha256_file(source_prepared),
            "reference_config": to_dict(reference),
            "reference_config_path": reference_config.as_posix(),
            "manifest": to_dict(manifest),
            "manifest_path": assets["manifest_path"],
            "variants": descriptors,
        },
    )
    write_json(root / "complete.json", {"status": "complete"})
    return prepared


@dataclass(frozen=True)
class PreparedCompression:
    matrix: CompressionReferenceConfig
    reference: ReferenceConfig
    manifest: DatasetManifest
    assets: dict


def load_compression_prepared(
    prepared: Path, workspace: Path, matrix: CompressionReferenceConfig | None = None,
) -> PreparedCompression:
    """Verify the entire source/matrix before benchmarking or diagnostics."""
    file = workspace_file(workspace, prepared.relative_to(workspace).as_posix())
    assets = json.loads(file.read_text(encoding="utf-8"))
    matrix = matrix or from_dict(CompressionReferenceConfig, assets["matrix"])
    if from_dict(CompressionReferenceConfig, assets["matrix"]) != matrix or json.loads(
        (file.parent / "complete.json").read_text(encoding="utf-8")
    ) != {"status": "complete"}:
        raise ValueError("compression preparation differs from matrix or is incomplete")
    reference_path = Path(assets["reference_config_path"])
    reference = load_reference_config(reference_path, workspace)
    if from_dict(ReferenceConfig, assets["reference_config"]) != reference:
        raise ValueError("reference settings differ from compression preparation")
    source_path = workspace_file(workspace, assets["source_prepared"])
    if sha256_file(source_path) != assets["source_prepared_sha256"]:
        raise ValueError("prepared reference identity changed")
    original = load_prepared(source_path, reference, workspace)
    if (
        assets["manifest"] != original["manifest"]
        or assets["manifest_path"] != original["manifest_path"]
    ):
        raise ValueError("compression evaluation differs from original verified manifest")
    manifest = from_dict(DatasetManifest, assets["manifest"])
    validate_descriptors(matrix, assets["variants"], original["variants"], manifest, workspace)
    return PreparedCompression(matrix, reference, manifest, assets)


def run_compression(
    matrix: CompressionReferenceConfig,
    prepared: Path,
    workspace: Path,
) -> list[Path]:
    loaded = load_compression_prepared(prepared, workspace, matrix)
    reference, manifest, assets = loaded.reference, loaded.manifest, loaded.assets
    groups = {}
    started = perf_counter()
    for descriptor in assets["variants"]:
        model = from_dict(ModelVariant, descriptor["model"])
        config = reference_experiment(
            reference, model, from_dict(ArtifactIdentity, descriptor["artifact"]),
            manifest, assets["manifest_path"], experiment=matrix.experiment,
        )
        records = groups.setdefault(model.runtime, [])
        record = run_worker(config, manifest, workspace)
        records.append(with_baseline(record, records[0] if records else None))
        print(f"Profile complete: {model.id} / {model.runtime}", flush=True)
    paths = []
    for records in groups.values():
        reports = render_reports(records)
        paths.extend(ResultStore(workspace).write(record, reports=reports) for record in records)
    print(
        f"Compression reference complete: {len(paths)} runs; "
        f"wall time {perf_counter() - started:.3f} s"
    )
    for path in paths:
        print(path)
    return paths


def validate_descriptors(
    matrix: CompressionReferenceConfig,
    descriptors: list,
    originals: list,
    manifest: DatasetManifest,
    workspace: Path,
) -> None:
    """Validate the whole matrix before starting any measurement worker."""
    matches = [item for item in originals if item["model"]["architecture"] == matrix.architecture]
    if len(matches) != 1:
        raise ValueError("original reference is missing the requested architecture")
    original = matches[0]
    base = from_dict(ModelVariant, original["model"])
    needed = {"pytorch" if recipe.method == "fp16" else "onnxruntime" for recipe in matrix.recipes}
    if len(descriptors) != len(matrix.recipes) + len(needed):
        raise ValueError("prepared compression matrix has a missing/extra variant")
    baselines, recipes = set(), set()
    nominal_fields = (
        "parameters",
        "flops",
        "counting_method",
        "counting_input_shape",
        "counting_exclusions",
    )
    for descriptor in descriptors:
        model = from_dict(ModelVariant, descriptor["model"])
        identity = from_dict(ArtifactIdentity, descriptor["artifact"])
        if model.runtime not in needed:
            raise ValueError("unexpected compression runtime")
        source = from_dict(ArtifactIdentity, original[model.runtime])
        details = model.compression.details
        if details is None:
            if model != replace(base, runtime=model.runtime) or identity != source:
                raise ValueError("FP32 baseline differs from the original verified reference")
            if model.runtime in baselines:
                raise ValueError("duplicated FP32 baseline")
            identity.verify(workspace)
            baselines.add(model.runtime)
            continue
        if details.recipe not in matrix.recipes or details.recipe.id in recipes:
            raise ValueError("unexpected/duplicated compression recipe")
        if model.runtime not in baselines:
            raise ValueError("compression variant precedes its FP32 baseline")
        runtime = "pytorch" if details.recipe.method == "fp16" else "onnxruntime"
        if (
            model.runtime != runtime
            or details.source_artifact != source
            or (identity.checkpoint_sha256 != base.weights_digest)
        ):
            raise ValueError("compression lineage differs from the original reference")
        expected_model = replace(
            base,
            id=base.id + "-" + details.recipe.id,
            precision=details.recipe.precision,
            compression=model.compression,
            runtime=runtime,
        )
        if model != expected_model or any(
            getattr(model.compression, field) != getattr(base.compression, field)
            for field in nominal_fields
        ):
            raise ValueError("compression changed the root model or nominal dense dimensions")
        metadata = identity.export_metadata.get("compression_details")
        if metadata is None or from_dict(CompressionDetails, metadata) != details:
            raise ValueError("prepared artifact and variant compression lineage differ")
        details.validate_evaluation(manifest)
        verify_prepared_identity(identity, workspace)
        recipes.add(details.recipe.id)
    if baselines != needed or recipes != {recipe.id for recipe in matrix.recipes}:
        raise ValueError("compression matrix is missing baseline/recipe coverage")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU compression: preparation and measurement are separate"
    )
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/compression/reference_cpu.yaml")
    )
    parser.add_argument(
        "--reference-config", type=Path, default=Path("configs/experiments/reference_cpu.yaml")
    )
    parser.add_argument("--reference-prepared", type=Path)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    matrix = load_matrix(args.config, workspace)
    if args.command == "prepare":
        if args.reference_prepared is None:
            parser.error("prepare requires existing --reference-prepared; no implicit downloads")
        print(
            prepare_compression(
                matrix,
                (workspace / args.reference_prepared).resolve(),
                workspace,
                args.reference_config,
            )
        )
    else:
        if args.prepared is None:
            parser.error("run requires --prepared from separate compression preparation")
        run_compression(matrix, (workspace / args.prepared).resolve(), workspace)


if __name__ == "__main__":
    main()
