"""Serializable CPU compression recipes, coverage and runtime evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from edge_ai.datasets.manifest import DatasetManifest
from edge_ai.utils.artifacts import ArtifactIdentity, validate_digest
from edge_ai.utils.serialization import nonempty, positive, safe_name, validate_types


@dataclass(frozen=True)
class CompressionRecipe:
    id: str
    method: Literal["fp16", "int8", "int4", "unstructured", "channel", "2:4"]
    amount: float | None = None
    block_size: int = 32

    def __post_init__(self) -> None:
        validate_types(self)
        safe_name(self.id, "compression recipe id")
        if self.method in ("unstructured", "channel"):
            if self.amount is None or not 0 < self.amount < 1:
                raise ValueError("magnitude pruning requires an amount in (0, 1)")
        elif self.amount is not None:
            raise ValueError("this method does not accept a pruning amount")
        if self.block_size < 16 or self.block_size & (self.block_size - 1):
            raise ValueError("INT4 block size must be a power of two >= 16")
        if self.method != "int4" and self.block_size != 32:
            raise ValueError("block size applies only to INT4")

    @property
    def precision(self) -> str:
        return self.method if self.method in ("fp16", "int8", "int4") else "fp32"

    @property
    def compression_method(self) -> str:
        if self.method in ("fp16", "int8", "int4"):
            return "quantization"
        return "unstructured_pruning" if self.method == "unstructured" else "structured_pruning"


@dataclass(frozen=True)
class KernelEvidence:
    artifact_sha256: str
    profile_path: str
    profile_sha256: str
    device_name: str
    runtime_version: str
    operator_counts: dict[str, int]
    provider: Literal["CPUExecutionProvider"] = "CPUExecutionProvider"

    def __post_init__(self) -> None:
        validate_types(self)
        for digest in (self.artifact_sha256, self.profile_sha256):
            validate_digest(digest)
        from pathlib import Path

        if (
            not self.profile_path
            or Path(self.profile_path).is_absolute()
            or (".." in Path(self.profile_path).parts)
        ):
            raise ValueError("kernel profile must be a confined relative path")
        nonempty(self.device_name, "kernel probe device")
        nonempty(self.runtime_version, "kernel probe runtime")
        if not self.operator_counts:
            raise ValueError("kernel evidence requires observed quantized CPU operators")
        for name, count in self.operator_counts.items():
            if name not in {
                "QLinearConv",
                "QGemm",
                "QLinearMatMul",
                "MatMulInteger",
                "ConvInteger",
                "MatMulNBits",
            }:
                raise ValueError("kernel evidence contains an unsupported operator family")
            positive(count, "observed operator count")


@dataclass(frozen=True)
class CompressionDetails:
    recipe: CompressionRecipe
    source_artifact: ArtifactIdentity
    changed_tensors: tuple[str, ...]
    target_elements: int
    total_weight_elements: int
    precision_scope: str
    versions: dict[str, str]
    zeros_before: int | None = None
    zeros_after: int | None = None
    masked_elements: int | None = None
    skipped_tensors: tuple[str, ...] = ()
    calibration_manifest: DatasetManifest | None = None
    kernel_evidence: KernelEvidence | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.precision_scope, "compression precision scope")
        positive(self.target_elements, "target weight elements")
        positive(self.total_weight_elements, "total weight elements")
        if self.target_elements > self.total_weight_elements:
            raise ValueError("target weight count exceeds total eligible weight count")
        if not self.changed_tensors or len(set(self.changed_tensors)) != len(self.changed_tensors):
            raise ValueError("compression requires unique changed tensor names")
        if set(self.changed_tensors) & set(self.skipped_tensors):
            raise ValueError("changed and skipped tensors overlap")
        if not self.versions or any(not value for value in self.versions.values()):
            raise ValueError("compression requires tool versions")
        if self.recipe.method in ("unstructured", "channel", "2:4"):
            if any(
                value is None
                for value in (
                    self.zeros_before,
                    self.zeros_after,
                    self.masked_elements,
                )
            ):
                raise ValueError("pruning requires actual zero/mask counts")
            if not 0 <= self.zeros_before < self.zeros_after <= self.target_elements:
                raise ValueError("pruning must add zeros within the declared target scope")
            if not 0 < self.masked_elements <= self.target_elements:
                raise ValueError("pruning mask count is out of bounds")
            if not (
                self.zeros_after - self.zeros_before <= self.masked_elements <= self.zeros_after
            ):
                raise ValueError("pruning zero and mask counts are inconsistent")
            if self.kernel_evidence is not None:
                raise ValueError("dense CPU pruning cannot claim a quantized kernel probe")
        elif any(
            value is not None
            for value in (
                self.zeros_before,
                self.zeros_after,
                self.masked_elements,
            )
        ):
            raise ValueError("zero/mask counts apply only to pruning")
        if self.recipe.method == "int8":
            if self.calibration_manifest is None or self.calibration_manifest.split != "train":
                raise ValueError("static INT8 requires a checksummed training calibration manifest")
        elif self.calibration_manifest is not None:
            raise ValueError("this method does not use calibration")
        if self.recipe.method in ("int8", "int4") and self.kernel_evidence is None:
            raise ValueError("quantized ONNX artifacts require observed CPU kernel evidence")
        if self.kernel_evidence is not None:
            names = set(self.kernel_evidence.operator_counts)
            if self.recipe.method == "int4" and "MatMulNBits" not in names:
                raise ValueError("INT4 requires observed MatMulNBits kernels")
            if self.recipe.method == "int8" and not names - {"MatMulNBits"}:
                raise ValueError("INT8 requires observed integer CPU kernels")
        if self.recipe.method == "fp16" and self.kernel_evidence is not None:
            raise ValueError("FP16 PyTorch kernel internals are not verified by an ORT probe")

    def validate_evaluation(self, evaluation: DatasetManifest) -> None:
        calibration = self.calibration_manifest
        if calibration is None:
            return
        if calibration.preprocessing != evaluation.preprocessing:
            raise ValueError("calibration and evaluation preprocessing differ")
        if (
            {sample.path for sample in calibration.samples}
            & {sample.path for sample in evaluation.samples}
        ) or (
            {sample.sha256 for sample in calibration.samples}
            & {sample.sha256 for sample in evaluation.samples}
        ):
            raise ValueError("calibration and evaluation overlap")
