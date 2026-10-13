"""Compression metadata and typed transformation/runtime evidence."""

from dataclasses import dataclass
from typing import Literal

from edge_ai.compression.recipe import CompressionDetails
from edge_ai.utils.serialization import nonempty, positive, validate_types


@dataclass(frozen=True)
class CompressionSpec:
    method: Literal[
        "none",
        "quantization",
        "structured_pruning",
        "unstructured_pruning",
        "distillation",
        "smaller_architecture",
    ] = "none"
    precision: str = "fp32"
    calibration: str | None = None
    sparsity_level: float | None = None
    sparsity_pattern: str | None = None
    kernel_support: bool | None = None
    kernel_support_method: str | None = None
    parameters: int | None = None
    flops: float | None = None
    counting_method: str | None = None
    counting_input_shape: tuple[int, ...] | None = None
    counting_exclusions: tuple[str, ...] = ()
    details: CompressionDetails | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.precision, "precision")
        if self.details is not None:
            recipe = self.details.recipe
            if self.method != recipe.compression_method or self.precision != recipe.precision:
                raise ValueError("compression metadata differs from its transformation recipe")
            if "pruning" in self.method:
                expected_pattern = {
                    "unstructured": "unstructured", "channel": "channel", "2:4": "2:4",
                }[recipe.method]
                if self.sparsity_pattern != expected_pattern or self.kernel_support is not False:
                    raise ValueError("CPU pruning requires its pattern and no sparse kernels")
                expected = self.details.zeros_after / self.details.target_elements
                if self.sparsity_level is None or abs(self.sparsity_level - expected) > 1e-12:
                    raise ValueError("sparsity differs from actual target-weight zero counts")
            elif recipe.method in ("int8", "int4"):
                if self.kernel_support is not True:
                    raise ValueError("quantized CPU operator evidence must be declared")
            elif self.kernel_support is not None:
                raise ValueError("FP16 internal kernel precision remains unknown")
            expected_calibration = (
                self.details.calibration_manifest.digest()
                if self.details.calibration_manifest is not None else None
            )
            if self.calibration != expected_calibration:
                raise ValueError("calibration identity differs from its embedded manifest")
        if self.sparsity_level is not None and not 0 <= self.sparsity_level <= 1:
            raise ValueError("sparsity_level must lie in [0, 1]")
        if "pruning" in self.method:
            if self.sparsity_level is None or not self.sparsity_pattern:
                raise ValueError("pruning requires sparsity_level and sparsity_pattern")
        if self.kernel_support is not None and not self.kernel_support_method:
            raise ValueError("kernel support requires detection/declaration provenance")
        if self.parameters is not None:
            positive(self.parameters, "parameters", zero=True)
        if self.flops is not None:
            positive(self.flops, "flops", zero=True)
        if self.parameters is not None or self.flops is not None:
            if not self.counting_method or not self.counting_input_shape:
                raise ValueError("nominal metrics require counting method and input shape")
            for dimension in self.counting_input_shape:
                positive(dimension, "counting input dimension")

    def sparsity_label(self) -> str:
        if "pruning" not in self.method:
            return "not pruned"
        support = {True: "supported", False: "without kernel support", None: "unknown"}
        return f"{self.sparsity_pattern}; kernel support: {support[self.kernel_support]}"
