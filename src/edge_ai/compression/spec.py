"""Compression metadata; transformations and compute accounting arrive later."""

from dataclasses import dataclass
from typing import Literal

from edge_ai.utils.serialization import nonempty, positive, validate_types


@dataclass(frozen=True)
class CompressionSpec:
    method: Literal[
        "none", "quantization", "structured_pruning", "unstructured_pruning",
        "distillation", "smaller_architecture",
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

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.precision, "precision")
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
