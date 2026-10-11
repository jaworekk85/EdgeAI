"""Serializable model identity, without loading weights."""

from dataclasses import dataclass

from edge_ai.compression.spec import CompressionSpec
from edge_ai.utils.serialization import nonempty, resolution, safe_name, validate_types


@dataclass(frozen=True)
class ModelVariant:
    id: str
    family: str
    task: str
    weights_source: str
    weights_digest: str
    precision: str
    compression: CompressionSpec
    resolution: tuple[int, int]
    runtime: str

    def __post_init__(self) -> None:
        validate_types(self)
        safe_name(self.id, "variant id")
        for name in ("family", "task", "weights_source", "weights_digest", "precision", "runtime"):
            nonempty(getattr(self, name), name)
        resolution(self.resolution)
        if self.precision != self.compression.precision:
            raise ValueError("model precision and compression precision must agree")
