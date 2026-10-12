"""Workload metadata; dataset readers belong to Phase 2."""

from dataclasses import dataclass
from typing import Literal

from edge_ai.conditions.spec import Condition
from edge_ai.datasets.preprocessing import PreprocessingSpec
from edge_ai.utils.serialization import nonempty, positive, resolution, validate_types


@dataclass(frozen=True)
class Workload:
    task: str
    source: Literal["dataset", "video", "camera", "synthetic"]
    source_id: str
    resolution: tuple[int, int]
    batch_size: int
    privacy: Literal["public", "private"] = "public"
    split: str | None = None
    subset: str | None = None
    frame_rate: float | None = None
    latency_constraint_s: float | None = None
    conditions: tuple[Condition, ...] = ()
    preprocessing: PreprocessingSpec | None = None
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.task, "workload task")
        nonempty(self.source_id, "source_id")
        resolution(self.resolution)
        if self.preprocessing is not None and self.preprocessing.crop != self.resolution:
            raise ValueError("preprocessing crop differs from workload resolution")
        if self.manifest_sha256 is not None:
            from edge_ai.utils.artifacts import validate_digest

            validate_digest(self.manifest_sha256)
        positive(self.batch_size, "batch_size")
        for name in ("frame_rate", "latency_constraint_s"):
            value = getattr(self, name)
            if value is not None:
                positive(value, name)
