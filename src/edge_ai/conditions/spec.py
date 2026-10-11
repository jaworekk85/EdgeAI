"""Condition metadata only; seeded transformations belong to Phase 6."""

from dataclasses import dataclass, field
from typing import Any

from edge_ai.utils.serialization import nonempty, positive, validate_types


@dataclass(frozen=True)
class Condition:
    kind: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)
    schedule: tuple[tuple[float, dict[str, Any]], ...] = ()

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.kind, "condition kind")
        positive(self.seed, "condition seed", zero=True)
        previous = -1.0
        for timestamp, _ in self.schedule:
            positive(timestamp, "schedule timestamp", zero=True)
            if timestamp <= previous:
                raise ValueError("schedule timestamps must increase strictly")
            previous = timestamp
