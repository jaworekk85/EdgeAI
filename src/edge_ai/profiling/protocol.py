"""Measurement settings, not an implemented profiling runner."""

from dataclasses import dataclass

from edge_ai.utils.serialization import positive, validate_types


@dataclass(frozen=True)
class ProfilingProtocol:
    warmup_iterations: int = 2
    iterations: int = 10
    repetitions: int = 1
    sustained_duration_s: float | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.warmup_iterations, "warmup_iterations", zero=True)
        positive(self.iterations, "iterations")
        positive(self.repetitions, "repetitions")
        if self.sustained_duration_s is not None:
            positive(self.sustained_duration_s, "sustained_duration_s")
