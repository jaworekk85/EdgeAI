"""Measurement settings shared by short and sustained CPU runs."""

from dataclasses import dataclass

from edge_ai.utils.serialization import positive, validate_types


@dataclass(frozen=True)
class ProfilingProtocol:
    warmup_iterations: int = 2
    iterations: int = 10
    repetitions: int = 1
    sustained_duration_s: float | None = None
    sustained_max_iterations: int = 100000

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.warmup_iterations, "warmup_iterations", zero=True)
        positive(self.iterations, "iterations")
        positive(self.repetitions, "repetitions")
        positive(self.sustained_max_iterations, "sustained_max_iterations")
        if self.sustained_duration_s is not None:
            positive(self.sustained_duration_s, "sustained_duration_s")
            if self.sustained_max_iterations < self.iterations * self.repetitions:
                raise ValueError("sustained cap must permit a complete workload pass")
