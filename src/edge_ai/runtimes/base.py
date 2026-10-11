"""Small runtime contract and memory observations, independent of a backend."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from edge_ai.models.variant import ModelVariant
from edge_ai.utils.serialization import positive, validate_types


@dataclass(frozen=True)
class MemoryReport:
    model_file_bytes: int | None = None
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    runtime_allocation_bytes: int | None = None
    measured: bool = False
    method: str = "unavailable"

    def __post_init__(self) -> None:
        validate_types(self)
        if not self.method.strip():
            raise ValueError("memory report requires method")
        for name in ("model_file_bytes", "peak_ram_bytes", "peak_vram_bytes",
                     "runtime_allocation_bytes"):
            value = getattr(self, name)
            if value is not None:
                positive(value, name, zero=True)


@runtime_checkable
class RuntimeBackend(Protocol):
    def load(self, variant: ModelVariant) -> None: ...

    def bind_batch(self, batch: Any) -> None: ...

    def warmup(self, n: int) -> None: ...

    def infer(self, batch: Any) -> Any: ...

    def memory_report(self) -> MemoryReport: ...

    def release(self) -> None: ...
