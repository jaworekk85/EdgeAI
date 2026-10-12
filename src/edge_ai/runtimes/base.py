"""Small runtime contract and memory observations, independent of a backend."""

from dataclasses import dataclass, field
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
    sampling_interval_s: float | None = None
    sample_times_s: tuple[float, ...] = ()
    rss_bytes: tuple[int, ...] = ()
    unavailable: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_types(self)
        if not self.method.strip():
            raise ValueError("memory report requires method")
        for name in (
            "model_file_bytes",
            "peak_ram_bytes",
            "peak_vram_bytes",
            "runtime_allocation_bytes",
        ):
            value = getattr(self, name)
            if value is not None:
                positive(value, name, zero=True)
        if self.sampling_interval_s is not None:
            positive(self.sampling_interval_s, "memory sampling interval")
        if len(self.sample_times_s) != len(self.rss_bytes):
            raise ValueError("memory sampling arrays differ")
        previous = -1.0
        for timestamp, value in zip(self.sample_times_s, self.rss_bytes):
            positive(timestamp, "memory sample time", zero=True)
            positive(value, "RSS bytes", zero=True)
            if timestamp <= previous:
                raise ValueError("memory sample timestamps must increase")
            previous = timestamp
        if self.rss_bytes and self.peak_ram_bytes != max(self.rss_bytes):
            raise ValueError("RSS peak differs from raw samples")


@runtime_checkable
class RuntimeBackend(Protocol):
    def load(self, variant: ModelVariant) -> None: ...

    def bind_batch(self, batch: Any) -> None: ...

    def prepare_batch(self, batch: Any) -> Any: ...

    def synchronize(self) -> None: ...

    def warmup(self, n: int) -> None: ...

    def infer(self, batch: Any) -> Any: ...

    def memory_report(self) -> MemoryReport: ...

    def release(self) -> None: ...
