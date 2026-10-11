"""CPU-only contract mock; outputs and memory estimates are synthetic."""

from __future__ import annotations

from random import Random
from typing import Any

from edge_ai.models.variant import ModelVariant
from edge_ai.runtimes.base import MemoryReport
from edge_ai.utils.serialization import finite, positive


class MockRuntime:
    def __init__(self, *, seed: int = 0) -> None:
        positive(seed, "seed", zero=True)
        if type(seed) is not int:
            raise ValueError("seed must be integer")
        self.seed = seed
        self.variant: ModelVariant | None = None
        self._batch: tuple[float, ...] | None = None
        self.inference_calls = 0
        self.warmup_calls = 0

    def load(self, variant: ModelVariant) -> None:
        if self.variant is not None:
            raise RuntimeError("release the loaded model before loading another")
        variant.__post_init__()
        if variant.runtime != "mock":
            raise ValueError("MockRuntime requires runtime='mock'")
        self.variant = variant
        self.inference_calls = self.warmup_calls = 0

    def _require_loaded(self) -> None:
        if self.variant is None:
            raise RuntimeError("no model loaded")

    @staticmethod
    def _validate_batch(batch: Any) -> tuple[float, ...]:
        if not isinstance(batch, (list, tuple)) or not batch:
            raise ValueError("mock batch must be a nonempty sequence of finite numbers")
        for value in batch:
            finite(value, "mock input")
        return tuple(float(value) for value in batch)

    def bind_batch(self, batch: Any) -> None:
        self._require_loaded()
        self._batch = self._validate_batch(batch)

    def warmup(self, n: int) -> None:
        self._require_loaded()
        positive(n, "warmup iterations", zero=True)
        if type(n) is not int:
            raise ValueError("warmup iterations must be integer")
        if self._batch is None:
            raise RuntimeError("bind a representative batch before warmup")
        for _ in range(n):
            self._output(self._batch)
            self.warmup_calls += 1

    def _output(self, batch: tuple[float, ...]) -> tuple[float, ...]:
        random = Random(self.seed)
        return tuple(value + random.random() for value in batch)

    def infer(self, batch: Any) -> tuple[float, ...]:
        self._require_loaded()
        values = self._validate_batch(batch)
        output = self._output(values)
        self.inference_calls += 1
        return output

    def memory_report(self) -> MemoryReport:
        self._require_loaded()
        return MemoryReport(method="mock; memory is not measured")

    def release(self) -> None:
        self.variant = None
        self._batch = None
