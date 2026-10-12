"""Modeled constant power over the observed clock window, always unmeasured."""

from math import floor
from time import perf_counter
from typing import Callable

from edge_ai.power.trace import PowerMetadata, PowerTrace
from edge_ai.utils.serialization import positive


class WindowMockPower:
    def __init__(self, watts: float, sampling_hz: float, *, clock: Callable = perf_counter):
        positive(watts, "assumed watts", zero=True)
        positive(sampling_hz, "sampling_hz")
        self.watts, self.clock = watts, clock
        self.metadata = PowerMetadata(
            "constant_mock",
            False,
            sampling_hz,
            ("total",),
            None,
            "monotonic",
            "Assumed constant watts on observed window; synthetic values, no physical sensor",
        )
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError("power source already started")
        self._start = self.clock()

    def stop(self):
        if self._start is None:
            raise RuntimeError("power source not started")
        start, end = self._start, self.clock()
        self._start = None
        if end <= start:
            raise ValueError("measurement window must have positive duration")
        count = floor((end - start) * self.metadata.sampling_hz)
        times = [start + index / self.metadata.sampling_hz for index in range(count + 1)]
        if times[-1] < end:
            times.append(end)
        return PowerTrace(self.metadata, tuple(times), {"total": (self.watts,) * len(times)})
