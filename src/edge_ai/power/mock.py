"""Deterministic synthetic power. Its clock is unrelated to wall-clock runtime."""

from random import Random

from edge_ai.power.trace import PowerMetadata, PowerTrace
from edge_ai.utils.serialization import positive


class MockPower:
    def __init__(
        self, *, seed: int = 0, watts: float = 5.0, noise_w: float = 0.0,
        sampling_hz: float = 10.0, samples: int = 11,
    ) -> None:
        positive(seed, "seed", zero=True)
        positive(watts, "watts", zero=True)
        positive(noise_w, "noise_w", zero=True)
        positive(sampling_hz, "sampling_hz")
        if type(seed) is not int or type(samples) is not int or samples < 2:
            raise ValueError("seed must be integer and samples must be integer >= 2")
        if noise_w > watts:
            raise ValueError("noise must not produce negative power")
        self.seed, self.watts, self.noise_w, self.samples = seed, watts, noise_w, samples
        self._metadata = PowerMetadata(
            source="mock", measured=False, sampling_hz=sampling_hz,
            rails=("total",), resolution_w=None, clock="synthetic",
            description="Synthetic samples; resolution is not a sensor observation",
        )
        self._started = False

    @property
    def metadata(self) -> PowerMetadata:
        return self._metadata

    def start(self) -> None:
        if self._started:
            raise RuntimeError("power source already started")
        self._started = True

    def stop(self) -> PowerTrace:
        if not self._started:
            raise RuntimeError("power source not started")
        self._started = False
        random = Random(self.seed)
        return PowerTrace(
            metadata=self.metadata,
            timestamps_s=tuple(index / self.metadata.sampling_hz for index in range(self.samples)),
            watts={"total": tuple(
                self.watts + random.uniform(-self.noise_w, self.noise_w)
                for _ in range(self.samples)
            )},
        )
