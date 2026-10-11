"""Power samples and auditable energy integration in watts, seconds, joules."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Literal

from edge_ai.utils.serialization import nonempty, positive, validate_types


@dataclass(frozen=True)
class PowerMetadata:
    source: str
    measured: bool
    sampling_hz: float
    rails: tuple[str, ...]
    resolution_w: float | None
    clock: Literal["monotonic", "synthetic"]
    description: str

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.source, "power source")
        nonempty(self.description, "power description")
        positive(self.sampling_hz, "sampling_hz")
        if not self.rails or len(set(self.rails)) != len(self.rails):
            raise ValueError("power rails must be nonempty and unique")
        for rail in self.rails:
            nonempty(rail, "rail")
        if self.resolution_w is not None:
            positive(self.resolution_w, "resolution_w")
        if self.measured and self.clock != "monotonic":
            raise ValueError("measured samples require a monotonic measurement clock")


@dataclass(frozen=True)
class PowerTrace:
    metadata: PowerMetadata
    timestamps_s: tuple[float, ...]
    watts: dict[str, tuple[float, ...]]
    missing_samples: int = 0

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.missing_samples, "missing_samples", zero=True)
        if len(self.timestamps_s) < 2:
            raise ValueError("power trace requires at least two samples")
        previous = -1.0
        for timestamp in self.timestamps_s:
            positive(timestamp, "power timestamp", zero=True)
            if timestamp <= previous:
                raise ValueError("power timestamps must increase strictly")
            previous = timestamp
        if set(self.watts) != set(self.metadata.rails):
            raise ValueError("samples and declared power rails differ")
        for samples in self.watts.values():
            if len(samples) != len(self.timestamps_s):
                raise ValueError("power sample counts differ from timestamps")
            for sample in samples:
                positive(sample, "power sample", zero=True)


def integrate_trace(trace: PowerTrace, rail: str, start_s: float, end_s: float) -> float:
    """Integrate one rail, interpolating window edges without extrapolation.

    Selecting a rail explicitly avoids adding aggregate and constituent rails.
    Traces with known missing samples are refused rather than silently bridged.
    """
    trace.__post_init__()
    positive(start_s, "window start", zero=True)
    positive(end_s, "window end")
    if rail not in trace.watts:
        raise ValueError("selected power rail is absent")
    times, samples = trace.timestamps_s, trace.watts[rail]
    if start_s >= end_s or start_s < times[0] or end_s > times[-1]:
        raise ValueError("energy window must be positive and covered by the trace")
    if trace.missing_samples:
        raise ValueError("cannot integrate a trace with missing samples without a gap policy")

    def power_at(timestamp: float) -> float:
        index = min(bisect_right(times, timestamp) - 1, len(times) - 2)
        fraction = (timestamp - times[index]) / (times[index + 1] - times[index])
        return samples[index] + fraction * (samples[index + 1] - samples[index])

    points = (start_s, *(time for time in times if start_s < time < end_s), end_s)
    energy = sum(
        (right - left) * (power_at(left) + power_at(right)) / 2
        for left, right in zip(points, points[1:])
    )
    positive(energy, "integrated energy", zero=True)
    return energy


@dataclass(frozen=True)
class EnergyResult:
    rail: str
    window_start_s: float
    window_end_s: float
    sampling_hz: float
    energy_j: float | None = None
    energy_estimate_j: float | None = None
    integration_method: str | None = None
    estimate_method: str | None = None
    inference_count: int | None = None
    per_inference_j: float | None = None
    per_inference_estimate_j: float | None = None
    per_inference_method: str | None = None
    idle_power_w: float | None = None

    def __post_init__(self) -> None:
        validate_types(self)
        nonempty(self.rail, "energy rail")
        positive(self.sampling_hz, "energy sampling_hz")
        positive(self.window_start_s, "energy window start", zero=True)
        positive(self.window_end_s, "energy window end")
        if self.window_end_s <= self.window_start_s:
            raise ValueError("energy window must have positive duration")
        if (self.energy_j is None) == (self.energy_estimate_j is None):
            raise ValueError("choose exactly one of measured or estimated total energy")
        measured = self.energy_j is not None
        method = self.integration_method if measured else self.estimate_method
        if not method or not method.strip():
            raise ValueError("energy requires its integration/estimate method")
        if (measured and self.estimate_method) or (not measured and self.integration_method):
            raise ValueError("energy method is in the wrong provenance field")
        for name in ("energy_j", "energy_estimate_j", "per_inference_j",
                     "per_inference_estimate_j", "idle_power_w"):
            value = getattr(self, name)
            if value is not None:
                positive(value, name, zero=True)
        if self.inference_count is None:
            if any(value is not None for value in (
                self.per_inference_j, self.per_inference_estimate_j, self.per_inference_method,
            )):
                raise ValueError("per-inference energy requires inference_count")
        else:
            positive(self.inference_count, "energy inference_count")
            average = self.per_inference_j if measured else self.per_inference_estimate_j
            wrong_average = self.per_inference_estimate_j if measured else self.per_inference_j
            total = self.energy_j if measured else self.energy_estimate_j
            if average is None or wrong_average is not None or not self.per_inference_method:
                raise ValueError("per-inference energy must retain the total energy provenance")
            if not math.isclose(average * self.inference_count, total, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("per-inference energy is inconsistent with total energy")


def energy_from_trace(
    trace: PowerTrace, *, rail: str, start_s: float, end_s: float,
    inference_count: int | None = None, idle_power_w: float | None = None,
) -> EnergyResult:
    total = integrate_trace(trace, rail, start_s, end_s)
    measured = trace.metadata.measured
    values = {
        "energy_j" if measured else "energy_estimate_j": total,
        "integration_method" if measured else "estimate_method":
            "trapezoidal" if measured else "synthetic_trace_trapezoidal",
    }
    if inference_count is not None:
        positive(inference_count, "inference_count")
        values.update({
            "per_inference_j" if measured else "per_inference_estimate_j":
                total / inference_count,
            "per_inference_method": "window_mean_power_times_duration_divided_by_inferences",
        })
    return EnergyResult(
        rail=rail, window_start_s=start_s, window_end_s=end_s,
        sampling_hz=trace.metadata.sampling_hz, inference_count=inference_count,
        idle_power_w=idle_power_w, **values,
    )
