"""Explicitly synthetic offline telemetry replay, never a live power source."""

from __future__ import annotations

from dataclasses import dataclass

from edge_ai.devices.tegrastats import TegrastatsSample
from edge_ai.power.trace import PowerMetadata, PowerTrace
from edge_ai.utils.serialization import nonempty, positive, validate_types


@dataclass(frozen=True)
class ReplaySample:
    timestamp_s: float
    telemetry: TegrastatsSample

    def __post_init__(self) -> None:
        validate_types(self)
        positive(self.timestamp_s, "relative replay timestamp", zero=True)


def replay_power(
    samples: tuple[ReplaySample, ...], *, rail: str, sampling_hz: float, description: str,
) -> PowerTrace:
    """Convert a complete, regularly sampled selected rail to synthetic watts.

    Caller supplies relative timestamps. Refuse missing/irregular samples rather
    than interpolating across unknown gaps. Never sum aggregate/constituent rails.
    """
    nonempty(rail, "replay rail")
    nonempty(description, "replay source description")
    positive(sampling_hz, "replay sampling_hz")
    if len(samples) < 2:
        raise ValueError("replay requires at least two timestamped samples")
    interval = 1 / sampling_hz
    previous = None
    watts = []
    for sample in samples:
        sample.__post_init__()
        if previous is not None and abs(sample.timestamp_s - previous - interval) > interval * 0.01:
            raise ValueError("replay timestamps are irregular or contain missing samples")
        if rail not in sample.telemetry.rails:
            raise ValueError("selected rail must be present in every replay sample")
        watts.append(sample.telemetry.rails[rail].instantaneous_w)
        previous = sample.timestamp_s
    return PowerTrace(
        PowerMetadata(
            "tegrastats_replay", False, sampling_hz, (rail,), None, "synthetic",
            f"Offline replay; not synchronized with current inference: {description}",
        ),
        tuple(sample.timestamp_s for sample in samples), {rail: tuple(watts)},
    )
